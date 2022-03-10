import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss

def Trainer(encoder, tfcc_model, static_vec_model, encoder_optimizer, tfcc_optimizer, static_optimizer, train_loader, valid_loader, test_loader, static_input, device, logger, loss_params, experiment_log_dir, training_mode, static_use):
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min')
    params = dict(loss_params)
    for epoch in range(1, int(params['num_epoch']) + 1):
        train_loss, train_acc = model_train(encoder, tfcc_model, static_vec_model, encoder_optimizer, tfcc_optimizer, static_optimizer, criterion, train_loader, static_input, loss_params, device, training_mode, static_use)
        valid_loss, valid_acc, _, _ =model_evaluate(encoder, tfcc_model, static_vec_model, test_loader, device, training_mode)

        if training_mode != "self_supervised":
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': encoder.state_dict(),
                'temporal_contr_model_state_dict': tfcc_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _ = model_evaluate(encoder, tfcc_model, static_vec_model, test_loader, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    logger.debug("\n################## Training is Done! #########################")

def model_train(encoder, tfcc_model, static_vec_model, encoder_optimizer, tfcc_optimizer, static_optimizer, criterion, train_loader, static_input, loss_params, device, training_mode, static_use):
    total_loss = []
    total_acc = []
    encoder.train()
    tfcc_model.train()
    static_vec_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.float().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        encoder_optimizer.zero_grad()
        tfcc_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = encoder(aug1)
            predictions2, features2 = encoder(aug2)

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            if static_use == True:
                static_optimizer.zero_grad()
                static_vec, sparse_weights = static_vec_model(static_input)
                features1 = torch.cat([features1, static_vec.unsqueeze(-1)], dim=2)
                features2 = torch.cat([features2, static_vec.unsqueeze(-1)], dim=2)

            temp_cont_loss1, temp_cont_feat1 = tfcc_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = tfcc_model(features1, features2)

            zis = temp_cont_feat1
            zjs = temp_cont_feat2
        else:
            output = encoder(data)

        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            params = dict(loss_params)
            nt_xent_criterion = NTXentLoss(device, int(params['batch_size']), float(params['temperature']),
                                           bool(params['use_cosine_similarity']))
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2
        else:
            prediction, features = output
            loss = criterion(prediction, labels)
            total_acc.append(labels.eq(prediction.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        encoder_optimizer.step()
        tfcc_optimizer.step()
        static_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc

def model_evaluate(encoder, tfcc_model, static_vec_model, test_loader, device, training_mode):
    encoder.eval()
    tfcc_model.eval()
    static_vec_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_loader:
            data, labels = data.float().to(device), labels.long().to(device)

            if training_mode == "self_supervised":
                pass
            else:
                output = encoder(data)

            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

            if training_mode != "self_supervised":
                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, outs, trgs