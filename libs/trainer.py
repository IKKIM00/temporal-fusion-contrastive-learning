import os
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss

import warnings

warnings.filterwarnings('always')


def Trainer(encoder, tfcc_model, static_embedding_model, static_variable_selection, encoder_model_type, encoder_optimizer, tfcc_optimizer,
            static_embedding_optimizer, static_variable_selection_optimizer, train_loader, valid_loader, test_loader, device, logger,
            loss_params, experiment_log_dir, training_mode, static_use=True):
    logger.debug("Training started ....")

    best_loss = 99999999999
    train_best_loss = 999999999
    patience = 0

    criterion = nn.CrossEntropyLoss()
    params = dict(loss_params)
    for epoch in range(1, int(params['num_epoch']) + 1):
        if patience == 20:
            break
        train_loss, train_acc = model_train(encoder, tfcc_model, static_embedding_model, static_variable_selection, encoder_model_type,
                                            encoder_optimizer, tfcc_optimizer, static_embedding_optimizer, static_variable_selection_optimizer,
                                            criterion, train_loader, loss_params, device, training_mode, static_use)
        valid_loss, valid_acc, _, _, _, _, _ = model_evaluate(encoder, tfcc_model, static_embedding_model,
                                                              static_variable_selection, encoder_model_type, valid_loader, device,
                                                              training_mode, static_use)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        if training_mode == "self_supervised" and train_loss < train_best_loss:
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'model_state_dict': encoder.state_dict(),
                        'temporal_contr_model_state_dict': tfcc_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

        if training_mode != "self_supervised" and valid_loss < best_loss:
            logger.debug(f'#################### Saving new model ####################')
            best_loss = valid_loss
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'model_state_dict': encoder.state_dict(),
                        'temporal_contr_model_state_dict': tfcc_model.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
        else:
            patience += 1

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _, precision, recall, f1 = model_evaluate(encoder, tfcc_model, static_embedding_model,
                                                                          static_variable_selection, encoder_model_type, test_loader, device,
                                                                          training_mode, static_use)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}\n'
                     f'Test F1 score    :{f1:0.4f}\t | Test Precision   : {precision:0.4f}\t | Test Recall  : {recall:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(encoder, tfcc_model, static_embedding_model, static_variable_selection, encoder_model_type, encoder_optimizer,
                tfcc_optimizer, static_embedding_optimizer, static_variable_selection_optimizer, criterion, train_loader, loss_params, device,
                training_mode, static_use):
    total_loss = []
    total_acc = []
    encoder.train()
    tfcc_model.train()
    if static_use:
        static_embedding_model.train()
        static_variable_selection.train()

    for batch_idx, (observed_real, labels, aug1, aug2, static_input) in enumerate(train_loader):
        observed_real, labels = observed_real.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        encoder_optimizer.zero_grad()
        tfcc_optimizer.zero_grad()

        if static_use:
            static_embedding_optimizer.zero_grad()
            static_variable_selection_optimizer.zero_grad()
            static_embedding = static_embedding_model(static_input.to(device))
            static_context_enrichment, static_vec, sparse_weights = static_variable_selection(static_embedding)

        if training_mode == "self_supervised":
            if static_use and encoder_model_type == 'CNN':
                predictions1, features1 = encoder(aug1, static_context_enrichment)
                predictions2, features2 = encoder(aug2, static_context_enrichment)
            elif static_use and encoder_model_type == 'LSTM':
                predictions1, features1 = encoder(aug1, static_vec, static_context_enrichment)
                predictions2, features2 = encoder(aug2, static_vec, static_context_enrichment)
            else:
                predictions1, features1 = encoder(aug1)
                predictions2, features2 = encoder(aug2)

            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = tfcc_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = tfcc_model(features1, features2)

            zis = temp_cont_feat1
            zjs = temp_cont_feat2
        else:
            if static_use and encoder_model_type == 'CNN':
                output = encoder(observed_real, static_vec)
            elif static_use and encoder_model_type == 'LSTM':
                output = encoder(observed_real, static_vec, static_context_enrichment)
            else:
                output = encoder(observed_real)

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
        if static_use:
            static_embedding_optimizer.step()
            static_variable_selection_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(encoder, tfcc_model, static_embedding_model, static_variable_selection, encoder_model_type, test_loader, device,
                   training_mode, static_use):
    encoder.eval()
    tfcc_model.eval()
    static_embedding_model.eval()
    static_variable_selection.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for observed_real, labels, _, _, static_input in test_loader:
            data, labels = observed_real.float().to(device), labels.long().to(device)

            if static_use:
                static_embedding = static_embedding_model(static_input.to(device))
                static_context_enrichment, static_vec, sparse_weights = static_variable_selection(static_embedding)

            if training_mode == "self_supervised":
                pass
            else:
                if static_use and encoder_model_type == 'CNN':
                    output = encoder(data, static_context_enrichment)
                elif static_use and encoder_model_type == 'LSTM':
                    output = encoder(data, static_vec, static_context_enrichment)
                else:
                    output = encoder(data)

            if training_mode != "self_supervised":
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                total_loss.append(loss.item())

                pred = predictions.argmax(dim=1)  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if training_mode != "self_supervised":
        total_loss = torch.tensor(total_loss).mean()  # average loss
    else:
        total_loss = 0
    if training_mode == "self_supervised":
        total_acc = 0
        return total_loss, total_acc, [], [], 0, 0, 0
    else:
        total_acc = torch.tensor(total_acc).mean()  # average acc
        precision = precision_score(trgs, outs, average='macro')
        recall = recall_score(trgs, outs, average='macro')
        f1 = f1_score(trgs, outs, average='macro')
    return total_loss, total_acc, outs, trgs, precision, recall, f1
