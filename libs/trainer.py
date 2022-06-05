import os
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.optim as optim
import torch.nn.functional as F

from models.loss import NTXentLoss

import warnings

warnings.filterwarnings('always')


def Trainer(encoder, logit, autoregressive, static_encoder, method, encoder_optimizer, logit_optimizer, ar_optimizer,
            static_encoder_optimizer, train_loader, valid_loader, test_loader, device, logger,
            loss_params, loss_func, experiment_log_dir, training_mode, batch_size, static_use=True):

    logger.debug("Training started ....")

    params = dict(loss_params)
    best_loss = 99999999999
    train_best_loss = 999999999
    patience = 0
    if training_mode == "self_supervised":
        epochs = 300
    else:
        epochs = int(params['num_epoch'])
    for epoch in range(1, epochs + 1):
        if patience == 20:
            break

        train_loss, train_acc = model_train(logger, encoder, logit, autoregressive, static_encoder, method, batch_size,
                                            encoder_optimizer, logit_optimizer, ar_optimizer, static_encoder_optimizer,
                                            loss_func, train_loader, loss_params, device, training_mode, static_use)
        valid_loss, valid_acc, _, _, _, _, _ = model_evaluate(encoder, logit, autoregressive, static_encoder,
                                                              method, valid_loader, device, training_mode, loss_func, static_use)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        if training_mode == "self_supervised" and train_loss < train_best_loss:
            logger.debug(f'#################### Saving new model ####################')
            train_best_loss = train_loss
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'encoder_model_state_dict': encoder.state_dict(),
                        'ar_model_state_dict': autoregressive.state_dict(),
                        'static_encoder_model_state_dict': static_encoder.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

        if training_mode != "self_supervised" and valid_loss < best_loss:
            logger.debug(f'#################### Saving new model ####################')
            patience = 0
            best_loss = valid_loss
            os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
            chkpoint = {'encoder_model_state_dict': encoder.state_dict(),
                        'ar_model_state_dict': autoregressive.state_dict(),
                        'static_encoder_model_state_dict': static_encoder.state_dict()}
            torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
            best_encoder_model = encoder
            best_logit_model = logit
            best_autoregressive = autoregressive
            best_static_encoder_model = static_encoder
        elif training_mode != "self_supervised" and valid_loss > best_loss:
            patience += 1

    if training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        encoder = best_encoder_model
        logit = best_logit_model
        autoregressive = best_autoregressive
        static_encoder_model = best_static_encoder_model
        test_loss, test_acc, _, _, precision, recall, f1 = model_evaluate(encoder, logit, autoregressive, static_encoder_model,
                                                                          method, test_loader, device,
                                                                          training_mode, loss_func, static_use)
        logger.debug(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}\n'
                     f'Test F1 score    :{f1:0.4f}\t | Test Precision   : {precision:0.4f}\t | Test Recall  : {recall:0.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(logger, encoder, logit, autoregressive, static_encoder, method, batch_size, encoder_optimizer,
                logit_optimizer, ar_optimizer, static_encoder_optimizer, criterion, train_loader, loss_params, device,
                training_mode, static_use):
    total_loss = []
    total_acc = []
    encoder.train()
    logit.train()
    autoregressive.train()
    if static_use:
        static_encoder.train()

    for batch_idx, (observed_real, labels, aug1, aug2, static_input) in enumerate(train_loader):
        observed_real, labels = observed_real.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # print("observed_real : ", observed_real.shape)
        # optimizer
        encoder_optimizer.zero_grad()
        logit_optimizer.zero_grad()
        ar_optimizer.zero_grad()

        if static_use:
            static_encoder_optimizer.zero_grad()
            static_context_variable, static_context_enrichment = static_encoder(static_input.to(device))

        if training_mode == "self_supervised":

            if static_use and method == 'TFCL':

                features1 = encoder(aug1, static_context_variable)
                features2 = encoder(aug2, static_context_variable)

            elif method in ["TFCL", "SimclrHAR", "CSSHAR"]:
                
                features1 = encoder(aug1)
                features2 = encoder(aug2)
            
            else: # CPCHAR
                
                feature = encoder(observed_real)

            if method == "TFCL":

                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

            if static_use:
                features1 = torch.cat([features1, static_context_enrichment.unsqueeze(-1)], dim=2)
                features2 = torch.cat([features2, static_context_enrichment.unsqueeze(-1)], dim=2)

            if method == "TFCL":
                temp_cont_loss1, temp_cont_feat1 = autoregressive(features1, features2)
                temp_cont_loss2, temp_cont_feat2 = autoregressive(features2, features1)

                zis = temp_cont_feat1
                zjs = temp_cont_feat2

            elif method in ["SimclrHAR", "CSSHAR"]:

                projection1 = autoregressive(features1)
                projection2 = autoregressive(features2)

            else: # CPCHAR

                nce, c_t = autoregressive(feature)
                

        else: # Train Linear or Fine-tune

            if static_use and method == "TFCL":
                enc_out = encoder(observed_real, static_context_variable)
                output = logit(enc_out)

            elif method in ["TFCL", "SimclrHAR", "CSSHAR"]:
                enc_out = encoder(observed_real)
                output = logit(enc_out)

            else:
                enc_out = encoder(observed_real)
                cpc_loss, out = autoregressive(enc_out)
                output = logit(out)

        params = dict(loss_params)
        
        if method != 'CPCHAR':
            nt_xent_criterion = NTXentLoss(device, batch_size, float(params['temperature']),
                                       bool(params['use_cosine_similarity']))

        if training_mode == "self_supervised" and method == "TFCL":
            lambda1 = 1.5
            lambda2 = 0.7
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + nt_xent_criterion(zis, zjs) * lambda2

        elif training_mode == "self_supervised" and method in ["SimclrHAR", "CSSHAR"]:
            loss = nt_xent_criterion(projection1, projection2)
        elif training_mode == "self_supervised" and method == "CPCHAR":
            loss = nce

        else:
            prediction = output
            loss = criterion(prediction, labels)
            total_acc.append(labels.eq(prediction.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        encoder_optimizer.step()
        if training_mode != 'self_supervised':
            logit_optimizer.step()
        ar_optimizer.step()
        if static_use:
            static_encoder_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if training_mode == "self_supervised":
        total_acc = 0
    else:
        total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(encoder, logit, autoregressive, static_encoder, method, test_loader, device,
                   training_mode, criterion, static_use):
    encoder.eval()
    logit.eval()
    autoregressive.eval()
    static_encoder.eval()

    total_loss = []
    total_acc = []

    outs = np.array([])
    trgs = np.array([])

    with torch.no_grad():
        for observed_real, labels, _, _, static_input in test_loader:
            data, labels = observed_real.float().to(device), labels.long().to(device)

            if static_use:
                static_context_variable, static_context_enrichment = static_encoder(static_input.to(device))

            if training_mode == "self_supervised":
                pass

            else:

                if static_use and method == 'TFCL':
                    out = encoder(data, static_context_variable)
                    output = logit(out)
                elif method in ["TFCL", "SimclrHAR", "CSSHAR"]:
                    out = encoder(data)
                    output = logit(out)
                else:
                    out = encoder(data)
                    nce, ar_out = autoregressive(out)
                    output = logit(ar_out)


            if training_mode != "self_supervised":
                predictions = output
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
