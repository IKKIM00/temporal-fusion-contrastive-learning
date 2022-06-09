import os
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from models.loss import NTXentLoss

import warnings
warnings.filterwarnings('always')


class SSL(pl.LightningModule):
    def __init__(self, model_type, encoder, autoregressive, static_encoder, static_use,
                 loss_params, lr, batch_size):
        """
        Train self-supervised learning method

        :param model_type:
        :param encoder:
        :param autoregressive:
        :param static_encoder:
        :param static_use:
        :param loss_params:
        :param lr:
        :param batch_size:
        :param criterion:
        """
        super(SSL, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.batch_size = batch_size
        self.model_type = model_type
        self.static_use = static_use

        self.encoder = encoder
        self.autoregressive = autoregressive
        if self.static_use:
            self.static_encoder = static_encoder

        self.loss_parmas = dict(loss_params)

    def configure_optimizers(self):
        encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        ar_optim = torch.optim.Adam(self.autoregressive.parameters(), lr=self.lr)
        if self.static_use:
            static_optim = torch.optim.Adam(self.static_encoder.parameters(), lr=self.lr)
            return encoder_optim, ar_optim, static_optim
        return encoder_optim, ar_optim

    def info_xtnex_loss(self, batch, mode='train'):
        obs_real, labels, aug1, aug2, static = batch
        obs_real, aug1, aug2, labels = obs_real.float(), aug1.float(), aug2.float(), labels.long()

        nt_xent_criterion = NTXentLoss(obs_real.device, obs_real.shape[0],
                                       float(self.loss_parmas['temperature']),
                                       bool(self.loss_parmas['use_cosine_similarity']))
        # create static encoder and static output
        if self.static_use and self.model_type == 'TFCL':
            static_context_variable, static_context_enrichment = self.static_encoder(static)
            features1 = self.encoder(aug1, static_context_variable)
            features2 = self.encoder(aug2, static_context_variable)
        elif self.model_type in ['TFCL', 'SimclrHAR', 'CSSHAR']:
            features1 = self.encoder(aug1)
            features2 = self.encoder(aug2)
        elif self.model_type == 'CPCHAR':
            feature = self.encoder(obs_real)

        if self.model_type == 'TFCL':
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

        if self.static_use:
            features1 = torch.cat([features1, static_context_enrichment.unsqueeze(-1)], dim=2).to(features1.device)
            features2 = torch.cat([features2, static_context_enrichment.unsqueeze(-1)], dim=2).to(features2.device)

        if self.model_type == 'TFCL':
            pred_cont_loss1, temp_cont_feat1 = self.autoregressive(features1, features2)
            pred_cont_loss2, temp_cont_feat2 = self.autoregressive(features2, features1)

            zis = temp_cont_feat1
            zjs = temp_cont_feat2

            lambda1 = float(self.loss_parmas['lambda1'])
            lambda2 = float(self.loss_parmas['lambda2'])
            
            loss = lambda1 * (pred_cont_loss1 + pred_cont_loss2) + lambda2 * nt_xent_criterion(zis, zjs)
        elif self.model_type in ['SimclrHAR', 'CSSHAR']:
            projection1 = self.auto_regressive(features1)
            projection2 = self.auto_regressive(features2)

            loss = nt_xent_criterion(projection1, projection2)
        elif self.model_type == 'CPCHAR':
            loss, c_t = self.auto_regressive(feature)

        self.log(f"{mode}_loss", loss.item())
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss = self.info_xtnex_loss(batch)
        return loss


class DownstreamTask(pl.LightningModule):
    def __init__(self, model_type, training_mode, encoder, static_encoder, logits, static_use, criterion, lr,
                 autoregressive=None):
        super(DownstreamTask, self).__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.training_mode = training_mode
        self.criterion = criterion
        self.lr = lr
        self.static_use = static_use

        self.encoder = encoder
        self.autoregressive = autoregressive
        self.logits = logits
        if self.static_use:
            self.static_encoder = static_encoder

    def configure_optimizers(self):
        logits_optim = torch.optim.Adam(self.logits.parameters(), lr=self.lr)
        optim_list = [logits_optim]

        encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        optim_list.append(encoder_optim)

        if self.static_use:
            static_encoder_optim = torch.optim.Adam(self.static_encoder.parameters(), lr=self.lr)
            optim_list.append(static_encoder_optim)
        return optim_list
    
    def forward(self, obs_real, static=None):
        if self.static_use and self.model_type == 'TFCL':
            static_context_variable, static_context_enrichment = self.static_encoder(static)
            output = self.encoder(obs_real, static_context_enrichment)
        elif self.model_type in ['TFCL', 'SimclrHAR', 'CSSHAR']:
            output = self.encoder(obs_real)
        elif self.model_type == 'CPCHAR':
            output = self.encoder
            cpc_loss, output = self.autoregressive(output)
        output = self.logits(output)
        return output

    def _forward(self, batch, mode='train'):
        obs_real, labels, aug1, aug2, static = batch
        obs_real, aug1, aug2, labels = obs_real.float(), aug1.float(), aug2.float(), labels.long()
        output = self.forward(obs_real, static)

        loss = self.criterion(output, labels)
        acc = labels.eq(output.detach().argmax(dim=1)).float().mean()

        self.log(f"{mode}_loss", loss.item())
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, acc = self._forward(batch, mode='train')
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._forward(batch, mode='val')
        return loss, acc

    def test_step(self, batch, batch_idx):
        loss, acc = self._forward(batch, mode='test')
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        obs_real, labels, aug1, aug2, static = batch
        obs_real, aug1, aug2, labels = obs_real.float(), aug1.float(), aug2.float(), labels.long()
        return self(obs_real, static).argmax(dim=1), labels


class Supervised(pl.LightningModule):
    def __init__(self, model_type, encoder, static_encoder, logits, static_use, criterion, lr):
        super(Supervised, self).__init__()
        self.save_hyperparameters()

        self.static_use = static_use
        self.criterion = criterion
        self.lr = lr

        self.model_type = model_type
        self.encoder = encoder
        if self.static_use:
            self.static_encoder = static_encoder
        self.logits = logits

    def configure_optimizers(self):
        encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        logits_optim = torch.optim.Adam(self.logits.parameters(), lr=self.lr)
        optim_list = [encoder_optim, logits_optim]
        if self.static_use:
            static_encoder_optim = torch.optim.Adam(self.static_encoder.parameters(), lr=self.lr)
            optim_list.append(static_encoder_optim)
        return optim_list

    def forward(self, obs_real, static=None):
        if self.static_use and self.model_type == 'TFCL':
            static_context_variable, static_context_enrichment = self.static_encoder(static)
            output = self.encoder(obs_real, static_context_enrichment)
        elif self.model_type in ['TFCL', 'SimclrHAR', 'CSSHAR']:
            output = self.encoder(obs_real)
        output = self.logits(output)
        return output

    def _forward(self, batch, mode='train'):
        obs_real, labels, aug1, aug2, static = batch
        obs_real, aug1, aug2, labels = obs_real.float(), aug1.float(), aug2.float(), labels.long()
        output = self.forward(obs_real, static)

        loss = self.criterion(output, labels)
        acc = labels.eq(output.detach().argmax(dim=1)).float().mean()

        self.log(f"{mode}_loss", loss.item())
        self.log(f"{mode}_acc", acc)
        return loss, acc

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss, acc = self._forward(batch, mode='train')
        return {'loss': loss, 'acc': acc}

    def validation_step(self, batch, batch_idx):
        loss, acc = self._forward(batch, mode='val')
        return loss, acc

    def test_step(self, batch, batch_idx):
        loss, acc = self._forward(batch, mode='test')
        return loss, acc

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        obs_real, labels, aug1, aug2, static = batch
        obs_real, aug1, aug2, labels = obs_real.float(), aug1.float(), aug2.float(), labels.long()
        return self(obs_real, static).argmax(dim=1), labels


def train_ssl(train_loader, model, checkpoint_dir, gpus, max_epochs=300, restart=True):
    checkpoint_callback = ModelCheckpoint(
                dirpath=os.path.join(checkpoint_dir, "saved_models"),
                filename='ckp_last',
                auto_insert_metric_name=False,
                monitor='train_loss',
                mode='min',
                save_weights_only=False
    )

    pretrained_filename = os.path.join(checkpoint_dir, "saved_models", "ckp_last.ckpt")
    if restart:
        trainer = pl.Trainer(
            default_root_dir=os.path.join(checkpoint_dir, "saved_models"),
            accelerator='gpu',
            strategy='dp',
            devices=gpus,
            max_epochs=max_epochs,
            num_nodes=4,
            callbacks=[checkpoint_callback]
        )
    else:
        trainer = pl.Trainer(
            default_root_dir=os.path.join(checkpoint_dir, "saved_models"),
            accelerator='gpu',
            strategy='dp',
            devices=gpus,
            num_nodes=4,
            max_epochs=max_epochs,
            callbacks=[checkpoint_callback],
            resume_from_checkpoint=pretrained_filename
        )
    
    trainer.fit(model=model,
                train_dataloaders=train_loader)
    return trainer.checkpoint_callback.best_model_path


def train_downstream_task(train_loader, valid_loader, test_loader, model, checkpoint_dir, training_mode, gpus,
                          max_epochs=300):
    checkpoint_callback = ModelCheckpoint(
                                dirpath=checkpoint_dir,
                                filename=f"{training_mode}_ckpt",
                                auto_insert_metric_name=False,
                                monitor='val_loss',
                                mode='min',
                                save_weights_only=True
    )
    earlystop_callback = EarlyStopping(
                            monitor='val_loss',
                            patience=20,
                            mode='min'
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_dir, "saved_models"),
        accelerator='gpu',
        strategy='dp',
        devices=gpus,
        num_nodes=4,
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, earlystop_callback]
    )
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader)
    best_model = DownstreamTask.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    results = trainer.predict(model=best_model,
                              dataloaders=test_loader,
                              return_predictions=True)
    
    results = list(results)
    trgs, outs = [], []
    for out, trg in results:
        trgs.extend(trg.detach().cpu().numpy())
        outs.extend(out.detach().cpu().numpy())
    
    acc = accuracy_score(trgs, outs)
    precision = precision_score(trgs, outs, average='macro')
    recall = recall_score(trgs, outs, average='macro')
    f1 = f1_score(trgs, outs, average='macro')
    return trgs, outs, acc, precision, recall, f1


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
