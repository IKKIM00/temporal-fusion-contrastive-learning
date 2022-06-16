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
            projection1 = self.autoregressive(features1)
            projection2 = self.autoregressive(features2)

            loss = nt_xent_criterion(projection1, projection2)
        elif self.model_type == 'CPCHAR':
            loss, c_t = self.autoregressive(feature)

        self.log(f"{mode}_loss", loss.item())
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        loss = self.info_xtnex_loss(batch)
        return loss


class FineTune(pl.LightningModule):
    def __init__(self, model_type, training_mode, encoder, static_encoder, logits, static_use, criterion, lr,
                 autoregressive=None):
        super(FineTune, self).__init__()
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
    
    def forward(self, obs_real, static=None):
        if self.static_use and self.model_type == 'TFCL':
            static_context_variable, static_context_enrichment = self.static_encoder(static)
            output = self.encoder(obs_real, static_context_enrichment)
        elif self.model_type in ['TFCL', 'SimclrHAR', 'CSSHAR']:
            output = self.encoder(obs_real)
        elif self.model_type == 'CPCHAR':
            output = self.encoder(obs_real)
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
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
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


class TrainLinear(pl.LightningModule):
    def __init__(self, model_type, training_mode, encoder, static_encoder, logits, static_use, criterion, lr,
                 autoregressive=None):
        super(TrainLinear, self).__init__()
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def forward(self, obs_real, static=None):
        if self.static_use and self.model_type == 'TFCL':
            static_context_variable, static_context_enrichment = self.static_encoder(static)
            output = self.encoder(obs_real, static_context_enrichment)
        elif self.model_type in ['TFCL', 'SimclrHAR', 'CSSHAR']:
            output = self.encoder(obs_real)
        elif self.model_type == 'CPCHAR':
            output = self.encoder(obs_real)
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

    def training_step(self, batch, batch_idx):
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
    if training_mode == 'fine_tune':
        best_model = FineTune.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    elif training_mode == 'train_linear':
        best_model = TrainLinear.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    elif training_mode == 'supervised':
        best_model = Supervised.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
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
