import torch
import torch.nn as nn
import numpy as np
from models.attention import Seq_Transformer
from models.grn import gated_residual_network
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TFCC(nn.Module):
    def __init__(self, model_params, device, static_use):
        super(TFCC, self).__init__()

        params = dict(model_params)
        self.output_dim = int(params['encoder_output_dim'])
        self.timestep = int(params['timestep'])
        self.Wk = nn.ModuleList([nn.Linear(int(params['hidden_dim']), self.output_dim) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Linear(int(params['hidden_dim']), int(params['encoder_output_dim']) // 2),
            nn.BatchNorm1d(int(params['encoder_output_dim']) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(params['encoder_output_dim']) // 2, int(params['encoder_output_dim']) // 4)
        )
        self.seq_transformer = Seq_Transformer(patch_size=self.output_dim, dim=int(params['hidden_dim']), depth=4,
                                               heads=4, mlp_dim=64)
        self.static_use = static_use
        self.seq_len = int(params["static_feature_len"])
        if self.static_use:
            self.grn_list = nn.ModuleList()
            for i in range(self.seq_len):
                grn = gated_residual_network(input_dim=self.output_dim,
                                             hidden_dim=self.output_dim,
                                             additional_context=True)
                self.grn_list.append(grn)

    def forward(self, feature_aug1, feature_aug2, static_info=None):
        seq_len = feature_aug1.shape[2]

        if self.static_use:
            enriched_feature_augs1 = torch.empty(feature_aug1.shape).float().to(self.device)

            for i in range(seq_len):
                enriched_feature_augs1[:, :, i] = self.grn_list[i](feature_aug1[:, :, i], static_info)
            feature_aug1 = enriched_feature_augs1

        z_aug1 = feature_aug1  # (batch_size, channels, seq_len)
        z_aug1 = torch.permute(z_aug1, (0, 2, 1)).contiguous()

        z_aug2 = feature_aug2
        z_aug2 = torch.permute(z_aug2, (0, 2, 1)).contiguous()

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)

        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.output_dim)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.output_dim)
        forward_seq = z_aug1[:, :t_samples + 1, :]  # transformer input value
        c_t = self.seq_transformer(forward_seq)
        pred = torch.empty((self.timestep, batch, self.output_dim)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)
        
class AR_Model(nn.Module):
    def __init__(self, model_params, device, static_use):
        super(AR_Model, self).__init__()

        params = dict(model_params)
        self.output_dim = int(params['encoder_output_dim'])
        self.timestep = int(params['timestep'])
        self.Wk = nn.ModuleList([nn.Linear(int(params['hidden_dim']), self.output_dim) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

        self.projection_head = nn.Sequential(
            nn.Linear(int(params['encoder_output_dim']), int(params['encoder_output_dim']) // 2),
            nn.ReLU(),
            nn.Linear(int(params['encoder_output_dim'])//2, int(params['encoder_output_dim'])//4),
            nn.ReLU(),
            nn.Linear(int(params['encoder_output_dim'])//4, int(params['encoder_output_dim']) // 8)
           
        ) # create_linear_model_from_base_model


    def forward(self, feature_aug1, feature_aug2, static_info=None):


        z_aug = feature_aug1  # (batch_size, channels, seq_len)
        z_aug = torch.permute(z_aug, (0, 2, 1)).contiguous()



        batch = z_aug.shape[0]

        return 0, self.projection_head(z_aug)



class TF_encoder(nn.Module):
    def __init__(self, model_params, device, static_use):
        super(TF_encoder, self).__init__()

        params = dict(model_params)
        self.output_dim = int(params['encoder_output_dim'])
        self.timestep = int(params['timestep'])
        self.feature_len = int(params['feature_len'])
        self.Wk = nn.ModuleList([nn.Linear(int(params['hidden_dim']), self.output_dim) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.device = device

        model_output_dim = self.feature_len * self.output_dim

        self.pos_encoding = PositionalEncoding(self.output_dim)

        self.flatten = nn.Flatten()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.output_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)

        self.projection_head = nn.Sequential(
            nn.Linear(model_output_dim, model_output_dim//2),
            nn.ReLU(),
            nn.Linear(model_output_dim//2, int(params['encoder_output_dim']) // 4)
        )# create_linear_model_from_base_model


    def forward(self, feature_aug1, feature_aug2, static_info=None):

        z_aug = feature_aug1  
        z_aug = torch.permute(z_aug, (2, 0, 1)).contiguous()

        pos_z_aug = self.pos_encoding(z_aug)

        pos_z_aug = self.transformer_encoder(pos_z_aug)

        pos_z_aug = torch.permute(pos_z_aug, (1, 2, 0)).contiguous()

        pos_z_aug = self.flatten(pos_z_aug)


        return 0, self.projection_head(pos_z_aug)