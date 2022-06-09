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


class BaseAR(nn.Module):
    def __init__(self, model_params, static_use):
        super(BaseAR, self).__init__()

        params = dict(model_params)
        self.output_dim = int(params['encoder_output_dim'])
        self.timestep = int(params['timestep'])
        self.Wk = nn.ModuleList([nn.Linear(int(params['hidden_dim']), self.output_dim) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)

        self.projection_head = nn.Sequential(
            nn.Linear(int(params['hidden_dim']), int(params['encoder_output_dim']) // 2),
            nn.BatchNorm1d(int(params['encoder_output_dim']) // 2),
            nn.ReLU(inplace=True),
            nn.Linear(int(params['encoder_output_dim']) // 2, int(params['encoder_output_dim']) // 4)
        )
        self.seq_transformer = Seq_Transformer(patch_size=self.output_dim, dim=int(params['hidden_dim']), depth=4,
                                               heads=4, mlp_dim=64)
        self.static_use = static_use
        if self.static_use:
            self.grn = gated_residual_network(input_dim=self.output_dim,
                                              hidden_dim=self.output_dim,
                                              additional_context=True)

    def forward(self, feature_aug1, feature_aug2, static_info=None):
        seq_len = feature_aug1.shape[2]

        if self.static_use:
            enriched_feature_augs1 = torch.empty(feature_aug1.shape).float()

            for i in range(seq_len):
                enriched_feature_augs1[:, :, i] = self.grn(feature_aug1[:, :, i], static_info)
            feature_aug1 = enriched_feature_augs1.to(feature_aug1.device)

        z_aug1 = feature_aug1  # (batch_size, channels, seq_len)
        z_aug1 = torch.permute(z_aug1, (0, 2, 1)).contiguous()

        z_aug2 = feature_aug2
        z_aug2 = torch.permute(z_aug2, (0, 2, 1)).contiguous()

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long()

        nce = 0
        encode_samples = torch.empty((self.timestep, batch, self.output_dim)).float()

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.output_dim)
        forward_seq = z_aug1[:, :t_samples + 1, :]  # transformer input value
        c_t = self.seq_transformer(forward_seq)
        pred = torch.empty((self.timestep, batch, self.output_dim)).float()
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)


class SimclrHARAR(nn.Module):
    def __init__(self, hidden_dim1=256, hidden_dim2=128, hidden_dim3=50):
        super(SimclrHARAR, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(96, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, hidden_dim3)
        )

    def forward(self, feature_aug1):
        return self.projection_head(feature_aug1)


class CSSHARAR(nn.Module):
    def __init__(self, model_params, projection_neuron=1024):
        super(CSSHARAR, self).__init__()

        params = dict(model_params)
        self.input_seq = int(params['input_seq'])
        self.projection_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.input_seq * 96, projection_neuron)
        )

    def forward(self, feature_aug1):
        return self.projection_head(feature_aug1)

class CPCHARAR(nn.Module):
    def __init__(self, model_params):
        super(CPCHARAR, self).__init__()

        params = dict(model_params)
        self.timestep = int(params['timestep'])
        self.num_classes = int(params['num_classes'])

        self.gru = nn.GRU(input_size=128,
                          hidden_size=256,
                          batch_first=True,
                          num_layers=2)
        self.Wk = nn.ModuleList([nn.Linear(256, 128) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax(dim=1)

        self.logit = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.Linear(128, self.num_classes)
        )

    def forward(self, feature_aug1):
        feature_aug1 = feature_aug1.permute(0, 2, 1).contiguous() # b, seq_len, c
        forward_seq = feature_aug1[:, :self.timestep + 1, :]
        forward_seq = forward_seq.permute(1,0,2).contiguous()
        batch = feature_aug1.shape[0]
        c_t, h_n = self.gru(feature_aug1[:, : self.timestep + 1, :])  # b, seq_len, 2 * c
        pred = torch.empty((self.timestep, batch, 128)).float()
        c_t = c_t.permute(0, 2, 1).contiguous()
#         print("shape : ", c_t.shape)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t[:, :, -1])
        nce = 0
        for i in np.arange(0, self.timestep):
            total = torch.mm(forward_seq[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, c_t
