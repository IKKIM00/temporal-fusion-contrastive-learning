import torch
import torch.nn as nn
import numpy as np
from models.attention import Seq_Transformer
from models.grn import gated_residual_network


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
