import torch
import torch.nn as nn
from models.autoregressive import PositionalEncoding


class BaseEncoder(nn.Module):
    def __init__(self, model_params, static_use=False):
        super(BaseEncoder, self).__init__()

        params = dict(model_params)
        self.kernel_size = int(params['kernel_size'])
        self.input_channels = int(params['input_size'])
        self.stride = int(params['stride'])
        self.dropout = float(params['dropout'])
        if static_use:
            self.feature_len = int(params['static_feature_len'])
        else:
            self.feature_len = int(params['feature_len'])
        self.output_dim = int(params['encoder_output_dim'])
        self.num_classes = int(params['num_classes'])
        self.static_use = static_use

        self.flatten = nn.Flatten()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32,
                      kernel_size=self.kernel_size,
                      stride=self.stride),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16, stride=4),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64,
                      kernel_size=self.kernel_size,
                      stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.output_dim,
                      kernel_size=self.kernel_size,
                      stride=1),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU()
        )
        self.static_linear = nn.Linear(self.output_dim, 32)
        self.globalMaxPool1D = nn.AdaptiveMaxPool1d(1)
        self.logits = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(self.output_dim // 2, self.num_classes),
#             nn.ReLU(),
#             nn.Dropout(p=0.2),
#             nn.Linear(self.output_dim // 4, self.num_classes),
            nn.ReLU(),
            nn.Dropout(p=0.2)
        )
        # model_output_dim = self.feature_len
        #
        # self.logits = nn.Linear(model_output_dim * self.output_dim, self.num_classes)

    def forward(self, obs_input, static_input=None):
        x = self.conv_block1(obs_input)
        if self.static_use:
            static_input = self.static_linear(static_input).unsqueeze(-1)
            x = torch.cat([x, static_input], dim=2)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = self.globalMaxPool1D(x).squeeze()
        logits = self.logits(x_flat)
        return logits, x


class SimclrHAREncoder(nn.Module):
    def __init__(self, model_params):
        super(SimclrHAREncoder, self).__init__()

        params = dict(model_params)
        self.kernel_size = int(params['kernel_size'])
        self.input_channels = int(params['input_size'])
        self.stride = int(params['stride'])
        self.num_classes = int(params['num_classes'])

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32,
                      kernel_size=self.kernel_size,
                      stride=self.stride),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64,
                      kernel_size=self.kernel_size,
                      stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 96,
                      kernel_size=self.kernel_size,
                      stride=1),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )

        self.globalMaxPool1D = nn.AdaptiveMaxPool1d(1)
        self.logits = nn.Linear(96, self.num_classes)

    def forward(self, obs_input):
        x = self.conv_block1(obs_input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        # global Max Pooling 1D
        x = self.globalMaxPool1D(x).squeeze()
        return self.logits(x), x


class CSSHAREncoder(nn.Module):
    def __init__(self, model_params):
        super(CSSHAREncoder, self).__init__()

        params = dict(model_params)
        self.kernel_size = int(params['kernel_size'])
        self.input_channels = int(params['input_size'])
        self.input_seq = int(params['input_seq'])
        self.stride = int(params['stride'])
        self.dropout = float(params['dropout'])
        self.output_dim = int(params['encoder_output_dim'])
        self.num_classes = int(params['num_classes'])

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32,
                      kernel_size=self.kernel_size,
                      stride=self.stride),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64,
                      kernel_size=self.kernel_size,
                      stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 96,
                      kernel_size=self.kernel_size,
                      stride=1),
            nn.BatchNorm1d(96),
            nn.ReLU()
        )

        self.positional_encodig = PositionalEncoding(96)
        encoder_layer = nn.TransformerEncoderLayer(d_model=96, nhead=1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.flatten = nn.Flatten()
        self.logits = nn.Sequential(
            nn.Linear(self.input_seq * 96, 96),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, obs_input):
        x = self.conv_block1(obs_input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = x.permute(2, 0, 1).contiguous() # seq, b, c
        x = self.positional_encodig(x)
        x = x.permute(1, 0, 2).contiguous()
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        logit = self.logits(x)
        return logit, x
