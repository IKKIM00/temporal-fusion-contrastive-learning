import torch
import torch.nn as nn
from models.grn import gated_residual_network
from models.static import StaticVariableSelection

class cnn_encoder(nn.Module):
    def __init__(self, model_params):
        super(cnn_encoder, self).__init__()

        params = dict(model_params)
        self.kernel_size = int(params['kernel_size'])
        self.input_channels = int(params['input_size'])
        self.stride = int(params['stride'])
        self.dropout = float(params['dropout'])
        self.feature_len = int(params['feature_len'])
        self.output_dim = int(params['output_dim'])
        self.num_classes = int(params['num_classes'])

        self.flatten = nn.Flatten()
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(self.input_channels, 32, kernel_size=self.kernel_size,
                      stride=self.stride, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16, stride=4),
            nn.Dropout(self.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=self.kernel_size, stride=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16, stride=4)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, self.output_dim, kernel_size=self.kernel_size, stride=1, bias=False),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16, stride=4),
        )

        model_output_dim = self.feature_len
        self.logits = nn.Linear(model_output_dim * self.output_dim, self.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = self.flatten(x)
        logits = self.logits(x_flat)
        return logits, x

class lstm_encoder(nn.Module):
    def __init__(self, model_params, static_info=False):
        super(lstm_encoder, self).__init__()

        params = dict(model_params)

        self.input_dim = int(params['input_dim'])
        self.hidden_dim = int(params['hidden_dim'])
        self.input_size = int(params['input_size'])
        self.dropout = float(params['dropout'])
        self.num_classes = int(params['num_classes'])
        self.static_info = static_info

        self.lstm = nn.LSTM(input_size=self.input_dim ,
                            hidden_size=self.hidden_dim,
                            batch_first=True)
        self.static_combine_and_mask = StaticVariableSelection(model_params=model_params)
        self.static_context_state_h = gated_residual_network(input_dim=self.hidden_dim,
                                                             output_dim=self.hidden_dim,
                                                             droupout_rate=self.dropout)
        self.static_context_state_c = gated_residual_network(input_dim=self.hidden_dim,
                                                             output_dim=self.hidden_dim,
                                                             droupout_rate=self.dropout)
        self.logits = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, x, embedding):
        static_encoder, static_weights = self.static_combine_and_mask(embedding)
        if self.static_info == True:
            static_h = self.static_context_state_h(static_encoder)
            static_c = self.static_context_state_c(static_encoder)
            output, (h_t, c_t) = self.lstm(x, (static_h, static_c))
        else:
            output, (h_t, c_t) = self.lstm(x)
        logits = self.logits(h_t)
        return logits, output
