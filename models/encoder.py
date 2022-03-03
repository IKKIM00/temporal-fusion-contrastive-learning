import torch
import torch.nn as nn
from models.grn import gated_residual_network
from models.static_utils import static_combine_and_mask

class cnn_encoder(nn.Module):
    def __init__(self, configs):
        super(cnn_encoder, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x

class lstm_encoder(nn.Module):
    def __init__(self, configs, static_info=False):
        super(lstm_encoder, self).__init__()

        self.input_dim = configs.input_channels
        self.hidden_dim = configs.hidden_size
        self.num_statics = configs.num_statics
        self.static_info = static_info

        self.lstm = nn.LSTM(input_size=configs.input_channels,
                            hidden_size=configs.hidden_size,
                            batch_first=True)
        self.static_combine_and_mask = static_combine_and_mask(configs=configs)
        self.static_context_state_h = gated_residual_network(input_dim=configs.hidden_size,
                                                             output_dim=configs.hidden_size,
                                                             droupout_rate=configs.dropout)
        self.static_context_state_c = gated_residual_network(input_dim=configs.hidden_size,
                                                             output_dim=configs.hidden_size,
                                                             droupout_rate=configs.dropout)
        self.logits = nn.Linear(configs.hidden_size, configs.num_classes)

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
