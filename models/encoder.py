import torch
import torch.nn as nn
from models.grn import gated_residual_network

class cnn_encoder(nn.Module):
    def __init__(self, model_params, static_use=False):
        super(cnn_encoder, self).__init__()

        params = dict(model_params)
        self.kernel_size = int(params['kernel_size'])
        self.input_channels = int(params['input_size'])
        self.stride = int(params['stride'])
        self.dropout = float(params['dropout'])
        self.feature_len = int(params['feature_len'])
        self.output_dim = int(params['encoder_output_dim'])
        self.num_classes = int(params['num_classes'])
        self.static_use = static_use

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
        if static_use:
            self.logits = nn.Linear((model_output_dim + 1) * self.output_dim, self.num_classes)
        else:
            self.logits = nn.Linear(model_output_dim * self.output_dim, self.num_classes)

    def forward(self, obs_input, static_input=None):
        x = self.conv_block1(obs_input)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        if self.static_use:
            x = torch.cat([x, static_input.unsqueeze(-1)], dim=2)

        x_flat = self.flatten(x)
        logits = self.logits(x_flat)
        return logits, x


class lstm_encoder(nn.Module):
    def __init__(self, model_params, static_info=False):
        super(lstm_encoder, self).__init__()

        params = dict(model_params)

        self.input_dim = int(params['input_size'])
        self.output_dim = int(params['encoder_output_dim'])
        self.feature_len = int(params['feature_len'])
        self.dropout = float(params['dropout'])
        self.num_classes = int(params['num_classes'])
        self.static_info = static_info

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.output_dim,
                            batch_first=True)
        if static_info:
            self.static_context_state_h = gated_residual_network(input_dim=self.output_dim,
                                                                 hidden_dim=self.output_dim,
                                                                 droupout_rate=self.dropout)
            self.static_context_state_c = gated_residual_network(input_dim=self.output_dim,
                                                                 hidden_dim=self.output_dim,
                                                                 droupout_rate=self.dropout)
            self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.feature_len + 1)
            self.logits = nn.Linear(self.output_dim * 2, self.num_classes)
        else:
            self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.feature_len)
            self.logits = nn.Linear(self.output_dim, self.num_classes)

    def forward(self, x, static_vec=None, static_enrichment_vec=None):
        if self.static_info:
            static_h = self.static_context_state_h(static_vec.unsqueeze(0))
            static_c = self.static_context_state_c(static_vec.unsqueeze(0))
            output, (h_t, c_t) = self.lstm(x, (static_h, static_c))
            h_t = torch.cat([h_t.squeeze(), static_enrichment_vec], dim=1)

        else:
            output, (h_t, c_t) = self.lstm(x)
        output = self.adaptive_pooling(torch.permute(output, (0, 2, 1)).contiguous())
        logits = self.logits(h_t)
        return logits.squeeze(), output

if __name__ == '__main__':
    from data_formatters.mobiact import MobiactFormatter
    from libs.dataloader import data_generator
    from models.static import *

    dataformatter = MobiactFormatter()
    dataset_dir = '../datasets/mobiact_preprocessed/'
    X_train, y_train, X_valid, y_valid, X_test, y_test = dataformatter.split_data(dataset_dir=dataset_dir)
    model_params, aug_params, loss_params = dataformatter.get_experiment_params()

    train_loader, _, _ = data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                        model_params, aug_params, 'mobiact', 'LSTM', 'train_linear')
    dataiter = iter(train_loader)
    observed_real, y, aug1, aug2, static = dataiter.next()

    # model = cnn_encoder(model_params, static_use=True)
    model = lstm_encoder(model_params, static_info=False)
    static_embedding_model = StaticEmbedding(model_params, 'cpu')
    static_variable_selection_model = StaticVariableSelection(model_params, 'cpu')
    static_embedding = static_embedding_model(static)
    static_context_enrichment, static_vec, sparse_weights = static_variable_selection_model(static_embedding)
    # logits, output = model(observed_real.float(), static_vec)
    # logits, output = model(observed_real.float(), static_vec, static_context_enrichment)
    logits, output = model(observed_real.float())
    print(logits.shape, output.shape, y.shape)
    # LSTM - (1, 512, 20), (512, 32, 30)
    # CNN - (512, 20), (512, 32, 30)
