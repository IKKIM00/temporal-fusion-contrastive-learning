import torch
import torch.nn as nn
from models.grn import gated_residual_network

class static_combine_and_mask(nn.Module):
    def __init__(self, configs):
        super(static_combine_and_mask, self).__init__()
        self.num_statics = configs.num_statics
        self.hidden_dim = configs.hidden_dim

        self.flatten = nn.Flatten()
        self.grn = gated_residual_network(input_dim=configs.input_channels,
                                          hidden_dim=configs.hidden_size,
                                          output_dim=configs.num_statics,
                                          droupout_rate=configs.dropout)
        self.softmax = nn.Softmax()

    def forward(self, embedding):
        """
        :param embedding: (batch_size, seq, c)
        :return:
        """
        flatten = self.flatten(embedding)
        mlp_output = self.grn(flatten)
        sparse_weights = self.softmax(mlp_output)
        sparse_weights = torch.unsqueeze(sparse_weights, dim=-1)

        trans_emb_list = []
        for i in range(self.num_statics):
            e = gated_residual_network(input_dim=embedding[:, i:i + 1, :],
                                       hidden_dim=self.hidden_dim)(embedding[:, i:i + 1, :])
            trans_emb_list.append(e)
        transformed_embedding = torch.cat(trans_emb_list, dim=1)
        combined = torch.mul(sparse_weights, transformed_embedding)

        static_vec = torch.sum(combined, dim=1)
        return static_vec, sparse_weights