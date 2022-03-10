import json

import torch
import torch.nn as nn
from models.grn import gated_residual_network

class StaticEmbedding(nn.Module):
    def __init__(self, model_params):
        super(StaticEmbedding, self).__init__()

        params = dict(model_params)

        self.input_size = int(params['input_size'])
        self.column_definition = params['column_definition']
        self.output_dim = int(params['output_dim'])
        self.category_counts = json.loads(str(params['category_counts']))
        self._static_regular_inputs = json.loads(str(params['static_regular_inputs']))
        self._static_categorical_inputs = json.loads(str(params['static_categorical_inputs']))

        self.num_categorical_variables = len(self.category_counts)
        self.num_regular_variables = self.input_size - self.num_categorical_variables

        ### Embedding Variables ###
        embedding_sizes = [
            self.output_dim for i , size in enumerate(self.category_counts)
        ]
        self.embeddings = []
        for i in range(self.num_categorical_variables):
            embedding = nn.Sequential(
                nn.Embedding(
                    num_embeddings=self.category_counts[i],
                    embedding_dim=embedding_sizes[i])
            )
            self.embeddings.append(embedding)

    def forward(self, all_inputs):
        regular_inputs, categorical_inputs = all_inputs[:, :, :self.num_regular_variables], all_inputs[:, :, self.num_regular_variables:]
        b, seq_len, c = regular_inputs.size()

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[Ellipsis, i]) for i in range(self.num_categorical_variables)
        ]
        static_inputs = [nn.Linear(1, self.output_dim)(regular_inputs[:, 0, i: i + 1]) for i in range(self.num_regular_variables)]\
                        + [embedded_inputs[i][:, 0, :] for i in range(self.num_categorical_variables)]

        static_inputs = torch.stack(static_inputs, dim=1)
        return static_inputs


class StaticVector(nn.Module):
    def __init__(self, model_params):
        super(StaticVector, self).__init__()

        params = dict(model_params)

        self.input_size = int(params['input_size'])
        self.output_dim = int(params['output_dim'])
        self.dropout = float(params['dropout'])

        self.static_embedding = StaticEmbedding(model_params)
        self.flatten = nn.Flatten()
        self.grn = gated_residual_network(input_dim=self.input_size * self.output_dim,
                                          hidden_dim=self.output_dim,
                                          output_dim=self.input_size,
                                          droupout_rate=self.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, all_inputs):
        """
        :param embedding: (batch_size, seq, c)
        :return:
        """
        embedding = self.static_embedding(all_inputs)
        flatten = self.flatten(embedding)
        mlp_output = self.grn(flatten)
        sparse_weights = self.softmax(mlp_output)
        sparse_weights = torch.unsqueeze(sparse_weights, dim=-1)

        trans_emb_list = []
        for i in range(self.num_statics):
            e = gated_residual_network(input_dim=self.output_dim,
                                       hidden_dim=self.output_dim)(embedding[:, i:i + 1, :])
            trans_emb_list.append(e)
        transformed_embedding = torch.cat(trans_emb_list, dim=1)
        combined = torch.mul(sparse_weights, transformed_embedding)

        static_vec = torch.sum(combined, dim=1)
        return static_vec, sparse_weights

class CombineFeatureAndStatic(nn.Module):
    def __init__(self, model_params):
        super(CombineFeatureAndStatic, self).__init__()

        params = dict(model_params)

        self.input_size = int(params['input_size'])
        self.output_dim = int(params['output_dim'])
        self.grn = gated_residual_network(input_dim=self.input_size,
                                          hidden_dim=self.output_dim,
                                          additional_context=True)
    def forward(self, feature, static_vec):
        return self.grn(feature, static_vec)
