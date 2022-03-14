# -*- coding: utf-8 -*-
import json

import torch
import torch.nn as nn
from models.grn import gated_residual_network

class StaticEmbedding(nn.Module):
    """
    Transform static data into static input.
    Applies linear transformation to real value variables and embedding to categorical variables

    :arg
        static inputs: static input to transform
    :returns
        static_inputs: transformed inputs
    """
    def __init__(self, model_params, device):
        super(StaticEmbedding, self).__init__()

        params = dict(model_params)

        self.input_size = len(params['column_definition'])
        self.column_definition = params['column_definition']
        self.output_dim = int(params['output_dim'])
        self.category_counts = json.loads(str(params['category_counts']))   # 각 categorical value들의 category 종류 개수
        self._static_regular_inputs = json.loads(str(params['static_regular_inputs']))
        self._static_categorical_inputs = json.loads(str(params['static_categorical_inputs']))

        self.num_categorical_variables = len(self.category_counts)
        self.num_regular_variables = self.input_size - self.num_categorical_variables

        ### Embedding Variables ###
        embedding_sizes = [
            self.output_dim for i, size in enumerate(self.category_counts)
        ]
        self.embeddings = []
        for i in range(self.num_categorical_variables):
            embedding = nn.Sequential(
                nn.Embedding(
                    num_embeddings=self.category_counts[i],
                    embedding_dim=embedding_sizes[i]),
            ).to(device)
            self.embeddings.append(embedding)

    def forward(self, all_inputs):
        """
        :param all_inputs: size of (num of people, num of feature), static inputs
        :return: embedded static input, size=(num of people, num of feature, embedding size)
        """
        regular_inputs, categorical_inputs = all_inputs[:, :self.num_regular_variables], all_inputs[:, self.num_regular_variables:]

        embedded_inputs = [
            self.embeddings[i](categorical_inputs[Ellipsis, i].int()) for i in range(self.num_categorical_variables)
        ]
        static_inputs = [nn.Linear(1, self.output_dim)(regular_inputs[:, i: i + 1].float()) for i in range(self.num_regular_variables)]\
                        + [embedded_inputs[i][:, :] for i in range(self.num_categorical_variables)]

        static_inputs = torch.stack(static_inputs, dim=1)
        return static_inputs


class StaticVariableSelection(nn.Module):
    def __init__(self, model_params):
        super(StaticVariableSelection, self).__init__()

        self.input_size = len(model_params['column_definition'])
        self.output_dim = int(model_params['output_dim'])
        self.dropout = float(model_params['dropout'])
        self.batch_size = int(model_params['batch_size'])
        self.feature_len = int(model_params['feature_len'])

        self.flatten = nn.Flatten()
        self.grn = gated_residual_network(input_dim=self.input_size * self.output_dim,
                                          hidden_dim=self.output_dim,
                                          output_dim=self.input_size,
                                          droupout_rate=self.dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding):
        """
        :param embedding: (batch_size, seq, c)
        :return:
        """
        num_person, num_statics, _ = embedding.shape
        flatten = self.flatten(embedding)
        mlp_output = self.grn(flatten)
        sparse_weights = self.softmax(mlp_output)
        sparse_weights = torch.unsqueeze(sparse_weights, dim=-1)

        trans_emb_list = []
        for i in range(num_statics):
            e = gated_residual_network(input_dim=self.output_dim,
                                       hidden_dim=self.output_dim)(embedding[:, i:i + 1, :])
            trans_emb_list.append(e)
        transformed_embedding = torch.cat(trans_emb_list, dim=1)
        combined = torch.mul(sparse_weights, transformed_embedding)
        static_vec = torch.sum(combined, dim=1)
        static_vec = nn.Linear(num_person, self.feature_len)(torch.permute(static_vec, (1, 0)).contiguous())
        return static_vec.repeat(self.batch_size, 1, 1), sparse_weights

class CombineFeatureAndStatic(nn.Module):
    def __init__(self, model_params):
        super(CombineFeatureAndStatic, self).__init__()

        params = dict(model_params)

        self.output_dim = int(params['output_dim'])
        self.feature_len = int(params['feature_len'])

        self.flatten = nn.Flatten()
        self.grn = gated_residual_network(input_dim=self.output_dim * self.feature_len,
                                          hidden_dim=self.output_dim,
                                          additional_context=True)
    def forward(self, feature, static_vec):
        feature = self.flatten(feature)
        return self.grn(feature, static_vec)
