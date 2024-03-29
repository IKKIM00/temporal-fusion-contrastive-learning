# -*- coding: utf-8 -*-
import json

import torch
import torch.nn as nn
from models.grn import gated_residual_network


class StaticEncoder(nn.Module):
    def __init__(self, model_params):
        super(StaticEncoder, self).__init__()

        self.static_embedding = StaticEmbedding(model_params)
        self.static_variable_selection = StaticVariableSelection(model_params)

    def forward(self, static_input):
        static_embedding = self.static_embedding(static_input)
        context_variable, context_enrichment = self.static_variable_selection(static_embedding)
        return context_variable, context_enrichment

class StaticEmbedding(nn.Module):
    """
    Transform static data into static input.
    Applies linear transformation to real value variables and embedding to categorical variables

    :arg
        static inputs: static input to transform
    :returns
        static_inputs: transformed inputs
    """
    def __init__(self, model_params):
        super(StaticEmbedding, self).__init__()

        params = dict(model_params)

        self.column_definition = params['column_definition']
        self.output_dim = int(params['encoder_output_dim'])
        self.category_counts = json.loads(str(params['category_counts']))   # 각 categorical value들의 category 종류 개수
        self._static_regular_inputs = json.loads(str(params['static_regular_inputs']))
        self._static_categorical_inputs = json.loads(str(params['static_categorical_inputs']))
        self.num_categorical_variables = len(self._static_categorical_inputs)
        self.num_regular_variables = len(self._static_regular_inputs)

        ### Embedding Variables ###
        embedding_sizes = [
            self.output_dim for i, size in enumerate(self.category_counts)
        ]
        self.embeddings = nn.ModuleList()
        for i in range(self.num_categorical_variables):
            embedding = nn.Embedding(
                num_embeddings=self.category_counts[i],
                embedding_dim=embedding_sizes[i])
            self.embeddings.append(embedding)

        self.emb_regulars = nn.ModuleList()
        for i in range(self.num_regular_variables):
            emb_reg = nn.Linear(1, self.output_dim)
            self.emb_regulars.append(emb_reg)

    def forward(self, all_inputs):
        """
        :param all_inputs: size of (num of people, num of feature), static inputs
        :return: embedded static input, size=(num of people, num of feature, embedding size)
        """
        regular_inputs, categorical_inputs = all_inputs[:, :self.num_regular_variables], all_inputs[:, self.num_regular_variables:]

        cate_embedded_inputs = [
            self.embeddings[i](categorical_inputs[Ellipsis, i].int()) for i in range(self.num_categorical_variables)
        ]
        reg_embedded_inputs = [
            self.emb_regulars[i](regular_inputs[:, i].float().reshape(-1, 1)) for i in range(self.num_regular_variables)
        ]
        static_inputs = cate_embedded_inputs + reg_embedded_inputs

        static_inputs = torch.stack(static_inputs, dim=1)
        return static_inputs


class StaticVariableSelection(nn.Module):
    def __init__(self, model_params):
        super(StaticVariableSelection, self).__init__()

        self.input_size = len(json.loads(str(model_params['static_regular_inputs']))) + len(json.loads(str(model_params['static_categorical_inputs'])))
        self.output_dim = int(model_params['encoder_output_dim'])
        self.dropout = float(model_params['dropout'])
        self.feature_len = int(model_params['feature_len'])

        self.flatten = nn.Flatten()
        self.trans_emb_grn = nn.ModuleList()
        for i in range(self.input_size):
            e = gated_residual_network(input_dim=self.output_dim,
                                       hidden_dim=self.output_dim)
            self.trans_emb_grn.append(e)

        self.grn0 = gated_residual_network(input_dim=self.input_size * self.output_dim,
                                           hidden_dim=self.output_dim,
                                           output_dim=self.input_size,
                                           droupout_rate=self.dropout)
        self.static_context_enrichment = gated_residual_network(input_dim=self.output_dim,
                                                                hidden_dim=self.output_dim)
        self.static_context_variable = gated_residual_network(input_dim=self.output_dim,
                                                              hidden_dim=self.output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, embedding):
        """
        :param embedding: (batch_size, seq, c)
        :return:
        """
        b, num_statics, _ = embedding.shape
        flatten = self.flatten(embedding)
        mlp_output = self.grn0(flatten)
        sparse_weights = self.softmax(mlp_output)
        sparse_weights = torch.unsqueeze(sparse_weights, dim=-1)

        trans_emb_list = []
        for i in range(self.input_size):
            e = self.trans_emb_grn[i](embedding[:, i:i + 1, :])
            trans_emb_list.append(e)
        transformed_embedding = torch.cat(trans_emb_list, dim=1)
        combined = torch.mul(sparse_weights, transformed_embedding)
        static_vec = torch.sum(combined, dim=1)

        return self.static_context_variable(static_vec), self.static_context_enrichment(static_vec)


if __name__ == '__main__':
    from data_formatters.mobiact import MobiactFormatter
    from libs.dataloader import *

    dataformatter = MobiactFormatter()
    dataset_dir = '../datasets/mobiact_preprocessed/'
    X_train, y_train, X_valid, y_valid, X_test, y_test = dataformatter.split_data(dataset_dir=dataset_dir)
    model_params, aug_params, loss_params = dataformatter.get_experiment_params()
    train_loader, valid_loader, test_loader = data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                             aug_params, data_type='mobiact',
                                                             model_type='CNN', training_mode='self_supervised')
    dataiter = iter(train_loader)
    observed_real, y, aug1, aug2, static = dataiter.next()
    device = 'cpu'
    static_embedding_model = StaticEmbedding(model_params, device)
    embedding = static_embedding_model(static)
    print(embedding.shape)
    static_variable_selection_model = StaticVariableSelection(model_params, device)
    static_context_variable, static_context_enrichment = static_variable_selection_model(embedding)
    print(static_context_variable.shape, static_context_enrichment.shape)
