import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss

def get_static_model(static_vec_model, combine_model, feature, static_input):
    static_vec, sparse_weights = static_vec_model(static_input)
    static_combined = combine_model(feature, static_vec)
    return static_combined


def model_train(encoder, tfcc_model, static_vec_model, combine_model, encoder_optimizer, tfcc_optimizer, static_optimizer, criterion, train_loader, static_input, model_params, device, training_mode, static_use):
    total_loss = []
    total_acc = []
    encoder.train()
    tfcc_model.train()
    static_vec_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        data, labels = data.float().to(device), labels.float().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        encoder_optimizer.zero_grad()
        tfcc_optimizer.zero_grad()

        if training_mode == "self_supervised":
            predictions1, features1 = encoder(aug1)
            predictions2, features2 = encoder(aug2)


            if static_use == True:
                static_optimizer.zero_grad()
                static_vec, sparse_weights = static_vec_model(static_input)
                static_combined1 = combine_model(features1, static_vec)
                static_combined2 = combine_model(features2, static_vec)
