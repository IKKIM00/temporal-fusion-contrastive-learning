import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from libs.augmentation import TSTCCDataTransform
from torchsampler import ImbalancedDatasetSampler


class MobiActDataset(Dataset):
    def __init__(self, X_data, y_data, aug_params, training_mode):
        super(MobiActDataset, self).__init__()
        self.training_mode = training_mode

        self.observed_real = torch.from_numpy(X_data['observed_real'])
        self.static_real = torch.from_numpy(X_data['static_real'])
        self.static_cate = torch.from_numpy(X_data['gender'])
        self.static = torch.cat([self.static_real, self.static_cate.unsqueeze(-1)], dim=1)
        self.y_data = torch.from_numpy(y_data)

        self.observed_real = torch.permute(self.observed_real, (1, 2, 0)).contiguous()

        self.len = self.observed_real.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = TSTCCDataTransform(self.observed_real, aug_params)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.observed_real[index], self.y_data[index], self.aug1[index], self.aug2[index], self.static[index]
        else:
            return self.observed_real[index], self.y_data[index], self.observed_real[index], self.y_data[index], self.static[index]

    def __len__(self):
        return self.len


class DLRDataset(Dataset):
    def __init__(self, X_data, y_data, aug_params, training_mode):
        super(DLRDataset, self).__init__()
        self.training_mode = training_mode

        self.observed_real = torch.from_numpy(X_data['observed_real'])
        self.static_real = torch.from_numpy(X_data['static_real'])
        self.static_cate = torch.from_numpy(X_data['gender'])
        self.static = torch.cat([self.static_real, self.static_cate.unsqueeze(-1)], dim=1)
        self.y_data = torch.from_numpy(y_data)

        self.observed_real = torch.permute(self. observed_real, (1, 2, 0)).contiguous()

        if training_mode == "self_supervised":
            self.aug1, self.aug2 = TSTCCDataTransform(self.observed_real, aug_params)

        self.len = self.observed_real.shape[0]

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.observed_real[index], self.y_data[index], self.aug1[index], self.aug2[index], self.static[index]
        else:
            return self.observed_real[index], self.y_data[index], self.observed_real[index], self.y_data[index], \
                   self.static[index]

    def __len__(self):
        return self.len


def data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test, aug_params, data_type,
                   training_mode, use_sampler=False):

    choose_dataset = {'mobiact': MobiActDataset,
                      'dlr': DLRDataset}

    train_dataset = choose_dataset[data_type](X_train, y_train, aug_params, training_mode)
    valid_dataset = choose_dataset[data_type](X_valid, y_valid, aug_params, training_mode)
    test_dataset = choose_dataset[data_type](X_test, y_test, aug_params, training_mode)

    if use_sampler:
        def get_y_label(dataset):
            return dataset.y_data.view(-1)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=aug_params['batch_size'],
                                                   sampler=ImbalancedDatasetSampler(train_dataset,
                                                                                    callback_get_label=get_y_label),
                                                   drop_last=aug_params['drop_last'],
                                                   num_workers=0)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=aug_params['batch_size'],
                                                   shuffle=True,
                                                   drop_last=aug_params['drop_last'],
                                                   num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=aug_params['batch_size'] // 4,
                                               shuffle=False,
                                               drop_last=aug_params['drop_last'],
                                               num_workers=0)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=aug_params['batch_size'] // 2,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader

