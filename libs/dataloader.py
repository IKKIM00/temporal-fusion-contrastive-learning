import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

import os
import numpy as np

from libs.augmentation import DataTransform


class MobiActDataset(Dataset):
    def __init__(self, X_data, y_data, aug_method1, aug_method2, aug_params, training_mode):
        super(MobiActDataset, self).__init__()
        self.training_mode = training_mode

        self.observed_real = torch.from_numpy(X_data['observed_real'])
        self.static_real = torch.from_numpy(X_data['static_real'])
        self.static_cate = torch.from_numpy(X_data['gender'])
        self.static = torch.cat([self.static_real, self.static_cate.unsqueeze(-1)], dim=1)
        self.y_data = torch.from_numpy(y_data)
        
        self.len = self.observed_real.shape[1]

        self.observed_real = torch.permute(self.observed_real, (1, 0, 2)).contiguous()

        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.observed_real, aug_method1, aug_method2, aug_params)
            self.aug1, self.aug2 = self.aug1.permute(0, 2, 1).contiguous(), self.aug2.permute(0, 2, 1).contiguous()

        self.observed_real = self.observed_real.permute(0, 2, 1).contiguous()

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.observed_real[index], self.y_data[index], self.aug1[index], self.aug2[index], self.static[index]
        else:
            return self.observed_real[index], self.y_data[index], self.observed_real[index], self.y_data[index], self.static[index]

    def __len__(self):
        return self.len


class DLRDataset(Dataset):
    def __init__(self, X_data, y_data, aug_method1, aug_method2, aug_params, training_mode):
        super(DLRDataset, self).__init__()
        self.training_mode = training_mode

        self.observed_real = torch.from_numpy(X_data['observed_real'])
        self.static_real = torch.from_numpy(X_data['static_real'])
        self.static_cate = torch.from_numpy(X_data['gender'])
        self.static = torch.cat([self.static_real, self.static_cate.unsqueeze(-1)], dim=1)
        self.y_data = torch.from_numpy(y_data)
        
        self.len = self.observed_real.shape[1]
        self.observed_real = torch.permute(self.observed_real, (1, 0, 2)).contiguous()

        if training_mode == "self_supervised":
            self.aug1, self.aug2 = DataTransform(self.observed_real, aug_method1, aug_method2, aug_params)
            self.aug1, self.aug2 = self.aug1.permute(0, 2, 1).contiguous(), self.aug2.permute(0, 2, 1).contiguous()

        self.observed_real = self.observed_real.permute(0, 2, 1).contiguous()

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.observed_real[index], self.y_data[index], self.aug1[index], self.aug2[index], self.static[index]
        else:
            return self.observed_real[index], self.y_data[index], self.observed_real[index], self.y_data[index], \
                   self.static[index]

    def __len__(self):
        return self.len


def data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test, aug_params, data_type,
                   aug_method1, aug_method2, batch_size, training_mode, sampler_use=False, ratio=1):

    choose_dataset = {'mobiact': MobiActDataset,
                      'dlr': DLRDataset}

    train_dataset = choose_dataset[data_type](X_train, y_train, aug_method1, aug_method2, aug_params, training_mode)
    valid_dataset = choose_dataset[data_type](X_valid, y_valid, aug_method1, aug_method2, aug_params, training_mode)
    test_dataset = choose_dataset[data_type](X_test, y_test, aug_method1, aug_method2, aug_params, training_mode)

    if sampler_use:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   sampler=RandomSampler(data_source=train_dataset,
                                                                         replacement=True,
                                                                         num_samples=int(ratio * len(train_dataset))),
                                                   drop_last=True,
                                                   num_workers=10,
                                                   pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=10,
                                                   pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=batch_size // 4,
                                               drop_last=True,
                                               shuffle=False,
                                               num_workers=10,
                                               pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size // 2,
                                              shuffle=False,
                                              num_workers=10,
                                              pin_memory=True)

    return train_loader, valid_loader, test_loader

