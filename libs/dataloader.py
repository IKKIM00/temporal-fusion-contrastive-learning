import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from libs.augmentation import TSTCCDataTransform


class MobiActDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, X_data, y_data, aug_params, model_type, training_mode):
        super(MobiActDataset, self).__init__()
        self.training_mode = training_mode

        self.observed_real = torch.from_numpy(X_data['observed_real'])
        self.static_real = torch.from_numpy(X_data['static_real'])
        self.static_cate = torch.from_numpy(X_data['gender'])
        self.static = torch.cat([self.static_real, self.static_cate.unsqueeze(-1)], dim=1)
        self.y_data = torch.from_numpy(y_data)

        if model_type == 'CNN':
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


def data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test, model_params, aug_params, data_type, model_type, training_mode):

    choose_dataset = {'mobiact': MobiActDataset}

    train_dataset = choose_dataset[data_type](X_train, y_train, aug_params, model_type, training_mode)
    valid_dataset = choose_dataset[data_type](X_valid, y_valid, aug_params, model_type, training_mode)
    test_dataset = choose_dataset[data_type](X_test, y_test, aug_params, model_type, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=model_params['batch_size'],
                                               shuffle=True, drop_last=model_params['drop_last'],
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=model_params['batch_size'] // 2,
                                               shuffle=False, drop_last=model_params['drop_last'],
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=model_params['batch_size'],
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader