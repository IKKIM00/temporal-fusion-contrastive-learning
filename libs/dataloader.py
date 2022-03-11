import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from libs.augmentation import TSTCCDataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, aug_params, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset.tensors[0]
        y_train = dataset.tensors[1]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = TSTCCDataTransform(self.x_data, aug_params)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, model_params, aug_params, training_mode):

    train_dataset = torch.load(os.path.join(data_path, "train.pt"))
    valid_dataset = torch.load(os.path.join(data_path, "valid.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, aug_params, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, aug_params, training_mode)
    test_dataset = Load_Dataset(test_dataset, aug_params, training_mode)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=model_params['batch_size'],
                                               shuffle=True, drop_last=model_params['drop_last'],
                                               num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=model_params['batch_size'],
                                               shuffle=False, drop_last=model_params['drop_last'],
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=model_params['batch_size'],
                                              shuffle=False, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader