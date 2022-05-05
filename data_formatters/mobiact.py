import os
import data_formatters.base
import libs.utils as utils
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

BaseForamtter = data_formatters.base.BaseFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class MobiactFormatter(BaseForamtter):
    _column_definition = [
        ('person_id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('height', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('weight', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('gender', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('age', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('acc_x', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('acc_y', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('acc_z', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('gyro_x', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('gyro_y', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('gyro_z', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT)
    ]

    def __init__(self):
        self.identifier = None
        self._cat_scalers = None
        self._static_real_scalers = None
        self._observe_real_scalers = None
        self.column_definition = self.get_column_definition()
        self.static_real_columns = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, self.column_definition,
                                                                    {InputTypes.ID, InputTypes.OBSERVED_INPUT})
        self.static_cate_columns = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, self.column_definition,
                                                                     {InputTypes.ID, InputTypes.OBSERVED_INPUT})
        self.observed_real_columns = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, self.column_definition,
                                                             {InputTypes.ID, InputTypes.STATIC_INPUT})
        self.id_column = utils.get_single_col_by_input_type(InputTypes.ID, self.column_definition)
        self._target_scalers = LabelEncoder()
        self._num_classes_per_cat_input = None

    def split_data(self, dataset_dir):
        print('Formatting train-valid-test static data splits')

        max_length = 2995

        train_dir, valid_dir, test_dir = 'train/', 'valid/', 'test/'
        for idx, dir_name in enumerate([train_dir, valid_dir, test_dir]):
            id_data, static_real_data, static_cate_data, observed_real_data, y = [], [], [], [], []
            file_dir = dataset_dir + dir_name
            file_list = os.listdir(file_dir)
            for file in file_list:
                label = file.split('_')[0]
                temp = pd.read_csv(file_dir + file)
                if len(temp) > max_length:
                    temp = temp.iloc[:max_length]

                id_col = temp[self.id_column]
                static_real_col = temp[self.static_real_columns]
                static_cate_col = temp[self.static_cate_columns]
                observed_real_col = temp[self.observed_real_columns]

                id_data.append(id_col.values[0])
                static_real_data.append(static_real_col.values[0].tolist())
                static_cate_data.append(static_cate_col.values[0].tolist())
                observed_real_data.append(torch.tensor(observed_real_col.values))
                y.append(label)
            observed_real_data = pad_sequence(observed_real_data)
            if idx == 0:
                self.set_scalers(static_real_data, static_cate_data, observed_real_data)
                X_train = self.transform_inputs(static_real_data, static_cate_data, observed_real_data)
                y_train = self._target_scalers.fit_transform(y)
            elif idx == 1:
                X_valid = self.transform_inputs(static_real_data, static_cate_data, observed_real_data)
                y_valid = self._target_scalers.transform(y)
            else:
                X_test = self.transform_inputs(static_real_data, static_cate_data, observed_real_data)
                y_test = self._target_scalers.transform(y)

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def set_scalers(self, static_real_data, static_cate_data, observed_real_data):
        print('Setting scalers with static training data')

        self._static_real_scalers = StandardScaler().fit(static_real_data)
        self._observe_real_scalers = StandardScaler().fit(observed_real_data.reshape(-1, observed_real_data.shape[-1]))

        categorical_scalers = {}
        num_classes = []
        for i in range(len(self.static_cate_columns)):
            srs = np.asarray(static_cate_data)[Ellipsis, i]
            categorical_scalers[self.static_cate_columns[i]] = LabelEncoder().fit(srs.reshape(-1, ))
            num_classes.append(len(np.unique(srs)))

        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, static_real_data, static_cate_data, observed_real_data):
        output = {}
        output['observed_real'] = self._observe_real_scalers.transform(observed_real_data.reshape(-1, observed_real_data.shape[-1])).reshape(observed_real_data.shape)
        output['static_real'] = self._static_real_scalers.transform(static_real_data)

        for i in range(len(self.static_cate_columns)):
            string_df = np.asarray(static_cate_data)[Ellipsis, i]
            output[self.static_cate_columns[i]] = self._cat_scalers[self.static_cate_columns[i]].transform(string_df)
        return output

    def get_model_params(self):
        model_params = {
            'input_size': 6,
            'kernel_size': 22,
            'stride': 1,
            'hidden_dim': 256,
            'encoder_output_dim': 128,
            'dropout': 0.35,
            'static_feature_len': 36,
            'feature_len': 35,
            'num_epoch': 400,
            'timestep': 20,
            'num_classes': 20,
            'beta1': 0.9,
            'beta2': 0.99
        }
        aug_params = {
            'jitter_scale_ration': 0.001,
            'jitter_ratio': 0.001,
            'max_seg': 5,
            'batch_size': 512,
            'drop_last': True
        }
        loss_params = {
            'num_epoch': 400,
            'lr': 0.001,
            'batch_size': 512,
            'temperature': 0.2,
            'use_cosine_similarity': True
        }
        return model_params, aug_params, loss_params

if __name__ == '__main__':
    dataformatter = MobiactFormatter()
    dataset_dir = '../datasets/mobiact_preprocessed/'
    X_train, y_train, X_valid, y_valid, X_test, y_test = dataformatter.split_data(dataset_dir)
