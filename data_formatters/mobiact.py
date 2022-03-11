import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing
import pandas as pd
import torch

BaseForamtter = data_formatters.base.BaseFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class MobiactFormatter(BaseForamtter):
    _column_definition = [
        ('height', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('weight', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT),
        ('gender', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
        ('age', DataTypes.REAL_VALUED, InputTypes.STATIC_INPUT)
    ]

    def __init__(self):
        self._cat_scalers = None
        self._real_scalers = None
        self._num_classes_per_cat_input = None

    def split_data(self, dataset_dir):
        print('Formatting train-valid-test static data splits')

        person_info = pd.read_csv(dataset_dir + 'person_info.csv', index_col=0)
        person_info = person_info[['age', 'height', 'weight', 'gender']]
        train = person_info.iloc[:50]
        valid = person_info.iloc[50: 57]
        test = person_info.iloc[57:]

        self.set_scalers(train)
        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        print('Setting scalers with static training data')

        column_definition = self.get_column_definition()

        self.real_inputs = utils.extract_cols_from_data_type(DataTypes.REAL_VALUED, column_definition)
        self.categorical_inputs = utils.extract_cols_from_data_type(DataTypes.CATEGORICAL, column_definition)

        real_data = df[self.real_inputs].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(real_data)

        categorical_scalers = {}
        num_classes = []
        for col in self.categorical_inputs:
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes

    def transform_inputs(self, df):
        output = df.copy()

        output[self.real_inputs] = self._real_scalers.transform(df[self.real_inputs].values)
        for col in self.categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)
        return torch.tensor(output.values)

    def get_model_params(self):
        model_params = {
            'input_size': 6,
            'kernel_size': 43,
            'stride': 1,
            'hidden_dim': 64,
            'output_dim': 32,
            'dropout': 0.35,
            'n_predicts': 12,
            'feature_len': 29,
            'num_epoch': 100,
            'batch_size': 512,
            'timestep': 10,
            'num_classes': 13,
            'lr': 3e-4,
            'beta1': 0.9,
            'beta2': 0.99
        }
        aug_params = {
            'jitter_scale_ration': 0.001,
            'jitter_ratio': 0.001,
            'max_seg': 5
        }
        loss_params = {
            'num_epoch': 100,
            'batch_size': 512,
            'temperature': 0.2,
            'use_cosine_similarity': True
        }
        return model_params, aug_params, loss_params
