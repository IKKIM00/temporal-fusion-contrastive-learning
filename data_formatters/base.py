import abc
import enum

class DataTypes(enum.IntEnum):
    REAL_VALUED = 0
    CATEGORICAL = 1

class InputTypes(enum.IntEnum):
    STATIC_INPUT = 0

class BaseFormatter(abc.ABC):

    @property
    @abc.abstractmethod
    def _column_definition(self):
        """define data types"""
        raise NotImplementedError()

    @abc.abstractmethod
    def split_data(self, dataset_dir):
        """preprocess dataset"""
        raise NotImplementedError()

    @abc.abstractmethod
    def set_scalers(self, df):
        """set scalers for dataset"""
        raise NotImplementedError()

    @abc.abstractmethod
    def transform_inputs(self, df):
        """transform dataset using set scalers"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_params(self):
        """define model configs

        :returns - Dictionaires of fixed params

            model_params = {
                'input_dim': 1,
                'kernel_size': 8,
                'stride': 1,
                'hidden_dim': 64,
                'output_dim': 128,
                'dropout': 0.35,
                'n_predicts': 12,
                'feature_len': 24,
                'num_epoch': 100,
                'batch_size': 512,
                'timestep': 10,
                'num_classes': 2,
                }
            aug_params = {
                'jitter_scale_ration': 0.001,
                'jitter_ratio': 0.001,
                'max_seg': 5
                }
            loss_params = {
                'temperature': 0.2,
                'use_cosine_similarity': True
                }
        """
    @property
    def num_classes_per_cat_input(self):
        return self._num_classes_per_cat_input

    def get_column_definition(self):
        column_definition = self._column_definition

        real_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.REAL_VALUED
        ]
        categorical_inputs = [
            tup for tup in column_definition if tup[1] == DataTypes.CATEGORICAL
        ]
        return real_inputs + categorical_inputs

    def _get_input_columns(self):
        return[tup[0] for tup in self.get_column_definition()]

    def _get_input_indicies(self):

        def _extract_tupes_from_data_type(data_type, defn):
            return [tup for tup in defn if tup[1]==data_type]

        def _get_locations(defn):
            return [i for i, tup in enumerate(defn)]

        column_definition = [
            tup for tup in self.get_column_definition()
        ]

        categorical_inputs = _extract_tupes_from_data_type(DataTypes.CATEGORICAL, column_definition)
        real_inputs = _extract_tupes_from_data_type(DataTypes.REAL_VALUED, column_definition)

        locations = {
            'category_counts': self.num_classes_per_cat_input,
            'static_regular_inputs': _get_locations(real_inputs),
            'static_categorical_inputs': _get_locations(categorical_inputs)
        }
        return locations

    def get_experiment_params(self):
        model_params, aug_params, loss_params = self.get_model_params()
        model_params['column_definition'] = self.get_column_definition()
        model_params.update(self._get_input_indicies())
        return model_params, aug_params, loss_params
