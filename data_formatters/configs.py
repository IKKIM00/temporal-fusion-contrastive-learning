import os
import data_formatters.mobiact

class ExperimentConfig(object):
    default_experiments = ['mobiact']

    def __init__(self, experiment='mobiact'):
        self.experiment = experiment

        if experiment not in self.default_experiments:
            raise ValueError('Unrecognised experiment={}'.format(experiment))

    @property
    def data_csv_path(self):
        csv_map = {
            'mobiact': 'mobiact_preprocessed/'
        }
        return csv_map

    def make_data_formatter(self):
        data_formatter_class = {
            'mobiact': data_formatters.mobiact.MobiactFormatter
        }
        return data_formatter_class[self.experiment]()