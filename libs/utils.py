import torch
import torch.nn as nn

def extract_cols_from_data_type(data_type, colmn_definition):
    return [
        tup[0]
        for tup in colmn_definition
        if tup[1] == data_type
    ]