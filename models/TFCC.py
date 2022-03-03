import torch
import torch.nn as nn
import numpy as np
from models.attention import Seq_Transformer
from models.static_utils import static_combine_and_mask

class TFCC(nn.Module):
    def __init__(self, config, device):
        super(TFCC, self).__init__()


