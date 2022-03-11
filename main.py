import torch

import os
import numpy as np
from datetime import datetime
import argparse
from libs.utils import _logger, set_requires_grad
from libs.dataloader import data_generator
from libs.trainer import Trainer, model_evaluate
from models.TFCC import TFCC
from libs.utils import _calc_metrics, copy_Files
from models.encoder import cnn_encoder

start_time = datetime.now()


parser = argparse.ArgumentParser()

######################## Model parameters ########################