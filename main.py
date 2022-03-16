import torch

import os
import numpy as np
from datetime import datetime
import argparse
from libs.utils import _logger, set_requires_grad
from libs.utils import _calc_metrics, copy_Files
from libs.dataloader import data_generator
from libs.trainer import Trainer, model_evaluate
from models.TFCC import TFCC
from models.encoder import cnn_encoder
from models.static import StaticEmbedding, StaticVariableSelection
from data_formatters.configs import ExperimentConfig

start_time = datetime.now()

parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=42, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--static_use', action=argparse.BooleanOptionalAction)
parser.add_argument('--selected_dataset', default='mobiact', type=str)
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cpu', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
method = 'TS-TCC'
run_description = args.run_description

SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# exec(f'from config_files.{data_type}_Configs import Config as Configs')
config = ExperimentConfig(data_type)
formatter = config.make_data_formatter()

# load_dataset
dataset_dir = 'datasets/' + config.data_csv_path
train, valid, test = formatter.split_data(dataset_dir=dataset_dir)
model_params, aug_params, loss_params = formatter.get_experiment_params()

train_loader, valid_loader, test_loader = data_generator(dataset_dir, model_params, aug_params, training_mode)
logger.debug("Data loaded ...")
#########################

static_embedding_model = StaticEmbedding(model_params, device).to(device)
static_variable_selection = StaticVariableSelection(model_params, device).to(device)
encoder = cnn_encoder(model_params).to(device)
tfcc_model = TFCC(model_params, device).to(device)

if training_mode == "fine_tune":
    # load saved model
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", "saved_models"))
    chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = encoder.state_dict()
    del_list = ['logits']
    pretrained_dict_copy= pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)

if training_mode == 'train_linear' or "tl" in training_mode:
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description, f"self_supervised_seed_{SEED}", 'saved_models'))
    chkpoint = torch.load(os.path.join(load_from, "ckpt_last.pt"), map_location=device)
    pretrained_dict = chkpoint["model_state_dict"]
    model_dict = encoder.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    del_list = ['logits']
    pretrained_dict_copy = pretrained_dict.copy()
    for i in pretrained_dict_copy.keys():
        for j in del_list:
            if j in i:
                del pretrained_dict[i]
    model_dict.update(pretrained_dict)
    encoder.load_state_dict(model_dict)
    set_requires_grad(encoder, pretrained_dict, requires_grad=False)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=model_params['lr'], betas=(model_params['beta1'], model_params['beta2']), weight_decay=3e-4)
tfcc_optimizer = torch.optim.Adam(tfcc_model.parameters(), lr=model_params['lr'], betas=(model_params['beta1'], model_params['beta2']), weight_decay=3e-4)
static_variable_selection_optimizer = torch.optim.Adam(static_variable_selection.parameters(), lr=model_params['lr'], betas=(model_params['beta1'], model_params['beta2']), weight_decay=3e-4)

if training_mode == "self_supervised":
    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), data_type)

Trainer(encoder, tfcc_model, static_embedding_model, static_variable_selection, encoder_optimizer, tfcc_optimizer, static_variable_selection_optimizer, train_loader, valid_loader, test_loader, train, device, logger, loss_params, experiment_log_dir, training_mode, static_use=True)

if training_mode != "self_supervised":
    outs = model_evaluate(encoder, tfcc_model, static_variable_selection, test_loader, device, training_mode)
    total_loss, total_acc, pred_labels, true_labels = outs
    _calc_metrics(pred_labels, true_labels, experiment_log_dir, args.home_path)

logger.debug(f"Training time is: {datetime.now() - start_time}")