import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl

import os
import numpy as np
from datetime import datetime
import argparse
from libs.utils import set_requires_grad
from libs.utils import copy_Files
from models.loss import FocalLoss
from libs.dataloader import data_generator
from libs.trainer import SSL, train_ssl, FineTune, TrainLinear, train_downstream_task, Supervised
from models.autoregressive import BaseAR, SimclrHARAR, CSSHARAR, CPCHARAR
from models.logit import BaseLogit, SimclrLogit, CSSHARLogit, CPCHARLogit
from models.encoder import BaseEncoder, SimclrHAREncoder, CSSHAREncoder, CPCHAR
from models.static import StaticEncoder
from data_formatters.configs import ExperimentConfig

# +
import warnings
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

import logging
logging.getLogger("lightning").setLevel(logging.WARNING)
# -

start_time = datetime.now()

parser = argparse.ArgumentParser()
######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--model_type', default='TFCL', type=str, help='TFCL, SimclrHAR, CSSHAR')
parser.add_argument('--training_mode', default='supervised', type=str)
parser.add_argument('--loss_func', default='cross_entropy', type=str)
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--aug_method1', default='jitter_scale', type=str)
parser.add_argument('--aug_method2', default='permutation_jitter', type=str)
parser.add_argument('--static_use', action=argparse.BooleanOptionalAction)
parser.add_argument('--sampler_use', action=argparse.BooleanOptionalAction)
parser.add_argument('--dataset', default='mobiact', type=str)
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--home_path', default=home_dir, type=str)
args = parser.parse_args()

device = args.device
experiment_description = args.experiment_description
data_type = args.dataset
training_mode = args.training_mode
method = args.model_type
loss_func = args.loss_func
batch_size = args.batch_size
aug_method1 = args.aug_method1
aug_method2 = args.aug_method2
static_use = args.static_use
sampler_use = args.sampler_use

print(f"Args: {args}")

SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

experiment_log_dir = os.path.join(logs_save_dir, method, experiment_description, training_mode + f"_seed_{SEED}_{data_type}_aug1_{aug_method1}_aug2_{aug_method2}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

print("=" * 45)
print(f'Dataset: {data_type}')
print(f'Method:  {method}')
print(f'Mode:    {training_mode}')
print(f'Augmentation 1: {aug_method1}, Augmentation 2: {aug_method2}')
print("=" * 45)

# exec(f'from config_files.{data_type}_Configs import Config as Configs')
config = ExperimentConfig(data_type)
formatter = config.make_data_formatter()

# load_dataset
dataset_dir = 'datasets/' + config.data_csv_path
X_train, y_train, X_valid, y_valid, X_test, y_test = formatter.split_data(dataset_dir=dataset_dir, training_mode=training_mode)
model_params, aug_params, loss_params = formatter.get_experiment_params()

model_params_df = pd.DataFrame.from_dict(model_params, orient='index')
aug_params_df = pd.DataFrame.from_dict(aug_params, orient='index')
loss_params_df = pd.DataFrame.from_dict(loss_params, orient='index')

model_params_df.to_csv(experiment_log_dir + '/model_params.csv')
aug_params_df.to_csv(experiment_log_dir + '/aug_params.csv')
loss_params_df.to_csv(experiment_log_dir + '/loss_params.csv')

train_loader, valid_loader, test_loader = data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test, aug_params,
                                                         data_type, aug_method1, aug_method2, batch_size, training_mode,
                                                         use_sampler=sampler_use)
print("Data loaded ...")

loss_funcs = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'focal': FocalLoss()
}

static_encoder = StaticEncoder(model_params)

if method == 'TFCL':
    encoder = BaseEncoder(model_params, static_use)
    logit = BaseLogit(model_params,static_use)
    autoregressive = BaseAR(model_params, static_use)
elif method == 'SimclrHAR':
    encoder = SimclrHAREncoder(model_params)
    logit = SimclrLogit(model_params)
    autoregressive = SimclrHARAR()
elif method == 'CSSHAR':
    encoder = CSSHAREncoder(model_params)
    logit = CSSHARLogit(model_params)
    autoregressive = CSSHARAR(model_params)
elif method == 'CPCHAR':
    encoder = CPCHAR(model_params)
    logit = CPCHARLogit(model_params)
    autoregressive = CPCHARAR(model_params)
else:
    print(f"Not Supported Method")


lr = loss_params['lr']

if training_mode == "self_supervised":
    copy_Files(os.path.join(logs_save_dir, experiment_description, method), data_type)
    
    print()
    print('-' * 20)
    print("Start Self-Supervised Learning")
    print('-' * 20)
    print()
    
    model = SSL(model_type=method,
                encoder=encoder,
                autoregressive=autoregressive,
                static_encoder=static_encoder,
                static_use=static_use,
                loss_params=loss_params,
                lr=lr,
                batch_size=batch_size
                )
    trained_model_path = train_ssl(
        train_loader=train_loader,
        model=model,
        checkpoint_dir=experiment_log_dir,
        gpus=device
        )
    
    best_model = SSL.load_from_checkpoint(checkpoint_path=trained_model_path)

    trained_encoder_state_dict = best_model.encoder.state_dict()
    encoder.load_state_dict(trained_encoder_state_dict)

    if static_use:
        trained_static_encoder_state_dict = best_model.static_encoder.state_dict()
        static_encoder.load_state_dict(trained_static_encoder_state_dict)

    if method == 'CPCHAR':
        trained_autoregressive_state_dict = best_model.autoregressive.state_dict()
        autoregressive.load_state_dict(trained_autoregressive_state_dict)
    
    print()
    print('-' * 20)
    print(f"Start Fine Tuning")
    print('-' * 20)
    print()

    train_loader, valid_loader, test_loader = data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                             aug_params,
                                                             data_type, aug_method1, aug_method2, batch_size,
                                                             training_mode='fine_tune',
                                                             use_sampler=sampler_use)
    if method != 'CPCHAR':
        finetune_model = FineTune(
            model_type=method,
            training_mode='fine_tune',
            encoder=encoder,
            static_encoder=static_encoder,
            logits=logit,
            static_use=static_use,
            criterion=loss_funcs[loss_func],
            lr=lr
        )
    else:
        finetune_model = FineTune(
            model_type=method,
            training_mode='fine_tune',
            encoder=encoder,
            static_encoder=static_encoder,
            logits=logit,
            autoregressive=autoregressive,
            static_use=static_use,
            criterion=loss_funcs[loss_func],
            lr=lr
        )
    fine_tune_results, lables, acc, precision, recall, f1 = train_downstream_task(
                                                                train_loader=train_loader,
                                                                valid_loader=valid_loader,
                                                                test_loader=test_loader,
                                                                model=finetune_model,
                                                                gpus=device,
                                                                checkpoint_dir=experiment_log_dir,
                                                                training_mode='fine_tune'
    )
    print(f"Fine Tune Accuracy     : {acc:0.4f} \n Fine Tune F1     : {f1:0.4f} \n "
          f"Fine Tune Precision     : {precision:0.4f} \n Fine Tune Recall     : {recall:0.4f}")
    
    print()
    print('-' * 20)
    print(f"Start Train Linear")
    print('-' * 20)
    print()

    encoder.load_state_dict(trained_encoder_state_dict)
    set_requires_grad(encoder, trained_encoder_state_dict, requires_grad=False)

    if static_use:
        static_encoder.load_state_dict(trained_static_encoder_state_dict)
    if method == 'CPCHAR':
        autoregressive.load_state_dict(trained_autoregressive_state_dict)
        set_requires_grad(autoregressive, trained_autoregressive_state_dict, requires_grad=False)

    train_loader, valid_loader, test_loader = data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                             aug_params,
                                                             data_type, aug_method1, aug_method2, batch_size,
                                                             training_mode='train_linear',
                                                             use_sampler=sampler_use)
    if method != 'CPCHAR':
        finetune_model = TrainLinear(
            model_type=method,
            training_mode='train_linear',
            encoder=encoder,
            static_encoder=static_encoder,
            logits=logit,
            static_use=static_use,
            criterion=loss_funcs[loss_func],
            lr=lr
        )
    else:
        finetune_model = TrainLinear(
            model_type=method,
            training_mode='train_linear',
            encoder=encoder,
            static_encoder=static_encoder,
            logits=logit,
            autoregressive=autoregressive,
            static_use=static_use,
            criterion=loss_funcs[loss_func],
            lr=lr
        )
    train_linear_results, lables, acc, precision, recall, f1 = train_downstream_task(
                                                                train_loader=train_loader,
                                                                valid_loader=valid_loader,
                                                                test_loader=test_loader,
                                                                model=finetune_model,
                                                                gpus=device,
                                                                checkpoint_dir=experiment_log_dir,
                                                                training_mode='train_linear'
    )
    print(f"Train Linear Accuracy     : {acc:0.4f} \n Train Linear F1     : {f1:0.4f} \n "
          f"Train Linear Precision     : {precision:0.4f} \n Train Linear Recall     : {recall:0.4f}")

if training_mode == 'supervised':
    train_loader, valid_loader, test_loader = data_generator(X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                             aug_params,
                                                             data_type, aug_method1, aug_method2, batch_size,
                                                             training_mode='supervised',
                                                             use_sampler=sampler_use)
    model = Supervised(
        model_type=method,
        encoder=encoder,
        static_encoder=static_encoder,
        logits=logit,
        static_use=static_use,
        criterion=loss_funcs[loss_func],
        lr=lr
    )
    supervised_results, lables, acc, precision, recall, f1 = train_downstream_task(
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        model=model,
        gpus=device,
        checkpoint_dir=experiment_log_dir,
        training_mode='supervised'
    )
    print(
        f"Supervised Accuracy     : {acc:0.4f} \n Supervised F1     : {f1:0.4f} \n "
        f"Supervised Precision     : {precision:0.4f} \n Supervised Recall     : {recall:0.4f}")
print(f"Training time is: {datetime.now() - start_time}")
