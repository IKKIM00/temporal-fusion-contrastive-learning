# Model Architecture

![](/Users/inkyungkim/Documents/Git/temporal-fusion-representation-learning/images/main_model_arch.eps "Model Architecture")

# Requirments

- `python` == 3.9
- `pytorch` == 1.9.1

# How to start

## Preprocess dataset
1. Download dataset from following urls
   1. [MobiAct Dataset](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/)
   2. [DRL Dataset](https://www.dlr.de/kn/en/desktopdefault.aspx/tabid-12705/22182_read-50785/)
2. Run following ipynb files for each dataset
   1. MobiAct Datset - `datasets/preprocess_mobiact.ipynb`
   2. DLR Dataset - 'datasets/preprocess_dlr.ipynb'

## Hyper-parameters for `main.py`
- `--experiment description` : set experiment description. Default : `Exp1`
- `--run_description`: set experiment description. Default: `run1`
- `--seed`: seed value. Default: `42`
- `--encoder_model`: choose encoder model. Default: `CNN`
- `--training_mode`: choose training mode between `self_supervised`, `fine_tune`, `train_linear`, `supervised`. Default: `supervised`
- `--loss_func`: choose between `focal` and `cross_entropy`. Default: `cross_entropy`
- `--static_use`: choose whether to use static data. Use static data: `--static_use`, not use static data: `--no-static_use`
- `--sampler_use`: choose whether to use imbalance dataset sampler. Use sampler: `--sampler_use`, not use sampler use: `--no-sampler_use`
- `--dataset`: choose dataset. Default: `mobiact`
- `--logs_save_dir`: saving directory. Default: `experiments_logs`
- `--device`: choose device. Default: `cpu`
- `--home_path`: home directory. Default: current directory

## Training
### with static data + sampler
> run `python main.py --training_mode 'self_supervised --static_use --sampler_use`

### with static data + no sampler
> run `python main.py --training_mode 'self_supervised' --static_use --no-sampler_use`

### without static data + sampler
> run `python main.py --training_mode 'self_supervised' --no-static_use --sampler_use`

### without static data + no sampler
> run `python main.py --training_mode 'self_supervised' --no-static_use  --no-sampler_use`