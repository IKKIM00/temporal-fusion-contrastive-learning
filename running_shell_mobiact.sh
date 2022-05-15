mkdir -p ./log
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'self_supervised' --loss_func 'focal' | tee log/TFCL_static_mobiact_self_supervised.txt
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'fine_tune' --loss_func 'focal'| tee log/TFCL_static_mobiact_fine_tune.txt
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'train_linear' --loss_func 'focal' | tee log/TFCL_static_mobiact_train_linear.txt
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'supervised' --loss_func 'focal' | tee log/TFCL_static_mobiact_supervised.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'self_supervised' --loss_func 'focal' | tee log/TFCL_Nostatic_mobiact_self_supervised.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'fine_tune' --loss_func 'focal' | tee log/TFCL_Nostatic_mobiact_fine_tune.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'train_linaer' --loss_func 'focal' | tee log/TFCL_Nostatic_mobiact_train_linear.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'mobiact' --device 'cuda:1' --training_mode 'supervised' --loss_func 'focal' | tee log/TFCL_Nostatic_mobiact_supervised.txt
