mkdir -p ./log
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'self_supervised' --loss_func 'focal' | tee log/TFCL_static_dlr_self_supervised.txt
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'fine_tune' --loss_func 'focal'| tee log/TFCL_static_dlr_fine_tune.txt
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'train_linear' --loss_func 'focal' | tee log/TFCL_static_dlr_train_linear.txt
python main.py --model_type 'TFCL' --static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'supervised' --loss_func 'focal' | tee log/TFCL_static_dlr_supervised.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'self_supervised' --loss_func 'focal' | tee log/TFCL_Nostatic_dlr_self_supervised.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'fine_tune' --loss_func 'focal' | tee log/TFCL_Nostatic_dlr_fine_tune.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'train_linaer' --loss_func 'focal' | tee log/TFCL_Nostatic_dlr_train_linear.txt
python main.py --model_type 'TFCL' --no-static_use --no-sampler_use --dataset 'dlr' --device 'cuda:2' --training_mode 'supervised' --loss_func 'focal' | tee log/TFCL_Nostatic_dlr_supervised.txt
