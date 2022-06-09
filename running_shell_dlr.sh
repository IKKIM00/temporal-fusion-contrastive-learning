# +
# mkdir -p ./log/tfcl/
python main.py --model_type 'TFCL' --experiment_description 'no_static_use'--static_use --no-sampler_use --dataset 'dlr' --device '2,3' --training_mode 'self_supervised' --loss_func 'focal' | tee log/tfcl/TFCL_static_dlr_self_supervised.txt
python main.py --model_type 'TFCL' --experiment_description 'no_static_use'--static_use --no-sampler_use --dataset 'dlr' --device '2,3' --training_mode 'supervised' --loss_func 'focal' | tee log/tfcl/TFCL_static_dlr_supervised.txt

python main.py --model_type 'TFCL' --experiment_description 'static_use'--static_use --no-sampler_use --dataset 'dlr' --device '2,3' --training_mode 'self_supervised' --loss_func 'focal' | tee log/tfcl/TFCL_static_dlr_self_supervised.txt
python main.py --model_type 'TFCL' --experiment_description 'static_use'--static_use --no-sampler_use --dataset 'dlr' --device '2,3' --training_mode 'supervised' --loss_func 'focal' | tee log/tfcl/TFCL_static_dlr_supervised.txt
