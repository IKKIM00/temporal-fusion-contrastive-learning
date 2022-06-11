training_mode="self_supervised"
dataset="mobiact dlr"

: << "END"

$n : argument order in Command Line

How to use (in Terminal):
bash runnning_shell.sh TFCL mobiact self_supervised static_use 2,3

$1 : model ( TFCL, SimclrHAR, CSSHAR, CPCHAR )
$2 : static_use  ( yes, no )
$3 : device ( 2,3 or 0,1,2,3 ... )

END

for data in $dataset
do
  for mode in $training_mode
  do
    if [ $mode == 'self_supervised' ]; then
      mkdir -p log/$1/$data/$mode/no_static/
      python main.py --model_type $1 --experiment_description no-static_use --no-sampler_use\
     --dataset $data --device $3 --training_mode $mode --loss_func 'focal' | tee log/$1/$data/$mode/no_static/log.txt
    else
      mkdir -p log/$1/$data/$mode/no_static/
      python main.py --model_type $1 --experiment_description no-static_use --no-sampler_use\
     --dataset $data --device $3 --training_mode $mode --loss_func 'focal' | tee log/$1/$data/$mode/no_static/log.txt
    fi
  done
done



