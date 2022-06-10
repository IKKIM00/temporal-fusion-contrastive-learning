aug_list=("jitter" "scale" "jitter_scale" "permutation" "permutation_jitter" "rotation" "invert" "timeflip" "shuffle" "warp")
training_mode="self_supervised supervised"
dataset="mobiact dlr"

: << "END"

$n : argument order in Command Line

How to use (in Terminal):
bash runnning_shell.sh TFCL mobiact self_supervised static_use 2,3

$1 : model ( TFCL, SimclrHAR, CSSHAR, CPCHAR )
$2 : static_use  ( static_use, no_static_use )
$3 : device ( 2,3 or 0,1,2,3 ... )

END

for data in $dataset
do
  for mode in $training_mode
  do
    if [ $mode == 'self_supervised' ]; then
      for ((i=0; i<10; i++))
      do
        for ((j=i; j<10; j++))
        do
          if [ $2 == 'yes' ]; then
            echo "mkdir -p log/$1/$data/$mode/static_use/${aug_list[${i}]}_and_${aug_list[${j}]}"
            echo "python main.py --model_type $1 --experiment_description static_use --static_use --no-sampler_use\
             --dataset $data --device $3 --training_mode $mode --loss_func 'focal'\
             --aug_method1 $aug1 --aug_method2 $aug2 | tee log/$1/$data/$mode/static_use/${aug_list[${i}]}_and_${aug_list[${j}]}/log.txt"
          else
            echo "mkdir -p log/$1/$data/$mode/no_static/${aug_list[${i}]}_and_${aug_list[${j}]}"
            echo "python main.py --model_type $1 --experiment_description no-static_use --no-sampler_use\
             --dataset $data --device $3 --training_mode $mode --loss_func 'focal'\
             --aug_method1 $aug1 --aug_method2 $aug2 | tee log/$1/$data/$mode/no_static/${aug_list[${i}]}_and_${aug_list[${j}]}/log.txt"
          fi
        done
      done
    else
      if [ $2 == 'yes' ]; then
        echo "mkdir -p log/$1/$data/$mode/static_use/"
        echo "python main.py --model_type $1 --experiment_description static_use --static_use --no-sampler_use\
       --dataset $data --device $3 --training_mode $mode --loss_func 'focal' | tee log/$1/$data/$mode/static_use/log.txt"
      else
        echo "mkdir -p log/$1/$data/$mode/no_static/"
        echo "python main.py --model_type $1 --experiment_description no-static_use --no-sampler_use\
       --dataset $data --device $3 --training_mode $mode --loss_func 'focal' | tee log/$1/$data/$mode/no_static/log.txt"
     fi
    fi
  done
done



