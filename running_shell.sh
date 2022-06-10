aug_list=("jitter" "scale" "jitter_scale" "permutation" "permutation_jitter" "rotation" "invert" "timeflip" "shuffle" "warp")
# model_list="TFCL SimclrHAR CSSHAR CPCHAR"

: << "END"

$n : argument order in Command Line

How to use (in Terminal):
bash runnning_shell.sh TFCL mobiact self_supervised static_use 2,3

$1 : model ( TFCL, SimclrHAR, CSSHAR, CPCHAR )
$2 : dataset ( mobiact, dlr )
$3 : training_mode ( self_supervised, train_linear, fine_tune )
$4 : static_use  ( static_use, no_static_use )
$5 : device ( 2,3 or 0,1,2,3 ... )

END

if [ $3 == 'self_supervised' ]; then
  for ((i=0; i<10; i++))
  do
    for ((j=i; j<10; j++))
    do
      mkdir -p log/$1/$2/$3/$4/${aug_list[${i}]}_and_${aug_list[${j}]}
      python main.py --model_type $1 --experiment_description $1_$2_$3_$4_${aug_list[${i}]}_and_${aug_list[${j}]} --$4 --no-sampler_use\
       --dataset $2 --device $5 --training_mode $3 --loss_func 'focal'\
       --aug_method1 $aug1 --aug_method2 $aug2 | tee log/$1/$2/$3/$4/${aug_list[${i}]}_and_${aug_list[${j}]}/log.txt
    done
  done
else
  mkdir -p log/$1/$2/$3/$4/
  python main.py --model_type $1 --experiment_description $1_$2_$3_$4 --no-sampler_use\
 --dataset $2 --device $5 --training_mode $3 --loss_func 'focal' | tee log/$1/$2/$3/$4/log.txt
fi
