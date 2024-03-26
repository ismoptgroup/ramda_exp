declare -a seed=(1 2 3)
declare -a gpu=(0 1 2)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} accelerate launch run_mim_no_trainer.py \
  --model_type vit \
  --dataset_name cifar10 \
  --optimizer RAMDA \
  --learning_rate 1e-2 \
  --lambda_ 1e-5 \
  --lr_scheduler_type multistep \
  --seed ${seed[i]} &
done