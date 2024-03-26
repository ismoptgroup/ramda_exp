declare -a seed=(1 2 3)
declare -a gpu=(0 1 2)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} accelerate launch run_mim_no_trainer.py \
  --model_type vit \
  --dataset_name cifar10 \
  --optimizer ProxAdamW \
  --learning_rate 1e-3 \
  --lambda_ 1e-4 \
  --lr_scheduler_type linear \
  --seed ${seed[i]} &
done