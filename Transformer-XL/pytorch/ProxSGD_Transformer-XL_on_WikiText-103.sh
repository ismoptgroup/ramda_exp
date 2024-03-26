declare -a seed=(1 2 3)
declare -a gpu=(0 1 2)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --config_file ProxSGD_wt103_base.yaml \
    --work_dir ./output/ \
    --seed ${seed[i]} &
done