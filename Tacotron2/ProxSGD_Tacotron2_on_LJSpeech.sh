declare -a seed=(1 2 3)
declare -a gpu=(0 1 2)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --output ./output/ \
    --optimizer ProxSGD \
    --learning-rate 1e0 \
    --lambda_ 2.5e-6 \
    --seed ${seed[i]} &
done