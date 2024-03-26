declare -a seed=(1 2 3)
declare -a gpu=(0 0 0)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --optimizer="ProxAdamW" \
    --lr=1e-3 \
    --lambda_=3.5e-3 \
    --weight-decay=0.0 \
    --seed=${seed[i]} &
done