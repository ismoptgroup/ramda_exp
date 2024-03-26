declare -a seed=(1 2 3)
declare -a gpu=(3 4 5)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="ResNet50" \
    --dataset="CIFAR100" \
    --optimizer="ProxAdamW" \
    --lr=1e-3 \
    --lambda_=1e-6 \
    --weight-decay=0.0 \
    --seed=${seed[i]} &
done