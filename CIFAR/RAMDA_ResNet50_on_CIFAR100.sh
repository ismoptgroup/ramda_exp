declare -a seed=(1 2 3)
declare -a gpu=(3 4 5)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="ResNet50" \
    --dataset="CIFAR100" \
    --optimizer="RAMDA" \
    --lr=1e-2 \
    --lambda_=5e-7 \
    --weight-decay=0.0 \
    --seed=${seed[i]} &
done