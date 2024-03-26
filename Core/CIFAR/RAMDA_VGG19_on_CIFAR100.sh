declare -a seed=(1 2 3)
declare -a gpu=(7 7 7)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="VGG19" \
    --dataset="CIFAR100" \
    --optimizer="RAMDA" \
    --lr=1e-2 \
    --lambda_=2e-7 \
    --weight-decay=0.0 \
    --seed=${seed[i]} &
done