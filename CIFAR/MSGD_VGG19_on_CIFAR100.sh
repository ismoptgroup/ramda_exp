declare -a seed=(1 2 3)
declare -a gpu=(7 7 7)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="VGG19" \
    --dataset="CIFAR100" \
    --optimizer="MSGD" \
    --lr=1e-1 \
    --lambda_=0.0 \
    --weight-decay=5e-4 \
    --seed=${seed[i]} &
done