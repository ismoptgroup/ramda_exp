declare -a seed=(1 2 3)
declare -a gpu=(3 4 5)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="VGG19" \
    --dataset="CIFAR100" \
    --optimizer="ProxSGD" \
    --lr=1e-1 \
    --lambda_=4e-5 \
    --weight-decay=0.0 \
    --seed=${seed[i]} &
done