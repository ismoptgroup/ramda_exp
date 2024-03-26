declare -a seed=(1 2 3)
declare -a gpu=(0 1 2)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="VGG19" \
    --dataset="CIFAR10" \
    --optimizer="RMDA" \
    --lr=1e-1 \
    --momentum=1e-1 \
    --lambda_=1.5e-4 \
    --weight-decay=0.0 \
    --seed=${seed[i]} &
done