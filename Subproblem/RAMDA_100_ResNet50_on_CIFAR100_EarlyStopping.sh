declare -a seed=(1 2 3)
declare -a gpu=(3 4 5)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="ResNet50" \
    --dataset="CIFAR100" \
    --optimizer="RAMDA" \
    --epochs=1000 \
    --lr=1e-2 \
    --lambda_=1e-6 \
    --max-iters=100 \
    --early-stopping \
    --gamma=1e-1 \
    --seed=${seed[i]} \
    --milestones 250 500 750 &
done