declare -a seed=(1 2 3)
declare -a gpu=(6 6 6)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --model="VGG19" \
    --dataset="CIFAR10" \
    --optimizer="ProxAdamW" \
    --epochs=1000 \
    --lr=1e-3 \
    --lambda_=1e-6 \
    --max-iters=100 \
    --gamma=1e-1 \
    --seed=${seed[i]} \
    --milestones 250 500 750 &
done