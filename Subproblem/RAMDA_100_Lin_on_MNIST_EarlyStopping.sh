declare -a seed=(1)

for ((i=0; i<${#seed[@]}; i++));
do
python train.py \
    --model="Lin" \
    --dataset="MNIST" \
    --optimizer="RAMDA" \
    --epochs=500 \
    --lr=1e-2 \
    --lambda_=1e-3 \
    --max-iters=100 \
    --early-stopping \
    --gamma=1e-1 \
    --seed=${seed[i]} \
    --milestones 100 200 300 400
done