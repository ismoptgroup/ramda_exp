declare -a seed=(1 2 3)

for ((i=0; i<${#seed[@]}; i++));
do
horovodrun -np 8 -H localhost:8 python train.py \
    --optimizer="ProxAdamW" \
    --fp16-allreduce \
    --use-mixed-precision \
    --lr=1e-3 \
    --lambda_=1.3e-7 \
    --weight-decay=0.0 \
    --seed=${seed[i]}
done