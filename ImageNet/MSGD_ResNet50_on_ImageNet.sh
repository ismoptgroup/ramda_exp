declare -a seed=(1 2 3)

for ((i=0; i<${#seed[@]}; i++));
do
horovodrun -np 8 -H localhost:8 python train.py \
    --optimizer="MSGD" \
    --fp16-allreduce \
    --use-mixed-precision \
    --lr=1e-1 \
    --lambda_=0.0 \
    --weight-decay=1e-4 \
    --seed=${seed[i]}
done