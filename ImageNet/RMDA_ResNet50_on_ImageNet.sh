declare -a seed=(1 2 3)

for ((i=0; i<${#seed[@]}; i++));
do
horovodrun -np 8 -H localhost:8 python train.py \
    --optimizer="RMDA" \
    --fp16-allreduce \
    --use-mixed-precision \
    --lr=1e0 \
    --momentum=1e-1 \
    --lambda_=1.35e-5 \
    --weight-decay=0.0 \
    --seed=${seed[i]}
done