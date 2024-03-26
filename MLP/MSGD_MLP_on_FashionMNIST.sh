declare -a seed=(1 2 3)
declare -a gpu=(0 0 0)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --optimizer="MSGD" \
    --lr=1e-2 \
    --lambda_=0.0 \
    --weight-decay=1e-3 \
    --seed=${seed[i]} &
done