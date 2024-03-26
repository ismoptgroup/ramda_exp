declare -a seed=(1 2 3)
declare -a gpu=(0 1 2)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --output ./output/ \
    --optimizer RMDA \
    --learning-rate 2.5e0 \
    --lambda_ 1e-6 \
    --seed ${seed[i]} &
done