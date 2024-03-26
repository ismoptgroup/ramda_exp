declare -a seed=(1 2 3)
declare -a gpu=(0 1 2)

for ((i=0; i<${#seed[@]}; i++));
do
CUDA_VISIBLE_DEVICES=${gpu[i]} python train.py \
    --output ./output/ \
    --optimizer RAMDA \
    --learning-rate 2.5e-3 \
    --lambda_ 6e-7 \
    --seed ${seed[i]} &
done