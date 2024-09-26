#!/bin/bash
#SBATCH -JSNN-exp
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 30GB
#SBATCH -G A100:1
#SBATCH -t 01:00:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3/2022.05.0.1
module load gcc/12.3.0
module load mvapich2/2.3.7-1
module load cuda/12.1.1

echo "Launching Training"

EXP_NAME='baseline'
D_MODEL=256
NUM_LAYERS=4
NUM_HEADS=4
D_FF=1024
MAX_SEQ_LENGTH=100
EPOCHS=5
LEARNING_RATE=5e-5
BATCH_SIZE=16
DEVICE='cuda'

echo $EXP_NAME

 ~/.conda/envs/torch/bin/python main.py \
    --exp_name $EXP_NAME \
    --d_model $D_MODEL \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --d_ff $D_FF \
    --max_seq_length $MAX_SEQ_LENGTH \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --device $DEVICE