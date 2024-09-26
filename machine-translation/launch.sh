#!/bin/bash

# # customize for your envnaem
# conda activate torch

EXP_NAME='baseline'
D_MODEL=512
NUM_LAYERS=6
NUM_HEADS=8
D_FF=2048
MAX_SEQ_LENGTH=128
EPOCHS=10
LEARNING_RATE=5e-5
BATCH_SIZE=16
DEVICE='mps'

echo $EXP_NAME

python main.py \
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