#!/bin/bash
#SBATCH -JNLP-exp
#SBATCH -N1 -n1
#SBATCH --mem-per-gpu 30GB
#SBATCH -G H100:4
#SBATCH -t 00:30:00
#SBATCH -oReport-%j.out

module load python/3.10.10
module load anaconda3/2022.05.0.1
module load gcc/12.3.0
module load mvapich2/2.3.7-1
module load cuda/12.1.1

# experiment name
EXP_NAME='baseline_e30_mbart'

echo "Launching distributed eval for" $EXP_NAME

MASTER_PORT=12355

if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
else
    # Fallback: Count the number of visible GPUs
    NUM_GPUS=$(nvidia-smi -L | wc -l)
fi

echo "Number of GPUs allocated: $NUM_GPUS"

SCRIPT_PATH="distributed_bleu.py"

MODEL_PATH="model_weights/model_epoch_6.pt"

OUTPUT_FILE="bleu_results.csv"

DECODING_STRATEGY="TOP_P"
BATCH_SIZE=64 # NOTE: BEAM_SEARCH means batch size has to be 1
# BEAM_WIDTH=5
P=0.9


echo "Starting distributed BLEU score calculation..."

~/.conda/envs/nlp/bin/python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --rdzv_id=bleu_calc \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:$MASTER_PORT \
    $SCRIPT_PATH \
    --model_path $MODEL_PATH \
    --output_file $OUTPUT_FILE \
    --decoding_strategy $DECODING_STRATEGY \
    --batch_size $BATCH_SIZE \
    --p $P
    # --beam_width $BEAM_WIDTH
