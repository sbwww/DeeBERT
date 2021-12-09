#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

PATH_TO_DATA=$1

MODEL_TYPE=$2
MODEL_SIZE=$3
DATASET=$4

MODEL_NAME=${MODEL_TYPE}-${MODEL_SIZE}
if [ $MODEL_TYPE = 'bert' ]
then
  MODEL_NAME=${MODEL_NAME}-uncased
fi

ENTROPIES="0 0.1 0.2 0.3 0.4 0.5 0.6 0.7"
# ENTROPIES="0 0.001 0.005 0.01 0.05 0.1 0.15 0.2 0.3 0.4 0.5 0.6 0.7"

for ENTROPY in $ENTROPIES; do
  echo $ENTROPY
  python examples/run_highway_glue.py \
    --model_type $MODEL_TYPE \
    --model_name_or_path ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/two_stage \
    --task_name $DATASET \
    --do_eval \
    --do_lower_case \
    --data_dir $PATH_TO_DATA/$DATASET \
    --output_dir ./saved_models/${MODEL_TYPE}-${MODEL_SIZE}/$DATASET/two_stage \
    --plot_data_dir ./plotting/ \
    --max_seq_length 128 \
    --early_exit_entropy $ENTROPY \
    --eval_highway \
    --overwrite_cache \
    --per_gpu_eval_batch_size=1
done
