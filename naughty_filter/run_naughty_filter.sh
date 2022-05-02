#!/bin/bash

export WANDB_MODE=offline
# export WANDB_DISABLED=true

which python
/bin/hostname -s

MODEL_NAME="KB/bert-base-swedish-cased"
# MODEL_NAME="KBLab/megatron-bert-large-swedish-cased-165k"

# --model_name_or_path $MODEL_NAME \
# /ceph/hpc/home/eujoeyo/group_space/joey/workspace/ner_kram/arielbert/base \

# run_cmd="python swedish_reviews/finetune_swe_reviews.py
run_cmd="python finetune_naughty_filter.py
        --model_name_or_path $MODEL_NAME \
        --output_dir ./results \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 32 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 2
        --num_train_epochs 5 \
        --evaluation_strategy steps \
        --save_strategy epoch \
        --skip_memory_metrics \
        --eval_steps 500 \
        --fp16 \
        --disable_tqdm 1 \
        --weight_decay 0.01 \
        --learning_rate 2e-5 \
        --max_input_length 256 \
        --warmup_ratio 0.05 \
        --data_fraction 1.0
        "


echo $run_cmd
$run_cmd
