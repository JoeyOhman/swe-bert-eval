#!/bin/bash

# MODEL_NAME="KB/bert-base-swedish-cased"
MODEL_NAME="../models/base"
# MODEL_NAME="../models/large"

# run_cmd="python swedish_reviews/finetune_swe_reviews.py
run_cmd="python finetune_swe_reviews.py
        --model_name_or_path $MODEL_NAME \
        --output_dir ./results \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 64 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 1
        --num_train_epochs 1 \
        --evaluation_strategy steps \
        --save_strategy epoch \
        --skip_memory_metrics \
        --eval_steps 500 \
        --fp16 \
        --disable_tqdm 1 \
        --weight_decay 0.01 \
        --learning_rate 2e-5 \
        --max_input_length 128 \
        --data_fraction 1.0
        "

# KB test acc: 0.9731
# KB eval acc: 0.9751
# base test acc: 0.969
# base eval acc: 0.9712

echo $run_cmd
$run_cmd
