#!/bin/bash

export WANDB_MODE=offline
# export WANDB_DISABLED=true

which python
/bin/hostname -s

# MODEL_NAME="KB/bert-base-swedish-cased"
# MODEL_NAME=
# MODEL_NAME="../models/base"
MODEL_NAME="../models/large"

# --model_name_or_path $MODEL_NAME \
# /ceph/hpc/home/eujoeyo/group_space/joey/workspace/ner_kram/arielbert/base \

# run_cmd="python swedish_reviews/finetune_swe_reviews.py
run_cmd="python finetune_swe_reviews.py
        --model_name_or_path bert-base-cased \
        --output_dir ./results \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 64 \
        --overwrite_output_dir \
        --gradient_accumulation_steps 1
        --num_train_epochs 3 \
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

# 1 epoch
# KB test acc: 0.9731
# KB eval acc: 0.9751
# base test acc: 0.969
# base eval acc: 0.9712
# large test acc: 0.9751
# large eval acc: 0.9758

# 3 epochs with warmup ratio 0.05
# KB test acc: 0.9781
# KB eval acc: 0.9791

# base test acc: 0.9748
# base eval acc: 0.976
# large test acc: 0.9804
# large eval acc: 0.9813

# multilingual test acc: 0.9685
# multilingual eval acc: 0.9699
# english test acc: 0.9637
# english eval acc: 0.965
# random test acc: 0.9539
# random eval acc: 0.9537

echo $run_cmd
$run_cmd
