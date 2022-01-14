#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=swe_reviews
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=2
#SBATCH --time=0-1:00:00
#SBATCH --output=logs/swe_reviews.log

# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')

PROJECT=/ceph/hpc/home/eujoeyo/group_space/joey/workspace/swe-bert-eval
LOGGING=$PROJECT/logs

srun -l --output=$LOGGING/%x_"$DATETIME".log "./run_swe_reviews.sh"
set +x

exit 0