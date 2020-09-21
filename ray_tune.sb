#!/bin/bash
#SBATCH -N 1
#SBATCH --time=10080  # Time in Minutes
#SBATCH --output=slurm_logs/%j.txt
#SBATCH --partition=lgns28
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=shurjo@umich.edu
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:10

#export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
echo $CUDA_VISIBLE_DEVICES

echo $HOSTNAME

wandb login 3be59e86854e7deac9e39bf127723eb2e4bf834d

cd $SLURM_SUBMIT_DIR
PYTHONPATH=""
source new_env/bin/activate

ulimit -n 50000

python train_raytune.py
