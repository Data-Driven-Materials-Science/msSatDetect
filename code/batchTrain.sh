#!/bin/bash
#SBATCH --job-name="batchTest"
#SBATCH --output="batchTest%j.%N.out"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --mem=95G
#SBATCH --account=wpi102
#SBATCH -t 01:00:00

source activate solar_env
python train_powder.py
