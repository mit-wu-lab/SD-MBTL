#!/bin/bash
#SBATCH -o logs/log-%j.log
#SBATCH --job-name=main
#SBATCH --array=0-35             # number of trials 0-3
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 8                     # number of cpu per task

source ~/.bashrc
conda activate sdmbtl
export OMP_NUM_THREADS=16

env_list=(cartpole walker intersectionZoo crop)
K_list=(15 20 30 40 50 80 100 200)

LEN_K_LIST=${#K_list[@]}

env_idx=$((SLURM_ARRAY_TASK_ID / LEN_K_LIST))
K_idx=$((SLURM_ARRAY_TASK_ID % LEN_K_LIST))

ENV_NAME=${env_list[$env_idx]}
K=${K_list[$K_idx]}

echo "Running $ENV_NAME with K=$K"
echo "Started at $(date)"
python main.py --env $ENV_NAME --K $K
echo "Ended at $(date)"