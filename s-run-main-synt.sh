#!/bin/bash
#SBATCH -o logs/log-%j.log
#SBATCH --job-name=main
#SBATCH --array=0-23             # number of trials 0-3
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task

source ~/.bashrc
conda activate sdmbtl
export OMP_NUM_THREADS=16

env_list=(synt_g)
noise_list=(0 5 15 30)
x_weight_list=("None" "Linear")
y_weight_list=("None" "Linear")
dist_left_list=("11-3" "3-3-3")
K_list=(15 30 50)

LEN_NOISE_LIST=${#noise_list[@]}
LEN_X_WEIGHT_LIST=${#x_weight_list[@]}
LEN_Y_WEIGHT_LIST=${#y_weight_list[@]}
LEN_DIST_LEFT_LIST=${#dist_left_list[@]}
LEN_K_LIST=${#K_list[@]}

env_idx=0
noise_idx=$((SLURM_ARRAY_TASK_ID / (LEN_X_WEIGHT_LIST * LEN_Y_WEIGHT_LIST * LEN_DIST_LEFT_LIST * LEN_K_LIST) % LEN_NOISE_LIST))
x_weight_idx=$((SLURM_ARRAY_TASK_ID / (LEN_Y_WEIGHT_LIST * LEN_DIST_LEFT_LIST * LEN_K_LIST) % LEN_X_WEIGHT_LIST))
y_weight_idx=$((SLURM_ARRAY_TASK_ID / (LEN_DIST_LEFT_LIST * LEN_K_LIST) % LEN_Y_WEIGHT_LIST))
dist_left_idx=$((SLURM_ARRAY_TASK_ID / (LEN_K_LIST) % LEN_DIST_LEFT_LIST))
K_idx=$((SLURM_ARRAY_TASK_ID % LEN_K_LIST))

ENV_NAME=${env_list[$env_idx]}
K=${K_list[$K_idx]}

echo "Running with SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Running $ENV_NAME with K=$K, noise=${noise_list[$noise_idx]},  x_weight=${x_weight_list[$x_weight_idx]}, y_weight=${y_weight_list[$y_weight_idx]}, dist_left=${dist_left_list[$dist_left_idx]}"
echo "Started at $(date)"
python main.py --env $ENV_NAME --K $K --noise ${noise_list[$noise_idx]} --xweight ${x_weight_list[$x_weight_idx]} --yweight ${y_weight_list[$y_weight_idx]} --weightleft ${dist_left_list[$dist_left_idx]} --weightright 333
echo "Ended at $(date)"
echo "Finished $ENV_NAME with K=$K"