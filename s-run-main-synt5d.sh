#!/bin/bash
#SBATCH -o logs/log-%j.log
#SBATCH --job-name=main
#SBATCH --array=0-23                 # 24 total combinations; alternatively compute TOTAL and use 0-$((TOTAL-1))
#SBATCH --time=120:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs

source ~/.bashrc
conda activate hybridmbtl
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

env_list=(synt_g_5d)
noise_list=(5)
x_weight_list=("None" "Linear")
y_weight_list=("None" "Linear")
dist_left_list=("3-3-3-3-3" "111-3-3" "1111-3")
K_list=(50 100)

LEN_NOISE_LIST=${#noise_list[@]}
LEN_X_WEIGHT_LIST=${#x_weight_list[@]}
LEN_Y_WEIGHT_LIST=${#y_weight_list[@]}
LEN_DIST_LEFT_LIST=${#dist_left_list[@]}
LEN_K_LIST=${#K_list[@]}

# Optional: compute TOTAL automatically for validation
TOTAL=$(( LEN_NOISE_LIST * LEN_X_WEIGHT_LIST * LEN_Y_WEIGHT_LIST * LEN_DIST_LEFT_LIST * LEN_K_LIST ))

env_idx=0
noise_idx=$(( SLURM_ARRAY_TASK_ID / (LEN_X_WEIGHT_LIST * LEN_Y_WEIGHT_LIST * LEN_DIST_LEFT_LIST * LEN_K_LIST) % LEN_NOISE_LIST ))
x_weight_idx=$(( SLURM_ARRAY_TASK_ID / (LEN_Y_WEIGHT_LIST * LEN_DIST_LEFT_LIST * LEN_K_LIST) % LEN_X_WEIGHT_LIST ))
y_weight_idx=$(( SLURM_ARRAY_TASK_ID / (LEN_DIST_LEFT_LIST * LEN_K_LIST) % LEN_Y_WEIGHT_LIST ))
dist_left_idx=$(( SLURM_ARRAY_TASK_ID / LEN_K_LIST % LEN_DIST_LEFT_LIST ))
K_idx=$(( SLURM_ARRAY_TASK_ID % LEN_K_LIST ))

ENV_NAME="${env_list[$env_idx]}"
K="${K_list[$K_idx]}"

echo "TOTAL combos: $TOTAL"
echo "Running with SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "Env=$ENV_NAME K=$K noise=${noise_list[$noise_idx]} x_weight=${x_weight_list[$x_weight_idx]} y_weight=${y_weight_list[$y_weight_idx]} dist_left=${dist_left_list[$dist_left_idx]}"
echo "Started at $(date)"

python main.py \
  --env "$ENV_NAME" \
  --K "$K" \
  --noise "${noise_list[$noise_idx]}" \
  --xweight "${x_weight_list[$x_weight_idx]}" \
  --yweight "${y_weight_list[$y_weight_idx]}" \
  --weightleft "${dist_left_list[$dist_left_idx]}" \
  --weightright 33333

echo "Ended at $(date)"
