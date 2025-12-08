#!/bin/bash
#SBATCH -o logs/log-%j.log
#SBATCH --job-name=main
#SBATCH --array=0-9                 # Total combinations: 2*2*5 = 20 -> indices 0-19
#SBATCH --time=120:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4

set -euo pipefail
mkdir -p logs

source ~/.bashrc
conda activate hybridmbtl
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# -------- Parameter lists (bash arrays are 0-indexed) ---------
dist_left_list=("3-3-3-3-3-3-3" "11111-3-3")
K_list=(50)
trial=(0 1 2 3 4)

# -------- Compute triple indices from SLURM_ARRAY_TASK_ID --------
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}

N_DIST=${#dist_left_list[@]}
N_K=${#K_list[@]}
N_TRIAL=${#trial[@]}
TOTAL=$(( N_DIST * N_K * N_TRIAL ))

if (( TASK_ID < 0 || TASK_ID >= TOTAL )); then
  echo "TASK_ID=${TASK_ID} is out of range [0, ${TOTAL}-1]" >&2
  exit 1
fi

# Map: TASK_ID -> (dist_left_idx, k_idx, trial_idx)
dist_left_idx=$(( TASK_ID / (N_K * N_TRIAL) ))
rem=$(( TASK_ID % (N_K * N_TRIAL) ))
k_idx=$(( rem / N_TRIAL ))
trial_idx=$(( rem % N_TRIAL ))

# Retrieve actual values
DIST_LEFT="${dist_left_list[$dist_left_idx]}"
K="${K_list[$k_idx]}"
TRIAL="${trial[$trial_idx]}"

echo "[$(date)] TASK_ID=${TASK_ID}  ->  dist_left_idx=${dist_left_idx} (${DIST_LEFT}), k_idx=${k_idx} (K=${K}), trial_idx=${trial_idx} (trial=${TRIAL})"
echo "TOTAL combinations = ${TOTAL}"

ENV_NAME="synt_g_7d"

# Prefer running with srun (optional)
python main.py \
  --env "$ENV_NAME" \
  --K "$K" \
  --noise "5" \
  --xweight "None" \
  --yweight "None" \
  --weightleft "$DIST_LEFT" \
  --weightright "3333333" \
  --trial "$TRIAL"

echo "Ended at $(date)"
