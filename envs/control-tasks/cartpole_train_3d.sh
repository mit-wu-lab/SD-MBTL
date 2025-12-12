#!/bin/bash
#SBATCH -o logs/cartpole_source/cartpole_source_log-%j.log
#SBATCH --job-name=cartpole_source
#SBATCH --array=0-999             # number of trials 1000 * repeat_times
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)

source ~/.bashrc
conda activate carl
export OMP_NUM_THREADS=16

SCALE_IDX=$SLURM_ARRAY_TASK_ID
echo $SCALE_IDX $TRIAL_IDX

# Define the scaling lists
SCALE_LIST=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2)
DEFAULT_LENPOLE=0.5
DEFAULT_MASSCART=1.0
DEFAULT_MASSPOLE=0.1

# Compute the indices for the combinations of variables
LENPOLE_IDX=$((SCALE_IDX % 10))          # Modulo for lenpole (0-9)
MASSCART_IDX=$(((SCALE_IDX / 10) % 10))   # Dividing by 10 for masscart (0-9)
MASSPOLE_IDX=$(((SCALE_IDX / 100) % 10))  # Dividing by 100 for masspole (0-9)

# Get the scaled values for each variable
LENPOLE_SCALE=${SCALE_LIST[LENPOLE_IDX]}
MASSCART_SCALE=${SCALE_LIST[MASSCART_IDX]}
MASSPOLE_SCALE=${SCALE_LIST[MASSPOLE_IDX]}

# Scale the default values
LENPOLE=$(python -c "print(round($DEFAULT_LENPOLE * $LENPOLE_SCALE, 2))")
MASSCART=$(python -c "print(round($DEFAULT_MASSCART * $MASSCART_SCALE, 2))")
MASSPOLE=$(python -c "print(round($DEFAULT_MASSPOLE * $MASSPOLE_SCALE, 2))")

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Using variables: lenpole=$LENPOLE, masscart=$MASSCART, masspole=$MASSPOLE"

# Trial list (unchanged)
TRIAL_LIST=(0 1 2)

TRIAL_IDX=2
TRIAL=${TRIAL_LIST[TRIAL_IDX]}
python train.py --save_path ./results/cartpole/masscart$MASSCART-lenpole$LENPOLE-masspole$MASSPOLE-force10.00-update0.02-PPO-trial$TRIAL \
	--masscart $MASSCART \
	--pole_length $LENPOLE \
	--masspole $MASSPOLE \
	--force_magnifier 10 \
	--update_interval 0.02 \
	--gravity 9.8 \
	--alg PPO \
	--total_steps 5000000 \
	--env cartpole