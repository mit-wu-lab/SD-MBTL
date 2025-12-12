#!/bin/bash
#SBATCH -o logs/cartpole_transfer/output-%j.log
#SBATCH --job-name=cartpole_transfer
#SBATCH --array=0-999             # number of source tasks = 10 * 10 * 10 = 1000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --time=120:00:00           # total run time limit (HH:MM:SS)

source ~/.bashrc
conda activate carl
export OMP_NUM_THREADS=1

# -----------------------------
# 1. Parse the array job index to determine the SOURCE task
# -----------------------------
SCALE_IDX=$SLURM_ARRAY_TASK_ID

# Definitions from your training script:
SCALE_LIST=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2)
DEFAULT_LENPOLE=0.5
DEFAULT_MASSCART=1.0
DEFAULT_MASSPOLE=0.1

# Map array index to lenpole/masscart/masspole indices:
LENPOLE_IDX=$(( SCALE_IDX % 10 ))             # 0-9
MASSCART_IDX=$(( (SCALE_IDX / 10) % 10 ))     # 0-9
MASSPOLE_IDX=$(( (SCALE_IDX / 100) % 10 ))    # 0-9

# Get the scaling factors for the source
LENPOLE_SCALE=${SCALE_LIST[LENPOLE_IDX]}
MASSCART_SCALE=${SCALE_LIST[MASSCART_IDX]}
MASSPOLE_SCALE=${SCALE_LIST[MASSPOLE_IDX]}

# Compute the actual numeric values for the source
LENPOLE=$(python -c "print(round($DEFAULT_LENPOLE * $LENPOLE_SCALE, 2))")
MASSCART=$(python -c "print(round($DEFAULT_MASSCART * $MASSCART_SCALE, 2))")
MASSPOLE=$(python -c "print(round($DEFAULT_MASSPOLE * $MASSPOLE_SCALE, 2))")

echo "------------------------------------------------------"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Source contexts: lenpole=$LENPOLE, masscart=$MASSCART, masspole=$MASSPOLE"
echo "------------------------------------------------------"

# -----------------------------
# 2. Define the TARGET ranges for each context dimension
# -----------------------------
TARGET_LENPOLE_SCALES=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2)
TARGET_MASSCART_SCALES=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2)
TARGET_MASSPOLE_SCALES=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2)

# -----------------------------
# 3. Loop over each possible TRIAL in the source
#    Then run transfers for all TARGET contexts
# -----------------------------
TRIAL_LIST=(0 1 2)
for TRIAL_IDX in "${!TRIAL_LIST[@]}"; do
    
    TRIAL=${TRIAL_LIST[TRIAL_IDX]}
    echo "========== Source Trial: $TRIAL =========="

    # Define the path to the policy you trained (SOURCE_PATH)
    SOURCE_PATH="./results/cartpole/masscart${MASSCART}-lenpole${LENPOLE}-masspole${MASSPOLE}-force10.00-update0.02-PPO-trial${TRIAL}"

    echo "Using SOURCE_PATH: $SOURCE_PATH"

    # Nested loops to run transfers for all target combinations
    for ln_scale in "${TARGET_LENPOLE_SCALES[@]}"; do
        for mc_scale in "${TARGET_MASSCART_SCALES[@]}"; do
            for mp_scale in "${TARGET_MASSPOLE_SCALES[@]}"; do

                LN_TGT=$(python -c "print(round($DEFAULT_LENPOLE * $ln_scale, 2))")
                MC_TGT=$(python -c "print(round($DEFAULT_MASSCART * $mc_scale, 2))")
                MP_TGT=$(python -c "print(round($DEFAULT_MASSPOLE * $mp_scale, 2))")

                echo "------------------------------------"
                echo "Transferring to: lenpole=$LN_TGT, masscart=$MC_TGT, masspole=$MP_TGT"
                echo "------------------------------------"

                # Define the result path for the TRANSFER run
                RESULT_PATH="./results/cartpole/masscart${MASSCART}-lenpole${LENPOLE}-masspole${MASSPOLE}-force10.00-update0.02-PPO-trial${TRIAL}/transfer_masscart${MC_TGT}-lenpole${LN_TGT}-masspole${MP_TGT}-force10.00-update0.02-PPO-trial${TRIAL}"

                # Make the directory if needed
                mkdir -p "$RESULT_PATH"

                # Check if we've already run this exact transfer
                if [ ! -f "$RESULT_PATH/test_reward.csv" ]; then
                    echo "No existing test_reward.csv, running transfer simulation..."
                    python -u transfer.py \
                        --save_path "$RESULT_PATH" \
                        --source_path "$SOURCE_PATH" \
                        --masscart "$MC_TGT" \
                        --pole_length "$LN_TGT" \
                        --masspole "$MP_TGT" \
                        --force_magnifier 10 \
                        --update_interval 0.02 \
                        --gravity 9.8 \
                        --alg PPO \
                        --env cartpole \
                        --total_steps 150000 \
                        --test_eps 100
                else
                    echo "Already exists: $RESULT_PATH/test_reward.csv; skipping."
                fi
            done
        done
    done
done
