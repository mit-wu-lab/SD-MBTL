#!/bin/bash
#SBATCH -o logs/walker_multi/walker_log-%j.log
#SBATCH --job-name=walker_m
#SBATCH --array=0-2              # number of trials 0-2999
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 8                     # number of cpu per task
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)

source ~/.bashrc
conda activate test-cmdp
export OMP_NUM_THREADS=16

TRIAL_IDX=$SLURM_ARRAY_TASK_ID
echo $TRIAL_IDX

TRIAL_LIST=(3 4 5)
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

DEFAULT_GRAVITY=10
DEFAULT_SCALE=30
DEFAULT_FRICTION=2.5

# echo "Job: multitask train 3D with trial $TRIAL"

# -----------------------------
# record training start time
# # -----------------------------
echo "Training start time: $(date)"
python train_multitask_load_checkpoint.py --save_path ./results/walker_multi/gravitymulti-scalemulti-frictionmulti-PPO-trial$TRIAL \
	--wk_GRAVITY_Y 10 \
	--wk_SCALE 30 \
	--wk_FRICTION 2.5 \
	--wk_MOTORS_TORQUE 80 \
	--alg PPO \
	--total_steps 5000000000 \
	--env walker \
	--variant 3d
# --resume False

echo "Job: multitask transfer 3D with trial $TRIAL"

NUM_STEPS_LIST=(5000000 10000000 15000000 20000000 25000000 30000000 35000000 40000000 45000000 50000000 55000000 60000000 65000000 70000000)

SOURCE_PATH="./results/walker_multi/gravitymulti-scalemulti-frictionmulti-PPO-trial$TRIAL"

# record training end time
echo "Training end time: $(date)"

# -----------------------------
# 2. Define the TARGET ranges for each context dimension
# -----------------------------
TARGET_SCALES=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2)

# -----------------------------
# 3. Loop over each possible TRIAL in the source
#    Then run transfers for all TARGET contexts
# -----------------------------

echo "Using SOURCE_PATH: $SOURCE_PATH"

# record transfer start time
echo "Transfer start time: $(date)"
# Nested loops to run transfers for all target combinations
for gravity_scale in "${TARGET_SCALES[@]}"; do
	for scale_scale in "${TARGET_SCALES[@]}"; do
		for friction_scale in "${TARGET_SCALES[@]}"; do

			GRAVITY_TGT=$(python -c "print(round($DEFAULT_GRAVITY * $gravity_scale, 2))")
			SCALE_TGT=$(python -c "print(round($DEFAULT_SCALE * $scale_scale, 2))")
			FRICTION_TGT=$(python -c "print(round($DEFAULT_FRICTION * $friction_scale, 2))")

			echo "------------------------------------"
			echo "Transferring to: GRAVITY=$GRAVITY_TGT, SCALE=$SCALE_TGT, FRICTION=$FRICTION_TGT"
			echo "------------------------------------"

			# Define the result path for the TRANSFER run
			RESULT_PATH="./results/walker_multi/gravitymulti-scalemulti-frictionmulti-PPO-trial$TRIAL/transfer_gravity${GRAVITY_TGT}-scale${SCALE_TGT}-friction${FRICTION_TGT}-PPO-trial${TRIAL}"

			# Make the directory if needed
			mkdir -p "$RESULT_PATH"

			# Check if we've already run this exact transfer
			if [ ! -f "$RESULT_PATH/test_reward.csv" ]; then
				echo "No existing test_reward.csv, running transfer simulation..."
				python -u transfer_multitask.py \
					--save_path "$RESULT_PATH" \
					--source_path "$SOURCE_PATH" \
					--wk_GRAVITY_Y $GRAVITY_TGT \
					--wk_SCALE $SCALE_TGT \
					--wk_FRICTION $FRICTION_TGT \
					--wk_MOTORS_TORQUE 80 \
					--alg PPO \
					--env walker \
					--total_steps 150000 \
					--test_eps 50
			else
				echo "Already exists: $RESULT_PATH/test_reward.csv; skipping."
			fi
		done
	done
done

echo "Transfer end time: $(date)"