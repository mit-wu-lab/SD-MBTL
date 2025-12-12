#!/bin/bash
#SBATCH -o logs/walker/walker_log-%j.log
#SBATCH --job-name=walker
#SBATCH --array=0-512             # number of trials 0-512
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)

source ~/.bashrc
conda activate test-cmdp
export OMP_NUM_THREADS=16

# SCALE_IDX=$SLURM_ARRAY_TASK_ID

# Define the scaling lists
SCALE_LIST=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6)
DEFAULT_GRAVITY=10
DEFAULT_SCALE=30
DEFAULT_FRICTION=2.5

# Compute the indices for the combinations of variables
GRAVITY_IDX=$((SLURM_ARRAY_TASK_ID % 8))          # Modulo for lenpole (0-9)
SCALE_IDX=$(((SLURM_ARRAY_TASK_ID / 8) % 8))   # Dividing by 10 for masscart (0-9)
FRICTION_IDX=$(((SLURM_ARRAY_TASK_ID / 64) % 8))  # Dividing by 100 for masspole (0-9)

# Get the scaled values for each variable
GRAVITY_SCALE=${SCALE_LIST[GRAVITY_IDX]}
SCALE_SCALE=${SCALE_LIST[SCALE_IDX]}
FRICTION_SCALE=${SCALE_LIST[FRICTION_IDX]}

# Scale the default values
GRAVITY=$(python -c "print(round($DEFAULT_GRAVITY * $GRAVITY_SCALE, 2))")
SCALE=$(python -c "print(round($DEFAULT_SCALE * $SCALE_SCALE, 2))")
FRICTION=$(python -c "print(round($DEFAULT_FRICTION * $FRICTION_SCALE, 2))")

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Using variables: GRAVITY=$GRAVITY, SCALE=$SCALE, FRICTION=$FRICTION"

# Trial list (unchanged)
TRIAL_LIST=(0 1 2)

TRIAL_IDX=0
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

# -----------------------------
# record training start time
# -----------------------------
SOURCE_PATH="./results/walker/gravity${GRAVITY}-scale${SCALE}-friction${FRICTION}-PPO-trial${TRIAL}"

echo "Training start time: $(date)"

if [ ! -f "$SOURCE_PATH/runs/source_task_training_checkpoints/model_5000000_steps.zip" ]; then
	echo "No existing model_5000000_steps.zip, running training simulation..."
	python train.py --save_path ./results/walker/gravity$GRAVITY-scale$SCALE-friction$FRICTION-PPO-trial$TRIAL \
		--wk_GRAVITY_Y $GRAVITY \
		--wk_SCALE $SCALE \
		--wk_FRICTION $FRICTION \
		--wk_MOTORS_TORQUE 80 \
		--alg PPO \
		--total_steps 5000000 \
		--env walker
else
	echo "Already exists: $SOURCE_PATH/model_5000000_steps.zip; skipping."
fi

# record training end time
echo "Training end time: $(date)"


# -----------------------------
# 2. Define the TARGET ranges for each context dimension
# -----------------------------
TARGET_SCALES=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6)

# -----------------------------
# 3. Loop over each possible TRIAL in the source
#    Then run transfers for all TARGET contexts
# -----------------------------

# Define the path to the policy you trained (SOURCE_PATH)
SOURCE_PATH="./results/walker/gravity${GRAVITY}-scale${SCALE}-friction${FRICTION}-PPO-trial${TRIAL}"

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
			RESULT_PATH="./results/walker/gravity${GRAVITY}-scale${SCALE}-friction${FRICTION}-PPO-trial${TRIAL}/transfer_gravity${GRAVITY_TGT}-scale${SCALE_TGT}-friction${FRICTION_TGT}-PPO-trial${TRIAL}"

			# Make the directory if needed
			mkdir -p "$RESULT_PATH"

			# Check if we've already run this exact transfer
			if [ ! -f "$RESULT_PATH/test_reward.csv" ]; then
				echo "No existing test_reward.csv, running transfer simulation..."
				python -u transfer.py \
					--save_path "$RESULT_PATH" \
					--source_path "$SOURCE_PATH" \
					--wk_GRAVITY_Y $GRAVITY_TGT \
					--wk_SCALE $SCALE_TGT \
					--wk_FRICTION $FRICTION_TGT \
					--wk_MOTORS_TORQUE 80 \
					--alg PPO \
					--env walker \
					--total_steps 100000 \
					--test_eps 30
			else
				echo "Already exists: $RESULT_PATH/test_reward.csv; skipping."
			fi
		done
	done
done

echo "Transfer end time: $(date)"

TRIAL_IDX=1
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

# -----------------------------
# record training start time
# -----------------------------
SOURCE_PATH="./results/walker/gravity${GRAVITY}-scale${SCALE}-friction${FRICTION}-PPO-trial${TRIAL}"

echo "Training start time: $(date)"

if [ ! -f "$SOURCE_PATH/runs/source_task_training_checkpoints/model_5000000_steps.zip" ]; then
	echo "No existing model_5000000_steps.zip, running training simulation..."
	python train.py --save_path ./results/walker/gravity$GRAVITY-scale$SCALE-friction$FRICTION-PPO-trial$TRIAL \
		--wk_GRAVITY_Y $GRAVITY \
		--wk_SCALE $SCALE \
		--wk_FRICTION $FRICTION \
		--wk_MOTORS_TORQUE 80 \
		--alg PPO \
		--total_steps 5000000 \
		--env walker
else
	echo "Already exists: $SOURCE_PATH/model_5000000_steps.zip; skipping."
fi

# record training end time
echo "Training end time: $(date)"


# -----------------------------
# 2. Define the TARGET ranges for each context dimension
# -----------------------------
TARGET_SCALES=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6)

# -----------------------------
# 3. Loop over each possible TRIAL in the source
#    Then run transfers for all TARGET contexts
# -----------------------------

# Define the path to the policy you trained (SOURCE_PATH)
SOURCE_PATH="./results/walker/gravity${GRAVITY}-scale${SCALE}-friction${FRICTION}-PPO-trial${TRIAL}"

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
			RESULT_PATH="./results/walker/gravity${GRAVITY}-scale${SCALE}-friction${FRICTION}-PPO-trial${TRIAL}/transfer_gravity${GRAVITY_TGT}-scale${SCALE_TGT}-friction${FRICTION_TGT}-PPO-trial${TRIAL}"

			# Make the directory if needed
			mkdir -p "$RESULT_PATH"

			# Check if we've already run this exact transfer
			if [ ! -f "$RESULT_PATH/test_reward.csv" ]; then
				echo "No existing test_reward.csv, running transfer simulation..."
				python -u transfer.py \
					--save_path "$RESULT_PATH" \
					--source_path "$SOURCE_PATH" \
					--wk_GRAVITY_Y $GRAVITY_TGT \
					--wk_SCALE $SCALE_TGT \
					--wk_FRICTION $FRICTION_TGT \
					--wk_MOTORS_TORQUE 80 \
					--alg PPO \
					--env walker \
					--total_steps 100000 \
					--test_eps 30
			else
				echo "Already exists: $RESULT_PATH/test_reward.csv; skipping."
			fi
		done
	done
done

echo "Transfer end time: $(date)"
