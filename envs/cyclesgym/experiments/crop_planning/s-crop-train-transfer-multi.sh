#!/bin/bash
#SBATCH -o logs/crop_transfer_multi_log-%j.log
#SBATCH --job-name=crop_transfer_multi
#SBATCH --array=0-2             # number of trials 0-214
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task
#SBATCH -p sched_any
# SBATCH -p newnodes
# SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)

source ~/.bashrc
conda activate cyclesgym
export OMP_NUM_THREADS=16

TRIAL_IDX=$SLURM_ARRAY_TASK_ID

# Trial list (unchanged)
TRIAL_LIST=(0 1 2)

TRIAL=${TRIAL_LIST[TRIAL_IDX]}

# -----------------------------
# record training start time
# -----------------------------
SOURCE_PATH="../../results/tempmulti_sunmulti_precmulti_trial${TRIAL}"

# if source path directory does not exist, create it
if [ ! -d "$SOURCE_PATH" ]; then
	mkdir -p $SOURCE_PATH
fi
echo "Source path: $SOURCE_PATH"

echo "Training start time: $(date)"

if [ ! -f "$SOURCE_PATH/wandb/latest-run/files/model.zip" ]; then
	echo "No existing model.zip, running training simulation..."
	python train_cmdp.py \
		--fixed_weather True \
		--non_adaptive True \
		--seed $TRIAL \
		--multi True
else
	echo "Already exists: $SOURCE_PATH/model.zip; skipping."
fi

# record training end time
echo "Training end time: $(date)"

# -----------------------------
# 2. Define the TARGET ranges for each context dimension
# -----------------------------
TARGET_SCALES=(0.5 1.0 1.5 2.0 2.5 3.0)
TARGET_TEMP_SCALE_LIST=(0.75 1.0 1.25 1.5 1.75 2.0)

# -----------------------------
# 3. Loop over each possible TRIAL in the source
#    Then run transfers for all TARGET contexts
# -----------------------------

echo "Using SOURCE_PATH: $SOURCE_PATH"

# record transfer start time
echo "Transfer start time: $(date)"
# Nested loops to run transfers for all target combinations
for temp_scale_t in "${TARGET_TEMP_SCALE_LIST[@]}"; do
	for solar_scale_t in "${TARGET_SCALES[@]}"; do
		for rain_scale_t in "${TARGET_SCALES[@]}"; do

			echo "------------------------------------"
			echo "Transferring to: temp_scale_t=$temp_scale_t, solar_scale_t=$solar_scale_t, rain_scale_t=$rain_scale_t"
			echo "------------------------------------"

			# Define the result path for the TRANSFER run
			RESULT_PATH="$SOURCE_PATH/transfer_temp${temp_scale_t}_sun${solar_scale_t}_prec${rain_scale_t}_trial${TRIAL}"

			# Make the directory if needed
			mkdir -p "$RESULT_PATH"

			# Check if we've already run this exact transfer
			if [ ! -f "$RESULT_PATH/fertilizer_table.csv" ]; then
				echo "No existing fertilizer_table.csv, running transfer simulation..."
				python -u transfer_cmdp.py \
					--temperature_t $temp_scale_t \
					--sunlight_t $solar_scale_t \
					--precipitation_t $rain_scale_t \
					--fixed_weather True \
					--non_adaptive True \
					--seed $TRIAL \
					--multi True
			else
				echo "Already exists: $RESULT_PATH/fertilizer_table.csv; skipping."
			fi
		done
	done
done

echo "Transfer end time: $(date)"
