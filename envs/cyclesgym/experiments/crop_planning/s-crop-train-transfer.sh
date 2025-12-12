#!/bin/bash
#SBATCH -o logs/crop_transfer_log-%j.log
#SBATCH --job-name=crop_transfer
#SBATCH --array=0-215             # number of trials 0-215, 146-215
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task
#SBATCH -p newnodes
# SBATCH -p mit_quicktest # mit_normal sched_any newnodes
# SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)

source ~/.bashrc
conda activate cyclesgym
export OMP_NUM_THREADS=16

SCALE_IDX=$SLURM_ARRAY_TASK_ID

# Define the scaling lists
# SCALE_LIST=(0.2 0.4 0.6 0.8 1 1.2 1.4 1.6 1.8 2)
SCALE_LIST=(0.5 1.0 1.5 2.0 2.5 3.0)
TEMP_SCALE_LIST=(0.75 1.0 1.25 1.5 1.75 2.0)
DEFAULT_RAIN=10.0
DEFAULT_TEMP=20.0
DEFAULT_SOLAR=15.0

# Compute the indices for the combinations of variables
RAIN_IDX=$((SCALE_IDX % 6))          # Modulo for lenpole (0-9)
TEMP_IDX=$(((SCALE_IDX / 6) % 6))   # Dividing by 10 for masscart (0-9)
SOLAR_IDX=$(((SCALE_IDX / 36) % 6))  # Dividing by 100 for masspole (0-9)

# Get the scaled values for each variable
RAIN_SCALE=${SCALE_LIST[RAIN_IDX]}
TEMP_SCALE=${TEMP_SCALE_LIST[TEMP_IDX]}
SOLAR_SCALE=${SCALE_LIST[SOLAR_IDX]}

# Scale the default values
RAIN=$(python -c "print(round($DEFAULT_RAIN * $RAIN_SCALE, 2))")
TEMPC=$(python -c "print(round($DEFAULT_TEMP * $TEMP_SCALE, 2))")
SOLAR=$(python -c "print(round($DEFAULT_SOLAR * $SOLAR_SCALE, 2))")

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Using variables: RAIN=$RAIN, TEMP=$TEMPC, SOLAR=$SOLAR"

# Trial list (unchanged)
TRIAL_LIST=(0 1 2)

TRIAL_IDX=0
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

# -----------------------------
# record training start time
# -----------------------------
SOURCE_PATH="../../results/temp${TEMP_SCALE}_sun${SOLAR_SCALE}_prec${RAIN_SCALE}_trial${TRIAL}"

# if source path directory does not exist, create it
if [ ! -d "$SOURCE_PATH" ]; then
	mkdir -p $SOURCE_PATH
fi
echo "Source path: $SOURCE_PATH"

echo "Training start time: $(date)"

if [ ! -f "$SOURCE_PATH/wandb/latest-run/files/model.zip" ]; then
	echo "No existing model.zip, running training simulation..."
	python train_cmdp.py \
		--temperature $TEMP_SCALE \
		--sunlight $SOLAR_SCALE \
		--precipitation $RAIN_SCALE \
		--fixed_weather True \
		--non_adaptive True \
		--seed $TRIAL
		# --save_path ./results/crop/RAIN${RAIN_SCALE}-TEMP${TEMP_SCALE}-SOLAR${SOLAR_SCALE}-PPO-trial$TRIAL \
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
					--temperature $TEMP_SCALE \
					--sunlight $SOLAR_SCALE \
					--precipitation $RAIN_SCALE \
					--temperature_t $temp_scale_t \
					--sunlight_t $solar_scale_t \
					--precipitation_t $rain_scale_t \
					--fixed_weather True \
					--non_adaptive True \
					--seed $TRIAL

			else
				echo "Already exists: $RESULT_PATH/fertilizer_table.csv; skipping."
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
SOURCE_PATH="../../results/temp${TEMP_SCALE}_sun${SOLAR_SCALE}_prec${RAIN_SCALE}_trial${TRIAL}"

# if source path directory does not exist, create it
if [ ! -d "$SOURCE_PATH" ]; then
	mkdir -p $SOURCE_PATH
fi
echo "Source path: $SOURCE_PATH"

echo "Training start time: $(date)"

if [ ! -f "$SOURCE_PATH/wandb/latest-run/files/model.zip" ]; then
	echo "No existing model.zip, running training simulation..."
	python train_cmdp.py \
		--temperature $TEMP_SCALE \
		--sunlight $SOLAR_SCALE \
		--precipitation $RAIN_SCALE \
		--fixed_weather True \
		--non_adaptive True \
		--seed $TRIAL
		# --save_path ./results/crop/RAIN${RAIN_SCALE}-TEMP${TEMP_SCALE}-SOLAR${SOLAR_SCALE}-PPO-trial$TRIAL \
else
	echo "Already exists: $SOURCE_PATH/model.zip; skipping."
fi

# record training end time
echo "Training end time: $(date)"



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
					--temperature $TEMP_SCALE \
					--sunlight $SOLAR_SCALE \
					--precipitation $RAIN_SCALE \
					--temperature_t $temp_scale_t \
					--sunlight_t $solar_scale_t \
					--precipitation_t $rain_scale_t \
					--fixed_weather True \
					--non_adaptive True \
					--seed $TRIAL

			else
				echo "Already exists: $RESULT_PATH/fertilizer_table.csv; skipping."
			fi
		done
	done
done

echo "Transfer end time: $(date)"

TRIAL_IDX=2
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

# -----------------------------
# record training start time
# -----------------------------
SOURCE_PATH="../../results/temp${TEMP_SCALE}_sun${SOLAR_SCALE}_prec${RAIN_SCALE}_trial${TRIAL}"

# if source path directory does not exist, create it
if [ ! -d "$SOURCE_PATH" ]; then
	mkdir -p $SOURCE_PATH
fi
echo "Source path: $SOURCE_PATH"

echo "Training start time: $(date)"

if [ ! -f "$SOURCE_PATH/wandb/latest-run/files/model.zip" ]; then
	echo "No existing model.zip, running training simulation..."
	python train_cmdp.py \
		--temperature $TEMP_SCALE \
		--sunlight $SOLAR_SCALE \
		--precipitation $RAIN_SCALE \
		--fixed_weather True \
		--non_adaptive True \
		--seed $TRIAL
		# --save_path ./results/crop/RAIN${RAIN_SCALE}-TEMP${TEMP_SCALE}-SOLAR${SOLAR_SCALE}-PPO-trial$TRIAL \
else
	echo "Already exists: $SOURCE_PATH/model.zip; skipping."
fi

# record training end time
echo "Training end time: $(date)"




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
					--temperature $TEMP_SCALE \
					--sunlight $SOLAR_SCALE \
					--precipitation $RAIN_SCALE \
					--temperature_t $temp_scale_t \
					--sunlight_t $solar_scale_t \
					--precipitation_t $rain_scale_t \
					--fixed_weather True \
					--non_adaptive True \
					--seed $TRIAL

			else
				echo "Already exists: $RESULT_PATH/fertilizer_table.csv; skipping."
			fi
		done
	done
done

echo "Transfer end time: $(date)"