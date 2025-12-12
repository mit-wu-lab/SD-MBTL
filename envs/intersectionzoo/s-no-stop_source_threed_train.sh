#!/bin/bash 
#SBATCH -o logs/no-stop_source/no-stop_source_log-%j.log
#SBATCH --job-name=no-stop_source
#SBATCH --array=0-342             # number of trials 0-342
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task

# source ~/.bashrc
# Loading the required module
# source /etc/profile
# module load anaconda/2022a

# conda activate no-stop
export WANDB_MODE="offline"
export OMP_NUM_THREADS=16

SCALE_LIST=(0.2 0.4 0.6 0.8 1 1.2 1.4)
TRIAL_LIST=(0) # 3,4

SCALE_LIST_LEN=${#SCALE_LIST[@]}
TRIAL_LIST_LEN=${#TRIAL_LIST[@]}

SCALE_IDX=$((${SLURM_ARRAY_TASK_ID}/TRIAL_LIST_LEN))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%TRIAL_LIST_LEN))

SCALE_IDX1=$((${SCALE_IDX}/SCALE_LIST_LEN/SCALE_LIST_LEN))
SCALE_IDX2=$((${SCALE_IDX}/SCALE_LIST_LEN%SCALE_LIST_LEN))
SCALE_IDX3=$((${SCALE_IDX}%SCALE_LIST_LEN))

echo $SCALE_IDX1 $SCALE_IDX2 $SCALE_IDX3 $TRIAL_IDX

SCALE1=${SCALE_LIST[SCALE_IDX1]}
SCALE2=${SCALE_LIST[SCALE_IDX2]}
SCALE3=${SCALE_LIST[SCALE_IDX3]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

echo $SCALE1 $SCALE2 $SCALE3 $TRIAL

INFLOW_DEFAULT=250 # VAR1
GREEN_PHASE_DEFAULT=20 # VAR2
RED_PHASE_DEFAULT=20 # fix
LANE_LENGTH_DEFAULT=400 # fix
SPEED_LIMIT_DEFAULT=15 # fix
PENETRATION_DEFAULT=0.5 # VAR3

VAR1=$(echo "scale=2; $SCALE1 * $INFLOW_DEFAULT" | bc)
VAR2=$(echo "scale=2; $SCALE2 * $GREEN_PHASE_DEFAULT" | bc)
VAR3=$(echo "scale=2; $SCALE3 * $PENETRATION_DEFAULT" | bc)

LANE_LENGTH=$(echo "scale=2; $LANE_LENGTH_DEFAULT" | bc)
SPEED_LIMIT=$(echo "scale=2; $SPEED_LIMIT_DEFAULT" | bc)

echo $VAR1 $VAR2 $VAR3

TRIAL=0

if [ ! -f "./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth" ]; then
	echo "File ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth does not exist, running simulation."
	python -u code/main.py --dir ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL \
		--kwargs "{'run_mode':'train'}" \
		--inflow $VAR1 \
		--green $VAR2 \
		--penrate $VAR3
else
	echo "File ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth exists, skipping."
fi

for SCALE1_T in "${SCALE_LIST[@]}"; do
	for SCALE2_T in "${SCALE_LIST[@]}"; do
		for SCALE3_T in "${SCALE_LIST[@]}"; do
			VAR1_T=$(echo "scale=2; $SCALE1_T * $INFLOW_DEFAULT" | bc)
			VAR2_T=$(echo "scale=2; $SCALE2_T * $GREEN_PHASE_DEFAULT" | bc)
			VAR3_T=$(echo "scale=2; $SCALE3_T * $PENETRATION_DEFAULT" | bc)
			echo "Target VAR1: $VAR1_T"
			echo "Target VAR2: $VAR2_T"
			echo "Target VAR3: $VAR3_T"
			RESULT_PATH="$CURRENT_DIR/results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/transfer_inflow$VAR1_T-penrate$VAR3_T-green$VAR2_T-PPO-trial$TRIAL"
			mkdir $RESULT_PATH
			if [ ! -f "$RESULT_PATH/eval_result.csv" ]; then
				echo "File $RESULT_PATH/eval_result.csv does not exist, running simulation."
				python -u code/main.py \
					--dir $RESULT_PATH \
					--source_path $CURRENT_DIR/results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL \
					--kwargs "{'run_mode':'single_eval', 'n_steps':10}" \
					--inflow $VAR1_T \
					--penrate $VAR2_T \
					--green $VAR3_T
			else
				echo "File $RESULT_PATH exists, skipping."
			fi
		done
	done
done


TRIAL=1

if [ ! -f "./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth" ]; then
	echo "File ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth does not exist, running simulation."
	python -u code/main.py --dir ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL \
		--kwargs "{'run_mode':'train'}" \
		--inflow $VAR1 \
		--green $VAR2 \
		--penrate $VAR3
else
	echo "File ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth exists, skipping."
fi

for SCALE1_T in "${SCALE_LIST[@]}"; do
	for SCALE2_T in "${SCALE_LIST[@]}"; do
		for SCALE3_T in "${SCALE_LIST[@]}"; do
			VAR1_T=$(echo "scale=2; $SCALE1_T * $INFLOW_DEFAULT" | bc)
			VAR2_T=$(echo "scale=2; $SCALE2_T * $GREEN_PHASE_DEFAULT" | bc)
			VAR3_T=$(echo "scale=2; $SCALE3_T * $PENETRATION_DEFAULT" | bc)
			echo "Target VAR1: $VAR1_T"
			echo "Target VAR2: $VAR2_T"
			echo "Target VAR3: $VAR3_T"
			RESULT_PATH="$CURRENT_DIR/results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/transfer_inflow$VAR1_T-penrate$VAR3_T-green$VAR2_T-PPO-trial$TRIAL"
			mkdir $RESULT_PATH
			if [ ! -f "$RESULT_PATH/eval_result.csv" ]; then
				echo "File $RESULT_PATH/eval_result.csv does not exist, running simulation."
				python -u code/main.py \
					--dir $RESULT_PATH \
					--source_path $CURRENT_DIR/results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL \
					--kwargs "{'run_mode':'single_eval', 'n_steps':10}" \
					--inflow $VAR1_T \
					--penrate $VAR2_T \
					--green $VAR3_T
			else
				echo "File $RESULT_PATH exists, skipping."
			fi
		done
	done
done


TRIAL=2

if [ ! -f "./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth" ]; then
	echo "File ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth does not exist, running simulation."
	python -u code/main.py --dir ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL \
		--kwargs "{'run_mode':'train'}" \
		--inflow $VAR1 \
		--green $VAR2 \
		--penrate $VAR3
else
	echo "File ./results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/models/model-5000.pth exists, skipping."
fi

for SCALE1_T in "${SCALE_LIST[@]}"; do
	for SCALE2_T in "${SCALE_LIST[@]}"; do
		for SCALE3_T in "${SCALE_LIST[@]}"; do
			VAR1_T=$(echo "scale=2; $SCALE1_T * $INFLOW_DEFAULT" | bc)
			VAR2_T=$(echo "scale=2; $SCALE2_T * $GREEN_PHASE_DEFAULT" | bc)
			VAR3_T=$(echo "scale=2; $SCALE3_T * $PENETRATION_DEFAULT" | bc)
			echo "Target VAR1: $VAR1_T"
			echo "Target VAR2: $VAR2_T"
			echo "Target VAR3: $VAR3_T"
			RESULT_PATH="$CURRENT_DIR/results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL/transfer_inflow$VAR1_T-penrate$VAR3_T-green$VAR2_T-PPO-trial$TRIAL"
			mkdir $RESULT_PATH
			if [ ! -f "$RESULT_PATH/eval_result.csv" ]; then
				echo "File $RESULT_PATH/eval_result.csv does not exist, running simulation."
				python -u code/main.py \
					--dir $RESULT_PATH \
					--source_path $CURRENT_DIR/results/no-stop/inflow$VAR1-penrate$VAR3-green$VAR2-PPO-trial$TRIAL \
					--kwargs "{'run_mode':'single_eval', 'n_steps':10}" \
					--inflow $VAR1_T \
					--penrate $VAR2_T \
					--green $VAR3_T
			else
				echo "File $RESULT_PATH exists, skipping."
			fi
		done
	done
done
