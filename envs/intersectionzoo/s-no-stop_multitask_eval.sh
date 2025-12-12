#!/bin/bash 
#SBATCH -o logs/no-stop_multi/multi_transfer_log-%j.log
#SBATCH --job-name=no-stop_multi
#SBATCH --array=0-11             # number of trials 0-1536, 2187, 3000
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 4                     # number of cpu per task

# conda activate no-stop
export WANDB_MODE="offline"
export OMP_NUM_THREADS=16

SCALE_IDX=$((${SLURM_ARRAY_TASK_ID}/3))
TRIAL_IDX=$((${SLURM_ARRAY_TASK_ID}%3))
echo $SCALE_IDX $TRIAL_IDX

CKPT_LIST=(10000 15000 20000 25000)
TRIAL_LIST=(0 1 2)

CKPT=${CKPT_LIST[SCALE_IDX]}
TRIAL=${TRIAL_LIST[TRIAL_IDX]}

echo "CKPT: $CKPT"
echo "TRIAL: $TRIAL"

echo "Job: eval multi trial $TRIAL"

SCALE_LIST=(0.2 0.4 0.6 0.8 1 1.2 1.4)

SCALE_LIST_LEN=${#SCALE_LIST[@]}

INFLOW_DEFAULT=250 # VAR1
GREEN_PHASE_DEFAULT=20 # VAR2
RED_PHASE_DEFAULT=20 # fix
LANE_LENGTH_DEFAULT=400 # fix
SPEED_LIMIT_DEFAULT=15 # fix
PENETRATION_DEFAULT=0.5 # VAR3

for SCALE1_T in "${SCALE_LIST[@]}"; do
	for SCALE2_T in "${SCALE_LIST[@]}"; do
		for SCALE3_T in "${SCALE_LIST[@]}"; do
			VAR1_T=$(echo "scale=2; $SCALE1_T * $INFLOW_DEFAULT" | bc)
			VAR2_T=$(echo "scale=2; $SCALE2_T * $GREEN_PHASE_DEFAULT" | bc)
			VAR3_T=$(echo "scale=2; $SCALE3_T * $PENETRATION_DEFAULT" | bc)
			echo "Target VAR1: $VAR1_T"
			echo "Target VAR2: $VAR2_T"
			echo "Target VAR3: $VAR3_T"
			RESULT_PATH="./results/no-stop/inflowmulti-penratemulti-greenmulti-PPO-trial$TRIAL/transfer_inflow$VAR1_T-penrate$VAR3_T-green$VAR2_T-PPO-trial$TRIAL"
			mkdir $RESULT_PATH
			if [ ! -f "$RESULT_PATH/eval_result_$CKPT.csv" ]; then
				echo "File $RESULT_PATH/eval_result_$CKPT.csv does not exist, running simulation."
				python -u code/main.py \
					--dir $RESULT_PATH \
					--source_path ./results/no-stop/inflowmulti-penratemulti-greenmulti-PPO-trial$TRIAL \
					--kwargs "{'run_mode':'single_eval', 'n_steps':10}" \
					--inflow $VAR1_T \
					--penrate $VAR2_T \
					--green $VAR3_T \
					--ckpt $CKPT
			else
				echo "File $RESULT_PATH exists, skipping."
			fi
		done
	done
done
