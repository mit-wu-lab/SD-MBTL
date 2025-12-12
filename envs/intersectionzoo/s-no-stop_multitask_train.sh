#!/bin/bash 
#SBATCH -o logs/no-stop_multi/log-%j.log
#SBATCH --job-name=no-stop_multi
#SBATCH --array=0-2             # number of trials 0-49
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 8                     # number of cpu per task

# source ~/.bashrc
# Loading the required module
# source /etc/profile
# module load anaconda/2022a

# conda activate no-stop
export WANDB_MODE="offline"
export OMP_NUM_THREADS=16

TRIAL_IDX=$SLURM_ARRAY_TASK_ID
echo $TRIAL_IDX

TRIAL_LIST=(0 1 2)

TRIAL=${TRIAL_LIST[TRIAL_IDX]}


echo "TRIAL: $TRIAL"

echo "Job: train multi trial $TRIAL"

python -u code/main.py --dir ./results/no-stop/inflowmulti-penratemulti-greenmulti-PPO-trial$TRIAL \
	--kwargs "{'run_mode':'train','n_steps':1000000}" \
	--inflow 250 \
	--penrate 0.5 \
	--green 20 \
	--inflow_multi True \
	--penrate_multi True \
	--green_multi True
