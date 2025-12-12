#!/bin/bash 
#SBATCH -o logs/cartpole_multi/cartpole_multi_log-%j.log
#SBATCH --job-name=cartpole_multi
#SBATCH --array=0-2             # number of trials 0-299
#SBATCH --time=120:00:00          # total run time limit (HH:MM:SS)
#SBATCH -N 1                     # number of nodes
#SBATCH -n 1                     # number of tasks
#SBATCH -c 8                     # number of cpu per task

source ~/.bashrc
conda activate carl
export OMP_NUM_THREADS=16

TRIAL_IDX=$SLURM_ARRAY_TASK_ID
echo $TRIAL_IDX

TRIAL_LIST=(0 1 2)
TRIAL=${TRIAL_LIST[TRIAL_IDX]}


echo "Job: multitask train 3D with trial $TRIAL"

python train_multitask_load_checkpoint.py --save_path ./results/cartpole_multi-task/masscartmulti-lenpolemulti-masspolemulti-force10.00-update0.02-PPO-trial$TRIAL \
	--masscart 1 \
	--pole_length 0.5 \
	--masspole 0.1 \
	--force_magnifier 10 \
	--update_interval 0.02 \
	--gravity 9.8 \
	--alg PPO \
	--total_steps 5000000000 \
	--env cartpole \
	--variant 3d \
	--resume True