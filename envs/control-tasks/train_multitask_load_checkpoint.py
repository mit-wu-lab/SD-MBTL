import os
import argparse
import json
from carl.envs.gymnasium.classic_control import CARLCartPole as CARLCartPoleEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC, DDPG, TD3, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure
import numpy as np
import importlib
import warnings
import re

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', dest='env_name', type=str, help='environment type')

# cartpole
parser.add_argument('--pole_length', dest='pole_length', default=0.5, type=float)
parser.add_argument('--masscart', dest='masscart', default=1.0, type=float)
parser.add_argument('--masspole', dest='masspole', default=0.1, type=float)
parser.add_argument('--force_magnifier', dest='force_magnifier', default=10.0, type=float)
parser.add_argument('--gravity', dest='gravity', default=9.8, type=float)
parser.add_argument('--update_interval', dest='update_interval', default=0.02, type=float)

# common config
parser.add_argument('--total_steps', dest='total_steps', default=2000000, type=int, help='number of training steps')
parser.add_argument('--save_freq', dest='save_freq', default=500000, type=int, help='frequency of saving checkpoints')
parser.add_argument('--save_path', dest='save_path', default="run_logs", type=str, help='path for model savings')
parser.add_argument('--alg', dest='alg', default="DQN", type=str, help='RL algorithm for training')
parser.add_argument('--test_eps', dest='test_eps', default=10, type=int, help='number of testing episodes')
parser.add_argument('--variant', dest='variant', default=None, type=str, help='varying context in multitask')
parser.add_argument('--resume', dest='resume', default=False, type=bool, help='whether use loaded checkpoint')
args = parser.parse_args()

# create log dir
os.makedirs(args.save_path, exist_ok=True)

# save args to a file
with open(args.save_path + '/args.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": args.total_steps,
    "env_name": args.env_name,
}

if args.env_name == "cartpole":
    context_features = CARLCartPoleEnv.get_context_features()
    new_context = {
        "gravity": args.gravity if args.gravity else context_features["gravity"].default_value,
        "masscart": args.masscart if args.masscart else context_features["masscart"].default_value,
        "masspole": args.masspole if args.masspole else context_features["masspole"].default_value,
        "length": args.pole_length if args.pole_length else context_features["length"].default_value,
        "force_mag": args.force_magnifier if args.force_magnifier else context_features["force_mag"].default_value,
        "tau": args.update_interval if args.update_interval else context_features["tau"].default_value,
        "initial_state_lower": context_features["initial_state_lower"].default_value,
        "initial_state_upper": context_features["initial_state_upper"].default_value,
    }
    contexts = {}
    if args.variant == "3d":
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    new_context_tmp = new_context.copy()
                    new_context_tmp["length"] = new_context["length"] * (i+1) * 0.2
                    new_context_tmp["masscart"] = new_context["masscart"] * (j+1) * 0.2
                    new_context_tmp["masspole"] = new_context["masspole"] * (k+1) * 0.2
                    contexts[100*(i+1) + 10*(j+1) + k+1] = new_context_tmp
    elif args.variant:
        for i in range(100):
            new_context_tmp = new_context.copy()
            new_context_tmp[args.variant] = new_context[args.variant] * (i+1) * 0.1
            contexts[i+1] = new_context_tmp
    env = CARLCartPoleEnv(contexts=contexts)
    env = FlattenObservation(env)
    
    # Separate evaluation env
    eval_env = CARLCartPoleEnv(contexts=contexts)
    eval_env = FlattenObservation(eval_env)
    
else:
    print("Environment not recognized!")
    exit()


env = Monitor(env, filename=args.save_path + "/" + args.env_name + "_data",)
eval_env = Monitor(eval_env, filename=args.save_path + "/" + args.env_name + "_eval_data",)

eval_callback = EvalCallback(eval_env, best_model_save_path=args.save_path, log_path=args.save_path, eval_freq=500)

checkpoint_callback = CheckpointCallback(
  save_freq=args.save_freq,
  save_path=f"{args.save_path}/runs/source_task_training_checkpoints/",
  name_prefix="model",
)

callbacks = CallbackList([checkpoint_callback, eval_callback])

logger = configure(args.save_path + "/" + args.env_name + "_log", ["stdout", "csv", "log"])

# Algorithm selection
algs = {
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "DDPG": DDPG,
    "TD3": TD3
}

if args.alg not in algs:
    raise ValueError(f"RL algorithm '{args.alg}' is not valid")

model_class = algs[args.alg]

# === RESUME LOGIC ===
if args.resume:
    # If no explicit checkpoint path is provided, try to find the latest checkpoint
    # in the checkpoint folder. This is just one strategy; you can tailor this as needed.
    checkpoint_dir = f"{args.save_path}/runs/source_task_training_checkpoints/"
    # find the last checkpoint .zip file in alphabetical order or by step naming
    # zip_files = sorted([f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")])
    # if len(zip_files) == 0:
    #     raise ValueError(f"No checkpoint files found in {checkpoint_dir} to resume from.")
    # last_checkpoint = zip_files[-1]  # e.g. "model_1000000_steps.zip"
    # checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)
    pattern = re.compile(r"model_(\d+)_steps\.zip")

    def get_step_count(filename):
        """Extract step count as integer or return -1 if pattern not matched."""
        match = pattern.match(filename)
        return int(match.group(1)) if match else -1

    zip_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]

    if len(zip_files) == 0:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir} to resume from.")

    # Sort by the integer value of the step count
    zip_files.sort(key=get_step_count)

    # Last element in sorted list has the maximum step count
    last_checkpoint = zip_files[-1]
    checkpoint_path = os.path.join(checkpoint_dir, last_checkpoint)

    print(f"Resuming from checkpoint: {checkpoint_path}")
    # Load the model and pass the environment again
    model = model_class.load(checkpoint_path, env=env)
else:
    # create a new model from scratch
    model = model_class('MlpPolicy', env, verbose=1, tensorboard_log=f"{args.save_path}/runs")

model.set_logger(logger)
model.learn(total_timesteps=args.total_steps, callback=callbacks, reset_num_timesteps=False)

print("Training completed!!")

