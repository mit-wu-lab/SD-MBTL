import os
import argparse
import json
from carl.envs.gymnasium.classic_control import CARLCartPole as CARLCartPoleEnv
from carl.envs.gymnasium.box2d import CARLBipedalWalker as CARLBipedalWalkerEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC, DDPG, TD3, DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback, EvalCallback
from stable_baselines3.common.logger import configure

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', dest='env_name', type=str, help='environment type')

# cartpole
parser.add_argument('--pole_length', dest='pole_length', default=0.5, type=float)
parser.add_argument('--masscart', dest='masscart', default=1.0, type=float)
parser.add_argument('--masspole', dest='masspole', default=0.1, type=float)
parser.add_argument('--force_magnifier', dest='force_magnifier', default=10.0, type=float)
parser.add_argument('--gravity', dest='gravity', default=9.8, type=float)
parser.add_argument('--update_interval', dest='update_interval', default=0.02, type=float)

# Walker
parser.add_argument('--wk_GRAVITY_Y', dest='wk_GRAVITY_Y', default=10, type=float)
parser.add_argument('--wk_FRICTION', dest='wk_FRICTION', default=2.5, type=float)
parser.add_argument('--wk_MOTORS_TORQUE', dest='wk_MOTORS_TORQUE', default=80, type=float)
parser.add_argument('--wk_SCALE', dest='wk_SCALE', default=30, type=float)
parser.add_argument('--wk_LEG_H', dest='wk_LEG_H', default=1.13, type=float)
parser.add_argument('--wk_LEG_W', dest='wk_LEG_W', default=0.26, type=float)

# common config
parser.add_argument('--total_steps', dest='total_steps', default=2000000, type=int, help='number of training steps')
parser.add_argument('--save_freq', dest='save_freq', default=500000, type=int, help='frequency of saving checkpoints')
parser.add_argument('--save_path', dest='save_path', default="run_logs", type=str, help='path for model savings')
parser.add_argument('--alg', dest='alg', default="DQN", type=str, help='RL algorithm for training')
parser.add_argument('--test_eps', dest='test_eps', default=10, type=int, help='number of testing episodes')
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

    # Create a new context dictionary with updated values
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
    contexts = {0: new_context}
    env = CARLCartPoleEnv(contexts=contexts)
    env = FlattenObservation(env)
    
    # Separate evaluation env
    eval_env = CARLCartPoleEnv(contexts=contexts)
    eval_env = FlattenObservation(eval_env)
elif args.env_name == "walker":    
    context_features = CARLBipedalWalkerEnv.get_context_features()

    new_context = {
        "FPS": context_features["FPS"].default_value,
        "GRAVITY_Y": -args.wk_GRAVITY_Y if args.wk_GRAVITY_Y else -context_features["GRAVITY_Y"].default_value,
        "GRAVITY_X": context_features["GRAVITY_X"].default_value,
        "SCALE": args.wk_SCALE if args.wk_SCALE else context_features["SCALE"].default_value,
        "FRICTION": args.wk_FRICTION if args.wk_FRICTION else context_features["FRICTION"].default_value,
        "MOTORS_TORQUE": args.wk_MOTORS_TORQUE if args.wk_MOTORS_TORQUE else context_features["MOTORS_TORQUE"].default_value,
        "LEG_H": args.wk_LEG_H if args.wk_LEG_H else context_features["LEG_H"].default_value,
        "LEG_W": args.wk_LEG_W if args.wk_LEG_W else context_features["LEG_W"].default_value,
        "TERRAIN_STEP": context_features["TERRAIN_STEP"].default_value,
        "TERRAIN_LENGTH": context_features["TERRAIN_LENGTH"].default_value,
        "TERRAIN_HEIGHT": context_features["TERRAIN_HEIGHT"].default_value,
        "TERRAIN_GRASS": context_features["TERRAIN_GRASS"].default_value,
        "TERRAIN_STARTPAD": context_features["TERRAIN_STARTPAD"].default_value,
        "SPEED_HIP": context_features["SPEED_HIP"].default_value,
        "SPEED_KNEE": context_features["SPEED_KNEE"].default_value,
        "LIDAR_RANGE": context_features["LIDAR_RANGE"].default_value,
        "LEG_DOWN": context_features["LEG_DOWN"].default_value,
        "INITIAL_RANDOM": context_features["INITIAL_RANDOM"].default_value,
        "VIEWPORT_W": context_features["VIEWPORT_W"].default_value,
        "VIEWPORT_H": context_features["VIEWPORT_H"].default_value,
    }
    contexts = {0: new_context}
    env = CARLBipedalWalkerEnv(contexts=contexts)
    env = FlattenObservation(env)
    
    # Separate evaluation env
    eval_env = CARLBipedalWalkerEnv(contexts=contexts)
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
    "TD3": TD3,
}

if args.alg not in algs:
    raise ValueError(f"RL algorithm '{args.alg}' is not valid")

model_class = algs[args.alg]
model = model_class('MlpPolicy', env, verbose=1, tensorboard_log=f"{args.save_path}/runs")
model.set_logger(logger)
model.learn(total_timesteps=args.total_steps, callback=callbacks)

print("Training completed!!")