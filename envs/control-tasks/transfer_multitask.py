import argparse
from carl.envs.gymnasium.classic_control import CARLCartPole as CARLCartPoleEnv
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import PPO, SAC, DDPG, TD3, DQN
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', dest='env_name', type=str, help='environment type')

# cartpole
parser.add_argument('--pole_length', dest='pole_length', default=0.5, type=float)
parser.add_argument('--masscart', dest='masscart', default=1.0, type=float)
parser.add_argument('--masspole', dest='masspole', default=0.1, type=float)
parser.add_argument('--force_magnifier', dest='force_magnifier', default=10.0, type=float)
parser.add_argument('--gravity', dest='gravity', default=9.8, type=float)
parser.add_argument('--update_interval', dest='update_interval', default=0.02, type=float)
parser.add_argument('--source_masscart', dest='masscart', default=1.0, type=float)
parser.add_argument('--source_lenpole', dest='source_lenpole', default=0.5, type=float)
parser.add_argument('--source_force', dest='source_force', default=10.0, type=float)

# common config
parser.add_argument('--total_steps', dest='total_steps', default=2000000, type=int, help='number of training steps')
parser.add_argument('--save_freq', dest='save_freq', default=100000, type=int, help='frequency of saving checkpoints')
parser.add_argument('--save_path', dest='save_path', default="run_logs", type=str, help='path for model savings')
parser.add_argument('--source_path', dest='source_path', default="run_logs", type=str, help='path for source trained model')
parser.add_argument('--alg', dest='alg', default="DQN", type=str, help='RL algorithm for training')
parser.add_argument('--test_eps', dest='test_eps', default=10, type=int, help='number of testing episodes')
parser.add_argument('--transfer_step', dest='transfer_step', default=100000, type=int, help='checkpoint to be used for tranfer learning')
parser.add_argument('--trial', dest='trial', default=0, type=int, help='trial number')
args = parser.parse_args()


# test the best model and get performance score
print("[INFO] TEST MODE ENABLED\n")

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
    test_env = CARLCartPoleEnv(contexts=contexts)
    test_env = FlattenObservation(test_env)

else:
    print("Environment not recognized!")
    exit()



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
model = model_class.load(args.source_path, env=test_env)
print(f"Loaded trained {args.alg} model from {args.source_path}")

avg_reward = 0
avg_reward_list = []
for ep in range(args.test_eps):
    #  do not update them at test time
    test_env.training = False

    # test the trained agent
    obs, _ = test_env.reset()
    i = 0
    tot_reward = 0

    while(True):
        action, _ = model.predict(obs, deterministic=True) 
        obs, reward, terminated, truncated, info = test_env.step(action)
        i+=1
        tot_reward += reward
        if terminated or truncated:
            print(f"Test episode reward = {tot_reward} and episode length = {i}")
            avg_reward += tot_reward
            break
    avg_reward_list.append(tot_reward)
# save avg_reward_list as csv
np.savetxt(args.save_path + "/test_reward.csv", avg_reward_list, delimiter=",")

print(f"Test average reward = {avg_reward/args.test_eps}")
