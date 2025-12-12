import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.utils import set_random_seed
from cyclesgym.utils.utils import EvalCallbackCustom, _evaluate_policy
from cyclesgym.utils.wandb_utils import WANDB_ENTITY, CROP_PLANNING_EXPERIMENT
from cyclesgym.utils.paths import PROJECT_PATH, CYCLES_PATH
from pathlib import Path
import gym
from cyclesgym.envs.corn import Corn
from cyclesgym.envs.crop_planning import CropPlanning, CropPlanningFixedPlanting
from cyclesgym.envs.crop_planning import CropPlanningFixedPlantingRotationObserver
from cyclesgym.envs.weather_generator import FixedWeatherGenerator, WeatherShuffler
import wandb
from wandb.integration.sb3 import WandbCallback
import random
import argparse
import os
import pandas as pd

class Transfer:
    """ Evaluation by zero-shot transfer """

    def __init__(self, experiment_config) -> None:
        self.config = experiment_config
        # rl config is configured from wandb config

    def create_envs(self):
        eval_env_train = self.env_maker(start_year=self.config['train_start_year'],
                                        end_year=self.config['train_end_year'],
                                        training=False,
                                        env_class=self.config['eval_env_class'],
                                        weather_generator_class=self.config.weather_generator_class,
                                        weather_generator_kwargs=self.config.weather_generator_kwargs)

        return [eval_env_train]

    def env_maker(self, env_class=CropPlanningFixedPlanting, weather_generator_class=FixedWeatherGenerator,
                  weather_generator_kwargs={'base_weather_file': CYCLES_PATH.joinpath('input', 'RockSprings.weather')},
                  training=True, n_procs=1, start_year=1980, end_year=2000, soil_file='GenericHagerstown.soil'):
        if not training:
            n_procs = 1

        if isinstance(env_class, str):
            env_class = globals()[env_class]
        if isinstance(weather_generator_class, str):
            weather_generator_class = globals()[weather_generator_class]

        # Convert to Path since pickle converted it to string
        # base_weather_path = weather_generator_kwargs['base_weather_file']
        weather_generator_kwargs['base_weather_file'] = Path(weather_generator_kwargs['base_weather_file'])

        def make_env():
            # creates a function returning the basic env. Used by SubprocVecEnv later to create a
            # vectorized environment
            def _f():
                env_conf = dict(start_year=start_year, end_year=end_year, soil_file=soil_file,
                                weather_generator_class=weather_generator_class,
                                weather_generator_kwargs=weather_generator_kwargs,
                                rotation_crops=['CornRM.100', 'SoybeanMG.3'])
                env = env_class(**env_conf)

                env = gym.wrappers.RecordEpisodeStatistics(env)
                return env

            return _f

        env = SubprocVecEnv([make_env() for i in range(n_procs)], start_method='fork')
        env = VecMonitor(env)
        norm_reward = (training and self.config['norm_reward'])
        env = VecNormalize(env, norm_obs=True, norm_reward=norm_reward, clip_obs=5000., clip_reward=5000.)
        return env

    def create_callback(self, model_dir):
        eval_freq = int(self.config['eval_freq'] / self.config['n_process'])

        [eval_env_train] = self.create_envs()
        def get_callback(env, suffix, deterministic):
            return EvalCallbackCustom(env, best_model_save_path=str(model_dir.joinpath(suffix)),
                                      log_path=str(model_dir.joinpath(suffix)), eval_freq=eval_freq,
                                      deterministic=deterministic, render=False, eval_prefix=suffix)

        eval_callback_det = get_callback(eval_env_train, 'eval_det', True)
        eval_callback_sto = get_callback(eval_env_train, 'eval_sto', False)

        return [eval_callback_det, eval_callback_sto]

    def train(self):
        train_env = self.env_maker(start_year=self.config['train_start_year'],
                                   end_year=self.config['train_end_year'],
                                   env_class=self.config['env_class'],
                                   training=True, n_procs=self.config['n_process'],
                                   weather_generator_class=self.config['weather_generator_class'],
                                   weather_generator_kwargs=self.config['weather_generator_kwargs'])
        dir = wandb.run.dir
        # make directory of dir
        os.makedirs(dir, exist_ok=True)
        model_dir = Path(dir).joinpath('models')

        if self.config["method"] == "A2C":
            model = A2C('MlpPolicy', train_env, verbose=self.config['verbose'], tensorboard_log=dir,
                        device=self.config['device'])
        elif self.config["method"] == "PPO":
            model = PPO('MlpPolicy', train_env, n_steps=self.config['n_steps'], batch_size=self.config['batch_size'],
                        n_epochs=self.config['n_epochs'], verbose=self.config['verbose'], tensorboard_log=dir,
                        device=self.config['device'])
        elif self.config["method"] == "DQN":
            model = DQN('MlpPolicy', train_env, verbose=self.config['verbose'], tensorboard_log=dir,
                        device=self.config['device'])
        else:
            raise Exception("Not an RL method that has been implemented")

        # The test environment will automatically have the same observation normalization applied to it by
        # EvalCallBack

        callback = self.create_callback(model_dir)

        callback = [WandbCallback(model_save_path=str(model_dir),
                                  model_save_freq=int(self.config['eval_freq'] / self.config['n_process']))] + callback
        model.learn(total_timesteps=self.config["total_timesteps"], callback=callback)

        return model
    
    def transfer(self):
        eval_env = self.env_maker(start_year=self.config['eval_start_year'],
                                    end_year=self.config['eval_end_year'],
                                    env_class=self.config['eval_env_class'],
                                    training=False, 
                                    n_procs=self.config['n_process'],
                                    weather_generator_class=self.config['weather_generator_class'],
                                    weather_generator_kwargs=self.config['weather_generator_kwargs'])
        
        source_dir = self.config['source_dir']
        dir = wandb.run.dir
        
        print(dir)
        os.makedirs(dir, exist_ok=True)
        os.makedirs(self.config['dir'], exist_ok=True)
        
        # assert if source_dir/wandb/latest-run/files/model.zip doesn't exist
        assert os.path.exists(source_dir + "/wandb/latest-run/files/model.zip"), "Model not found in source directory"
        
        if self.config["method"] == "A2C":
            model = A2C.load(source_dir + "/wandb/latest-run/files/model.zip", device=self.config['device'])
        elif self.config["method"] == "PPO":
            model = PPO.load(source_dir + "/wandb/latest-run/files/model.zip", device=self.config['device'])
        elif self.config["method"] == "DQN":
            model = DQN.load(source_dir + "/wandb/latest-run/files/model.zip", device=self.config['device'])
        else:
            raise Exception("Not an RL method that has been implemented")
        
        mean_r_det, fertilizer_table = self.evaluate_log(model, eval_env)
        
        # save the results into a csv file
        os.makedirs(self.config['dir'], exist_ok=True)
        
        fertilizer_table.to_csv(self.config['dir'] + "/fertilizer_table.csv")
        return mean_r_det
        

    def evaluate_log(self, model, eval_env):
        """
        Runs policy deterministically (5 episodes)
        logs the fertilization actions taken by the model

        Parameters
        ----------
        model: trained agent
        eval_env

        Returns
        -------
        mean deterministic reward

        """
        mean_r_det, _, actions_det, episode_rewards_det, _, _ = _evaluate_policy(model,
                                                                           env=eval_env,
                                                                           n_eval_episodes=1,
                                                                           deterministic=True)
        # wandb.log({'deterministic_return': mean_r_det})
        # episode_actions_names = [*list(f"det{i + 1}" for i in range(len(actions_det)))]
        # episode_actions = [*actions_det]
        # fertilizer_table = wandb.Table(
        #     columns=['Run', 'Total Fertilizer', *[f'Week{i}' for i in range(53)]])
        # for i in range(len(episode_actions)):
        #     acts = episode_actions[i]
        #     data = [[week, fert] for (week, fert) in zip(range(53), acts)]
        #     table = wandb.Table(data=data, columns=['Week', 'N added'])
        #     fertilizer_table.add_data(
        #         *[episode_actions_names[i], np.sum(acts), *acts])
        #     wandb.log({f'train/actions/{episode_actions_names[i]}':
        #                    wandb.plot.bar(table, 'Week', 'N added',
        #                                   title=f'Training action sequence {episode_actions_names[i]}')})
        # wandb.log({'train/fertilizer': fertilizer_table})
        
        
        episode_actions_names = [f"det{i + 1}" for i in range(len(actions_det))]
        episode_actions = [*actions_det]
        
        max_len = max(len(acts) for acts in actions_det)

        # Create a DataFrame to store fertilizer data
        columns = ['Run', 'Total Fertilizer'] + [f'Week{i}' for i in range(max_len)] + ['Mean Reward']
        fertilizer_data = []

        for i in range(len(episode_actions)):
            acts = episode_actions[i]
            row = [episode_actions_names[i], np.sum(acts)] + list(acts) + [mean_r_det]
            fertilizer_data.append(row)

        fertilizer_df = pd.DataFrame(fertilizer_data, columns=columns)

        return mean_r_det, fertilizer_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-temp', '--temperature', type=float, default=20.0)
    parser.add_argument('-sun', '--sunlight', type=float, default=15.0)
    parser.add_argument('-prec', '--precipitation', type=float, default=10.0)
    parser.add_argument('-temp_t', '--temperature_t', type=float, default=20.0)
    parser.add_argument('-sun_t', '--sunlight_t', type=float, default=15.0)
    parser.add_argument('-prec_t', '--precipitation_t', type=float, default=10.0)
    parser.add_argument('-fw', '--fixed_weather', default='False',
                        help='Whether to use a fixed weather')
    parser.add_argument('-na', '--non_adaptive', default='False',
                        help='Whether to use a non-adaptive policy (observation space being only a trailing window of'
                             'the crop rotation used so far')
    parser.add_argument('-s', '--seed', type=int, default=0, metavar='N',
                        help='The random seed used for all number generators')
    parser.add_argument('-mul', '--multi', type=bool, default=False, help='Multitask or not')

    args = vars(parser.parse_args())

    set_random_seed(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])

    if args['non_adaptive'] == 'True':
        env_class = 'CropPlanningFixedPlantingRotationObserver'
        eval_env_class = 'CropPlanningFixedPlantingRotationObserver'
    else:
        env_class = 'CropPlanningFixedPlanting'
        eval_env_class = 'CropPlanningFixedPlanting'

    train_start_year = 1980
    train_end_year = 1998
    eval_start_year = 1998
    eval_end_year = 2014

    if args['fixed_weather'] == 'True':
        weather_generator_class = 'FixedWeatherGenerator'
        # if args['multi']:
        #     weather_generator_kwargs = {
        #         'base_weather_file': CYCLES_PATH.joinpath('input', f"Toy_tempmulti_sunmulti_precmulti.weather")}
        # else:
        weather_generator_kwargs = {
            'base_weather_file': CYCLES_PATH.joinpath('input', f"Toy_temp{args['temperature_t']}_sun{args['sunlight_t']}_prec{args['precipitation_t']}.weather")}
    else:
        assert False, "Not considered in CMDP setting yet"

    config = dict(train_start_year=train_start_year, train_end_year=train_end_year, eval_start_year=eval_start_year, eval_end_year=eval_end_year,
                  total_timesteps=100, eval_freq=1000, n_steps=80, batch_size=1, n_epochs=10, run_id=0,
                  norm_reward=True, method="PPO", verbose=1, n_process=1, device='auto',
                  env_class=env_class, eval_env_class=eval_env_class, weather_generator_class=weather_generator_class,
                  weather_generator_kwargs=weather_generator_kwargs, 
                  source_dir=PROJECT_PATH.joinpath(f"results/tempmulti_sunmulti_precmulti_trial{args['seed']}") if args['multi'] else PROJECT_PATH.joinpath(f"results/temp{args['temperature']}_sun{args['sunlight']}_prec{args['precipitation']}_trial{args['seed']}"),
                  dir=PROJECT_PATH.joinpath(f"results/tempmulti_sunmulti_precmulti_trial{args['seed']}/transfer_temp{args['temperature_t']}_sun{args['sunlight_t']}_prec{args['precipitation_t']}_trial{args['seed']}") if args['multi'] else PROJECT_PATH.joinpath(f"results/temp{args['temperature']}_sun{args['sunlight']}_prec{args['precipitation']}_trial{args['seed']}/transfer_temp{args['temperature_t']}_sun{args['sunlight_t']}_prec{args['precipitation_t']}_trial{args['seed']}"))

    config.update(args)

    wandb.init(
        config=config,
        sync_tensorboard=True,
        project=CROP_PLANNING_EXPERIMENT,
        entity=WANDB_ENTITY,
        monitor_gym=False,  # automatically upload gym environements' videos
        save_code=False,
        dir=PROJECT_PATH.joinpath(f"results/tempmulti_sunmulti_precmulti_trial{args['seed']}/transfer_temp{args['temperature_t']}_sun{args['sunlight_t']}_prec{args['precipitation_t']}_trial{args['seed']}") if args['multi'] else PROJECT_PATH.joinpath(f"results/temp{args['temperature']}_sun{args['sunlight']}_prec{args['precipitation']}_trial{args['seed']}/transfer_temp{args['temperature_t']}_sun{args['sunlight_t']}_prec{args['precipitation_t']}_trial{args['seed']}"),
    )

    config = wandb.config

    trainer = Transfer(config)
    mean_r_det = trainer.transfer()
    print(f"mean_r_det: {mean_r_det}")