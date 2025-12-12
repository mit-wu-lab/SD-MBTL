from pathlib import Path
import numpy as np

import argparse
from containers.config import Config
from containers.constants import *
from experiment import MainExperiment
from containers.task_context import ContinuousSelector, NetGenTaskContext, PathTaskContext

if __name__ == '__main__':
    # set the number of workers here
    parser = argparse.ArgumentParser(description='Model arguments')
    parser.add_argument('--dir', default='wd/new_exp', type=str, help='Result directory')
    parser.add_argument('--source_path', type=str, help='Source directory')
    parser.add_argument('--kwargs', default='{}', help='args to be added to the config')
    parser.add_argument('--task_context_kwargs', default='{}', help='args to be added to the task_context')
    parser.add_argument('--inflow', default=300, type=float)
    parser.add_argument('--penrate', default=0.2, type=float)
    parser.add_argument('--green', default=35, type=float)
    parser.add_argument('--inflow_multi', default=False, type=bool)
    parser.add_argument('--penrate_multi', default=False, type=bool)
    parser.add_argument('--green_multi', default=False, type=bool)
    parser.add_argument('--ckpt', default=5000, type=int)
    args = parser.parse_args()

    # task = PathTaskContext(**{
    #     'single_approach':True,
    #     'dir':Path('../datasets/salt-lake-city'),
    #     'penetration_rate':0.2,
    #     'aadt_conversion_factor':0.084,  #0.055
    #     **eval(args.task_context_kwargs)
    # })
    # peak_or_off_peak = 'peak' if task.aadt_conversion_factor == 0.084 else 'off-peak'

    # task = NetGenTaskContext(
    #     base_id=[42],
    #     penetration_rate=[0.1],
    #     single_approach=True,
    #     inflow=ContinuousSelector(200, 400, 4),
    #     lane_length=ContinuousSelector(50, 750, 4),
    #     speed_limit=ContinuousSelector(6.5, 30, 4),
    #     green_phase=ContinuousSelector(15, 35, 3), 
    #     red_phase=ContinuousSelector(30, 55, 4),  
    #     offset=ContinuousSelector(0, 1, 5),
    #     # inflow=ContinuousSelector(200, 400, 3),
    #     # lane_length=ContinuousSelector(75, 600,4),
    #     # speed_limit=ContinuousSelector(10, 20, 3), 
    #     # green_phase=ContinuousSelector(25, 35, 3),
    #     # red_phase=ContinuousSelector(30, 45, 3),
    #     # offset=0,       
    # )
    task = NetGenTaskContext(
        base_id=[11],
        penetration_rate=list(np.linspace(0.2, 1.2, 6)*args.penrate) if args.penrate_multi else [args.penrate],
        single_approach=True,
        inflow=list(np.linspace(0.2, 1.2, 6)*args.inflow) if args.inflow_multi else [args.inflow],
        lane_length=[250],
        speed_limit=[14],
        green_phase=list(np.linspace(0.2, 1.2, 6)*args.green) if args.green_multi else [args.green], 
        red_phase=[35],  
        offset=[0],
        # inflow=ContinuousSelector(200, 400, 3),
        # lane_length=ContinuousSelector(75, 600,4),
        # speed_limit=ContinuousSelector(10, 20, 3), 
        # green_phase=ContinuousSelector(25, 35, 3),
        # red_phase=ContinuousSelector(30, 45, 3),
        # offset=0,       
    )

    # training config
    config = Config(
        run_mode='train',
        task_context=task,
        working_dir=Path(args.dir),
        source_dir=Path(args.source_path+"/models/model-"+str(args.ckpt)+".pth") if args.source_path is not None else None,

        ckpt=args.ckpt if args.ckpt is not None else 5000,
        wandb_proj='greenwave_final_models',
        visualize_sumo=False,
        
        stop_penalty=35,
        emission_penalty=3,
        fleet_reward_ratio=0.0,
        
        moves_emissions_models=['68_46'], #'44_74', '44_64', '30_79'
        moves_emissions_models_conditions=['68_46'], #'44_74', '44_64', '30_79'
    )

    # moves_emissions_models_config=['68_46']
    # moves_emissions_models_config=[
    #         '76_65',
    #         '65_87',
    #         '74_70',
    #         '64_92',
    #         '89_62',
    #         '78_90',
    #         '60_72',
    #         '52_88',
    #         '76_65',
    #         '65_87',
    #         '74_70',
    #         '64_92',
    #         '89_62',
    #         '78_90',
    #         '60_72',
    #         '52_88',
    #         '70_73',
    #         '71_81',
    #         '73_84',
    #         '71_86',
    #         '84_74',
    #         '82_79',
    #         '65_81',
    #         '57_86',
    #     ]

    # moves_emissions_models_conditions_config=[REGULAR for _ in range(len(moves_emissions_models_config))]
    
    # # evaluation config
    # config = Config(
    #     trajectories_output=False,
    #     run_mode='full_eval',
    #     task_context=task,
    #     working_dir=Path(args.dir),

    #     enable_wandb=False,
    #     visualize_sumo=True,
        
    #     stop_penalty=35,
    #     emission_penalty=3,
    #     fleet_reward_ratio=0.0,
        
    #     moves_emissions_models=moves_emissions_models_config,
    #     moves_emissions_models_conditions=moves_emissions_models_conditions_config,

    #     episode_to_eval=5000,
    #     full_eval_run_baselines=False,
    #     n_steps=1,
    #     report_uncontrolled_region_metrics=True,
    #     csv_output_custom_name=(str(task.dir)).split('/')[-1] + '_' + peak_or_off_peak,
    # )

    # sumo visualization config
    # config = Config(
    #     run_mode='single_eval',
    #     task_context=task,
    #     working_dir=Path(args.dir),

    #     stop_penalty=35,
    #     emission_penalty=3, 
    #     fleet_reward_ratio=0.0,

    #     enable_wandb=False,
    #     visualize_sumo=True,

    #     moves_emissions_models=moves_emissions_models_config,
    #     moves_emissions_models_conditions=moves_emissions_models_conditions_config,

    #     episode_to_eval=700,
    #     parallelization_size=1,
    # )
    assert len(config.moves_emissions_models) == len(config.moves_emissions_models_conditions), "The evaluations conditions does not have the same dimensions"
    config = config.update({**eval(args.kwargs)})

    main_exp = MainExperiment(config)

    # run experiment
    main_exp.run()
