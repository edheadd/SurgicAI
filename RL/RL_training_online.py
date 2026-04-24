import argparse
import pickle
import numpy as np
import gymnasium as gym
import importlib
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from algorithm_configs_online import get_algorithm_config
import gc
import torch
import sys
import threading
from pathlib import Path

from rl_paths import ExperimentKey, checkpoints_dir, ensure_dir, experiment_dir, rl_dir
from RL.utils.cli_args import add_common_logging_args, add_experiment_variant_arg, add_seed_arg, add_threshold_args
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.seed import seed_everything
from RL.utils.utils import resolve_src_env, default_step_size, threshold_from_args, experiment_variant

gc.collect()
torch.cuda.empty_cache()
logger = get_logger(__name__)

def _try_load_pickle(path: Path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def load_expert_data(task_name: str, expert_data_path: str | None):
    """
    Load expert trajectories if provided/available.

    Search order:
    - `--expert-data PATH` if provided
    - `RL/Expert_traj/<task_name>/all_episodes_merged.pkl` (historical default)
    """
    candidates: list[Path] = []
    if expert_data_path:
        candidates.append(Path(expert_data_path).expanduser())
    candidates.append(rl_dir() / "Expert_traj" / str(task_name) / "all_episodes_merged.pkl")

    for p in candidates:
        data = _try_load_pickle(p)
        if data is not None:
            logger.info("Loaded expert data from %s", p)
            return data

    logger.warning("No expert data found (continuing without expert trajectories).")
    return None

def create_model(args, env, expert_data):
    algorithm_config = get_algorithm_config(args.algorithm, env, args.task_name, args.reward_type, args.seed, expert_data)
    model_class = algorithm_config['class']
    model_params = algorithm_config['params']
    return model_class(**model_params)

def setup_environment(args):
    max_episode_steps = 1000
    step_size = default_step_size(trans_step=2.0e-3, angle_step_deg=3.0, jaw_step=0.05)
    threshold = threshold_from_args(args.trans_error, args.angle_error)
    SRC_class = resolve_src_env(args.task_name)
    
    gym.envs.register(id=f"{args.algorithm}_{args.reward_type}", entry_point=SRC_class, max_episode_steps=max_episode_steps)
    env = gym.make(f"{args.algorithm}_{args.reward_type}", render_mode="human", reward_type=args.reward_type,
                   max_episode_steps=max_episode_steps, seed=args.seed, step_size=step_size, threshold=threshold, stepDR=args.stepDR)
    return env, step_size, threshold, max_episode_steps

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm to use')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='dense', help='Reward type')
    parser.add_argument('--total_timesteps', type=int, default=150000, help='Total timesteps for training')
    parser.add_argument('--save_freq', type=int, default=50000, help='Frequency of saving checkpoints')
    add_seed_arg(parser, name="--seed", default=10)
    add_threshold_args(parser)
    parser.add_argument('--randomization_params', type=str, default='0,0,0,0,0', help='Randomization parameters')
    parser.add_argument('--stepDR', action='store_true', help='Enable state-space domain randomization')
    add_experiment_variant_arg(parser)
    parser.add_argument('--gui', action='store_true', help='Enable GUI for domain randomization')
    parser.add_argument('--expert-data', type=str, default=None, help='Optional path to expert trajectories pickle')
    add_common_logging_args(parser)
    return parser.parse_args()

def run_training(args, env):
      
    
    # Load expert data
    expert_data = load_expert_data(args.task_name, args.expert_data)
    
    # Create the model
    model = create_model(args, env, expert_data)
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(ensure_dir(checkpoints_dir(ExperimentKey(
            task_name=args.task_name,
            algorithm=args.algorithm,
            reward_type=args.reward_type,
            seed=args.seed,
            variant="base_env",
        )))),
        name_prefix="rl_model"
    )
        
    # callback_list = CallbackList([checkpoint_callback, domain_randomization_callback])
    callback_list = CallbackList([checkpoint_callback])
    
    # Train the model
    model.learn(total_timesteps=args.total_timesteps, progress_bar=True, callback=callback_list, reset_num_timesteps=False)
    
    # Save the final model
    
    randomization_str = experiment_variant(
        variant=args.variant,
        stepDR=bool(args.stepDR),
        randomization_params=str(args.randomization_params),
    )
    
    out_dir = ensure_dir(experiment_dir(ExperimentKey(
        task_name=args.task_name,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
        seed=args.seed,
        variant=randomization_str,
    )))
    save_path = out_dir / "final_model"
    model.save(str(save_path))
    logger.info("Final model saved to %s", save_path)

    env.close()


if __name__ == "__main__":
    
    args = parse_arguments()
    setup_logging(level=args.log_level, log_file=args.log_file)
    seed_everything(args.seed)
    
    # Setup the environment
    env, step_size, threshold, max_episode_steps = setup_environment(args)
    domain_randomization_callback = None
    if args.randomization_params != "0,0,0,0,0" or args.gui:
        # Import optional ROS/Qt dependencies only when requested.
        from Domain_randomization.Domain_callback import DomainRandomizationCallback
        domain_randomization_callback = DomainRandomizationCallback(env, args.randomization_params, args.seed)

        if args.gui:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QTimer

            app = QApplication(sys.argv)
            # Delay GUI creation until Qt event loop starts.
            QTimer.singleShot(0, lambda: domain_randomization_callback.start_gui(app))

    # Start RL training in a background thread
    training_thread = threading.Thread(target=run_training, args=(args, env,))
    training_thread.start()

    # if args.gui:
    #     sys.exit(app.exec_())
