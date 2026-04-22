import argparse
import pickle
import numpy as np
import gymnasium as gym
import Offline_RL_algo.d3rlpy as d3rlpy
from Offline_RL_algo.d3rlpy.dataset import MDPDataset
from Offline_RL_algo.d3rlpy.metrics.evaluators import EnvironmentEvaluator_dict
from algorithm_configs_offline import get_algorithm_config
import torch
import gc
import importlib
from pathlib import Path

from rl_paths import ExperimentKey, ensure_dir, experiment_dir, rl_dir
from RL.utils.cli_args import add_common_logging_args, add_seed_arg, add_threshold_args
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.seed import seed_everything
from RL.utils.utils import resolve_src_env, default_step_size, threshold_from_args

gc.collect()
torch.cuda.empty_cache()
MAX_EPISODE_STEPS = 200
logger = get_logger(__name__)

def load_expert_data(task_name: str, expert_data_path: str | None):
    if expert_data_path:
        p = Path(expert_data_path).expanduser()
    else:
        p = rl_dir() / "Expert_traj" / str(task_name) / "all_episodes_merged.pkl"

    try:
        with open(p, 'rb') as file:
            data = pickle.load(file)
            observations = []
            actions = []
            rewards = []
            terminals = []
            for episode in data:
                observations.extend([episode['obs']['observation']])
                actions.extend([episode['action']])
                rewards.extend([episode['reward']])
                terminals.extend([episode['done']])
            observations = np.array(observations, dtype=np.float32)
            actions = np.array(actions, dtype=np.float32)
            rewards = np.array(rewards, dtype=np.float32)
            terminals = np.array(terminals, dtype=bool)
            logger.info("Loaded expert data from %s", p)
            return MDPDataset(observations, actions, rewards, terminals)
    except FileNotFoundError:
        logger.error("Expert data file not found: %s", p)
        return None

def setup_environment(args):
    step_size = default_step_size(trans_step=1.0e-3, angle_step_deg=3.0, jaw_step=0.05)
    threshold = threshold_from_args(args.trans_error, args.angle_error)
    SRC_class = resolve_src_env(args.task_name)
    gym.envs.register(id=f"{args.algorithm}_{args.reward_type}", entry_point=SRC_class, max_episode_steps=MAX_EPISODE_STEPS)
    env = gym.make(f"{args.algorithm}_{args.reward_type}", render_mode="human", reward_type=args.reward_type,
                   max_episode_step=MAX_EPISODE_STEPS, seed=args.seed, step_size=step_size, threshold=threshold)
    return env

def run_evaluation(env, model, num_episodes, max_episode_steps):
    total_length = 0
    total_timecost = 0
    total_success = 0
    all_lengths = []
    all_timecosts = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        trajectory_length = 0
        for timestep in range(max_episode_steps):
            action = model.predict(np.array([obs['observation']]))[0]
            next_obs, reward, terminated, truncated, info = env.step(action)
            trajectory_length += np.linalg.norm(action[0:3] * env.step_size[0:3] * 1000)
            obs = next_obs
            if terminated:
                total_success += 1
                total_length += trajectory_length
                total_timecost += timestep + 1
                all_lengths.append(trajectory_length)
                all_timecosts.append(timestep + 1)
                break
    
    success_rate = total_success / num_episodes
    avg_length = total_length / total_success if total_success > 0 else 0
    avg_timecost = total_timecost / total_success if total_success > 0 else 0
    
    return success_rate, avg_length, avg_timecost, all_lengths, all_timecosts

def save_results(args, results):
    success_rate, avg_length, avg_timecost, all_lengths, all_timecosts = results
    
    mean_success_rate = success_rate
    std_success_rate = 0  # Since it's a single success rate over multiple episodes
    
    mean_avg_length = np.mean(all_lengths)
    std_avg_length = np.std(all_lengths)
    
    mean_avg_timecost = np.mean(all_timecosts)
    std_avg_timecost = np.std(all_timecosts)
    
    out_dir = ensure_dir(experiment_dir(ExperimentKey(
        task_name=args.task_name,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
        seed=args.seed,
        variant="offline",
    )))
    results_dir = ensure_dir(out_dir / "evaluation_results")
    
    # Save detailed results to txt file
    txt_file = results_dir / f"{args.task_name}_{args.algorithm}_{args.reward_type}_results.txt"
    with open(txt_file, 'w') as f:
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Reward Type: {args.reward_type}\n")
        f.write(f"Evaluation seed: {args.seed}\n\n")
        f.write("Results:\n")
        f.write(f"Success Rate: {mean_success_rate:.2%}\n")
        f.write(f"Average Trajectory Length: {mean_avg_length:.2f} ± {std_avg_length:.2f} mm\n")
        f.write(f"Average Time Cost: {mean_avg_timecost:.2f} ± {std_avg_timecost:.2f} steps\n")
    
    logger.info("Detailed results saved to %s", txt_file)

    # Save numeric results to txt file
    numbers_file = results_dir / f"{args.task_name}_{args.algorithm}_{args.reward_type}_numbers.txt"
    with open(numbers_file, 'w') as f:
        f.write(f"{mean_success_rate} {std_success_rate} ")
        f.write(f"{mean_avg_length} {std_avg_length} ")
        f.write(f"{mean_avg_timecost} {std_avg_timecost}")
    
    logger.info("Numeric results saved to %s", numbers_file)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an offline RL agent.")
    parser.add_argument('--algorithm', type=str, required=True, choices=['CQL', 'CalQL', 'IQL', 'BCQ', 'AWAC'], help='Name of the offline RL algorithm to use')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    parser.add_argument('--n_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--n_steps_per_epoch', type=int, default=200, help='Number of steps per epoch')
    add_seed_arg(parser, name="--seed", default=10)
    add_threshold_args(parser)
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--expert-data', type=str, default=None, help='Optional path to expert trajectories pickle')
    add_common_logging_args(parser)
    return parser.parse_args()

def main():
    args = parse_arguments()
    setup_logging(level=args.log_level, log_file=args.log_file)
    seed_everything(args.seed)
    # d3rlpy seed is handled in seed_everything, but keep explicit call for safety.
    try:
        d3rlpy.seed(int(args.seed))
    except Exception:
        pass

    env = setup_environment(args)
    dataset = load_expert_data(args.task_name, args.expert_data)
    if dataset is None:
        logger.error("No expert data found. Exiting.")
        return

    model = get_algorithm_config(args.algorithm, env, args.task_name, args.reward_type, args.seed, args.use_gpu)

    model.fit(
        dataset,
        n_steps=args.n_epochs * args.n_steps_per_epoch,
        n_steps_per_epoch=args.n_steps_per_epoch,
        experiment_name=f"{args.task_name}_{args.algorithm}_{args.reward_type}",
        with_timestamp=False,
        save_interval=1,
        evaluators={"environment": EnvironmentEvaluator_dict(env)},
        show_progress=True,
    )

    out_dir = ensure_dir(experiment_dir(ExperimentKey(
        task_name=args.task_name,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
        seed=args.seed,
        variant="offline",
    )))
    save_path = out_dir / "final_model.d3"
    model.save(str(save_path))

    logger.info("Starting model evaluation...")
    num_episodes = 20
    evaluation_results = run_evaluation(env, model, num_episodes, MAX_EPISODE_STEPS)
    
    save_results(args, evaluation_results)
    
    env.close()

if __name__ == "__main__":
    main()