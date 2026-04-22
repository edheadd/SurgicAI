import os
import argparse
import numpy as np
import gymnasium as gym
import importlib
from algorithm_configs_online import get_algorithm_config
import gc
import torch
from pathlib import Path

from rl_paths import ExperimentKey, ensure_dir, experiment_dir
from RL.utils.cli_args import add_common_logging_args, add_experiment_variant_arg, add_threshold_args
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.seed import seed_everything
from RL.utils.utils import resolve_src_env, default_step_size, threshold_from_args, experiment_variant

gc.collect()
torch.cuda.empty_cache()
logger = get_logger(__name__)

def setup_environment(args, test_env):
    max_episode_steps = 1000
    step_size = default_step_size(trans_step=1.0e-3, angle_step_deg=3.0, jaw_step=0.05)
    threshold = threshold_from_args(args.trans_error, args.angle_error)
    SRC_class = resolve_src_env(args.task_name)
    
    if test_env == "stepDR_env":
        stepDR = True
    else:
        stepDR = False

    gym.envs.register(id=f"{args.algorithm}_{args.reward_type}", entry_point=SRC_class, max_episode_steps=max_episode_steps)
    env = gym.make(f"{args.algorithm}_{args.reward_type}", render_mode="human", reward_type=args.reward_type,
                   max_episode_step=max_episode_steps, seed=args.eval_seed, step_size=step_size, threshold=threshold, stepDR=stepDR)
    return env, step_size, threshold, max_episode_steps

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate trained RL models.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm to evaluate')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    add_threshold_args(parser)
    parser.add_argument('--eval_seed', type=int, default=42, help='Fixed seed for evaluation')
    # Backwards-compatible flags (kept), plus canonical --variant.
    parser.add_argument('--randomized', action='store_true', help='Model was trained with world randomization enabled')
    parser.add_argument('--stepDR', action='store_true', help='Model was trained with stepDR enabled')
    add_experiment_variant_arg(parser)
    parser.add_argument('--model-path', type=str, default=None, help='Explicit path to model (overrides derived path)')
    parser.add_argument('--train-seeds', type=int, nargs='*', default=None, help='Seeds to evaluate (default: a standard list)')
    parser.add_argument('--num-episodes', type=int, default=20, help='Evaluation episodes per seed')
    add_common_logging_args(parser)
    return parser.parse_args()

def load_model(algorithm, env, task_name, reward_type, seed, randomized, stepDR, model_path: str | None, variant: str | None):
    randomization_str = experiment_variant(
        variant=variant,
        randomized=bool(randomized),
        stepDR=bool(stepDR),
    )

    if model_path is not None:
        resolved_model_path = Path(model_path).expanduser()
    else:
        # Default: reuse the same directory structure as training scripts.
        candidate_variants = [randomization_str]
        if randomization_str == "base_env":
            candidate_variants = ["base_env", "no_randomization"]

        resolved_model_path = None
        for variant in candidate_variants:
            candidate = experiment_dir(ExperimentKey(
                task_name=task_name,
                algorithm=algorithm,
                reward_type=reward_type,
                seed=seed,
                variant=variant,
            )) / "final_model"
            if candidate.exists() or candidate.with_suffix(".zip").exists():
                resolved_model_path = candidate
                break
        if resolved_model_path is None:
            # Last resort: return the derived path even if it doesn't exist,
            # so the error message is actionable.
            resolved_model_path = experiment_dir(ExperimentKey(
                task_name=task_name,
                algorithm=algorithm,
                reward_type=reward_type,
                seed=seed,
                variant=candidate_variants[0],
            )) / "final_model"

    algorithm_config = get_algorithm_config(algorithm, env, task_name, reward_type, seed, None, True)
    model_class = algorithm_config['class']
    return model_class.load(str(resolved_model_path), env=env)

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
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.unwrapped.step(action)
            trajectory_length += np.linalg.norm(action[0:3] * env.unwrapped.step_size[0:3] * 1000)
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

def save_results(args, results, train_seeds, test_env):
    variant = experiment_variant(variant=args.variant, randomized=args.randomized, stepDR=args.stepDR)
    out_dir = ensure_dir(experiment_dir(ExperimentKey(
        task_name=args.task_name,
        algorithm=args.algorithm,
        reward_type=args.reward_type,
        seed=args.eval_seed,
        variant=f"{variant}_evaluation",
    )))
    results_dir = ensure_dir(out_dir / "evaluation_results" / str(test_env))

    # Save detailed results to txt file (include test environment to avoid overwriting)
    safe_test_env = str(test_env)
    #txt_file = os.path.join(results_dir, f"{args.task_name}_{args.algorithm}_{args.reward_type}_{randomization_str}_{safe_test_env}_results.txt")
    txt_file = results_dir / "results.txt"
    with open(txt_file, 'w') as f:
        f.write(f"Task: {args.task_name}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Reward Type: {args.reward_type}\n")
        f.write(f"Number of seeds: {len(train_seeds)}\n")
        f.write(f"Evaluation seed: {args.eval_seed}\n")
        f.write(f"Test Environment: {test_env}\n")
        f.write("Results:\n")
        f.write(f"Success Rate: {results['mean_success_rate']:.2%} ± {results['std_success_rate']:.2%}\n")
        f.write(f"Average Trajectory Length: {results['mean_avg_length']:.2f} ± {results['std_avg_length']:.2f} mm\n")
        f.write(f"Average Time Cost: {results['mean_avg_timecost']:.2f} ± {results['std_avg_timecost']:.2f} steps\n\n\n")
    
    logger.info("Detailed results saved to %s", txt_file)

    numbers_file = results_dir / "numbers.txt"
    with open(numbers_file, 'w') as f:
        f.write(f"{results['mean_success_rate']} {results['std_success_rate']} ")
        f.write(f"{results['mean_avg_length']} {results['std_avg_length']} ")
        f.write(f"{results['mean_avg_timecost']} {results['std_avg_timecost']}")
    
    logger.info("Numeric results saved to %s", numbers_file)

def main():
    args = parse_arguments()
    setup_logging(level=args.log_level, log_file=args.log_file)
    seed_everything(args.eval_seed)
    
    test_envs = ["stepDR_env"] if args.stepDR else ["base_env"]
    
    for test_env in test_envs:
        logger.info("Evaluating in environment: %s", test_env)

        env, step_size, threshold, max_episode_steps = setup_environment(args, test_env)
        
        train_seeds = args.train_seeds if args.train_seeds is not None else [1, 5, 10, 15, 100, 150, 1000, 1500, 10000, 15000]
        all_success_rates = []
        all_lengths = []
        all_timecosts = []
        
        for train_seed in train_seeds:
            model = load_model(
                args.algorithm,
                env,
                args.task_name,
                args.reward_type,
                train_seed,
                args.randomized,
                args.stepDR,
                args.model_path,
                args.variant,
            )
            
            success_rate, avg_length, avg_timecost, lengths, timecosts = run_evaluation(env, model, args.num_episodes, max_episode_steps)
            
            all_success_rates.append(success_rate)
            all_lengths.extend(lengths)
            all_timecosts.extend(timecosts)
            
            logger.info(
                "Seed %s: success=%0.2f%%, avg_len=%0.2fmm, avg_steps=%0.2f",
                train_seed,
                success_rate * 100.0,
                avg_length,
                avg_timecost,
            )
        
        # Calculate mean and standard deviation across all seeds
        mean_success_rate = np.mean(all_success_rates)
        std_success_rate = np.std(all_success_rates)
        mean_avg_length = np.mean(all_lengths)
        std_avg_length = np.std(all_lengths)
        mean_avg_timecost = np.mean(all_timecosts)
        std_avg_timecost = np.std(all_timecosts)
        
        logger.info(
            "Final across seeds: success=%0.2f%%±%0.2f%%, avg_len=%0.2f±%0.2fmm, avg_steps=%0.2f±%0.2f",
            mean_success_rate * 100.0,
            std_success_rate * 100.0,
            mean_avg_length,
            std_avg_length,
            mean_avg_timecost,
            std_avg_timecost,
        )
        
        results = {
            'mean_success_rate': mean_success_rate,
            'std_success_rate': std_success_rate,
            'mean_avg_length': mean_avg_length,
            'std_avg_length': std_avg_length,
            'mean_avg_timecost': mean_avg_timecost,
            'std_avg_timecost': std_avg_timecost
        }

        save_results(args, results, train_seeds, test_env)

        env.close()

if __name__ == "__main__":
    main()