import os
import argparse
import numpy as np
import importlib
from algorithm_configs_online import get_algorithm_config
import gc
import torch
from pathlib import Path
import time

from rl_paths import ExperimentKey, experiment_dir
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.utils import default_step_size, frame_to_vector
from sensor_msgs.msg import JointState
from RL.utils.kinematics.DH import enforce_limits
from RL.utils.real_psm_arm import PSM
from ros_abstraction_layer import ral

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
    parser = argparse.ArgumentParser(description="Deploy trained RL model on real PSM.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    parser.add_argument('--train_seed', type=int, default=1, help='Training seed')
    parser.add_argument('--randomized', action='store_true', help='Model was trained with randomization')
    parser.add_argument('--stepDR', action='store_true', help='Model was trained with stepDR')
    add_experiment_variant_arg(parser)
    parser.add_argument('--model-path', type=str, default=None, help='Explicit path to model')
    parser.add_argument('--goal', type=float, nargs=7, default=None, help='Desired goal [x,y,z,roll,pitch,yaw,jaw] in camera frame')
    parser.add_argument('--max-steps', type=int, default=100, help='Max deployment steps')
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

def load_model_for_deploy(algorithm, task_name, reward_type, seed, randomized, stepDR, model_path: str | None, variant: str | None):
    randomization_str = experiment_variant(
        variant=variant,
        randomized=bool(randomized),
        stepDR=bool(stepDR),
    )

    if model_path is not None:
        resolved_model_path = Path(model_path).expanduser()
    else:
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
            resolved_model_path = experiment_dir(ExperimentKey(
                task_name=task_name,
                algorithm=algorithm,
                reward_type=reward_type,
                seed=seed,
                variant=candidate_variants[0],
            )) / "final_model"

    # For deployment, we don't need the full env, but we need to create a dummy env to load the model
    # Since the model expects the env for normalization, etc.
    # But to minimize, perhaps load without env, but Stable Baselines needs env.
    # So, create a minimal env.
    max_episode_steps = 1000
    step_size = default_step_size(trans_step=1.0e-3, angle_step_deg=3.0, jaw_step=0.05)
    threshold = [0.005, np.deg2rad(30)]  # default
    SRC_class = resolve_src_env(task_name)
    gym.envs.register(id=f"{algorithm}_{reward_type}_deploy", entry_point=SRC_class, max_episode_steps=max_episode_steps)
    env = gym.make(f"{algorithm}_{reward_type}_deploy", render_mode=None, reward_type=reward_type,
                   max_episode_step=max_episode_steps, seed=42, step_size=step_size, threshold=threshold, stepDR=False)

    algorithm_config = get_algorithm_config(algorithm, env, task_name, reward_type, seed, None, True)
    model_class = algorithm_config['class']
    model = model_class.load(str(resolved_model_path), env=env)
    return model, env, step_size

def deploy_loop(model, psm, step_size, max_steps=100, goal_vec7=None):
    """
    Minimal deployment loop.
    Assumes goal_vec7 is the desired goal in camera frame (7D: x,y,z,roll,pitch,yaw,jaw).
    If None, use current pose as goal (for testing).
    """
    if goal_vec7 is None:
        # For demo, use current pose as goal
        current_mat = psm.m_cp
        current_vec6 = frame_to_vector(convert_mat_to_frame(current_mat))
        goal_vec7 = np.append(current_vec6, 0.0)  # assume jaw 0

    logger.info("Starting deployment with goal: %s", goal_vec7)

    for step in range(max_steps):
        # Get current observation
        current_mat = psm.m_cp
        if current_mat is None:
            logger.warning("No measured pose yet, skipping")
            time.sleep(0.1)
            continue
        current_vec6 = frame_to_vector(convert_mat_to_frame(current_mat))
        current_jaw = 0.0  # assume or get from somewhere
        achieved_vec7 = np.append(current_vec6, current_jaw)

        # Observation dict like in env
        obs = {
            'achieved_goal': achieved_vec7,
            'desired_goal': goal_vec7,
            'observation': achieved_vec7  # or whatever the model expects
        }

        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        logger.info("Step %d: Action %s", step, action)

        # Scale action
        scaled_action = action * step_size

        # Apply to PSM
        # Compute IK and publish
        from sensor_msgs.msg import JointState
        from RL.utils.kinematics.DH import enforce_limits
        T_t_b = convert_mat_to_frame(new_pose)  # new_pose is 6D vector
        ik_solution = psm._kd.compute_IK(T_t_b)
        ik_solution = enforce_limits(ik_solution, psm._kd.JOINT_LIMITS_LOWER, psm._kd.JOINT_LIMITS_UPPER)
        _cmd = JointState()
        _cmd.position = ik_solution[:6].tolist()
        psm.psm_pub.publish(_cmd)
        logger.info("Published command: %s", _cmd.position)

        time.sleep(0.1)  # control rate

        # Check if close to goal
        dist_trans = np.linalg.norm(achieved_vec7[:3] - goal_vec7[:3])
        dist_angle = np.linalg.norm(achieved_vec7[3:6] - goal_vec7[3:6])
        if dist_trans < 0.005 and dist_angle < np.deg2rad(5):
            logger.info("Reached goal at step %d", step)
            break

def main():
    args = parse_arguments()
    setup_logging(level=args.log_level, log_file=args.log_file)

    # Load model
    model, env, step_size = load_model_for_deploy(
        args.algorithm, args.task_name, args.reward_type, args.train_seed,
        args.randomized, args.stepDR, args.model_path, args.variant
    )

    # Initialize real PSM
    ral_instance = ral("deploy_psm")
    ral_instance.spin()
    psm = PSM(ral_instance, "PSM2")  # assume PSM2

    # Start command sending thread
    import threading
    cmd_thread = threading.Thread(target=psm.send_cmds, daemon=True)
    cmd_thread.start()

    time.sleep(1)  # wait for init

    # Run deployment
    goal = np.array(args.goal) if args.goal else None
    deploy_loop(model, psm, step_size, args.max_steps, goal)

    env.close()

if __name__ == "__main__":
    main()