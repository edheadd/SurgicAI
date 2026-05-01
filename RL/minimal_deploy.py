import os
import argparse
import numpy as np
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
from PyKDL import Frame, Rotation, Vector
from stable_baselines3 import TD3, SAC, PPO

gc.collect()
torch.cuda.empty_cache()
logger = get_logger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Deploy trained RL model on real PSM.")
    parser.add_argument('--algorithm', type=str, required=True, help='Name of the RL algorithm')
    parser.add_argument('--task_name', type=str, required=True, help='Name of the task/environment')
    parser.add_argument('--reward_type', type=str, choices=['dense', 'sparse'], default='sparse', help='Reward type')
    parser.add_argument('--train_seed', type=int, default=1, help='Training seed')
    parser.add_argument('--model-path', type=str, default=None, help='Explicit path to model')
    parser.add_argument('--goal', type=float, nargs=7, default=None, help='Desired goal [x,y,z,roll,pitch,yaw,jaw] in camera frame')
    parser.add_argument('--max-steps', type=int, default=100, help='Max deployment steps')
    return parser.parse_args()

def load_model_for_deploy(algorithm, task_name, reward_type, seed, model_path: str | None):
    if model_path is not None:
        resolved_model_path = Path(model_path).expanduser()
    else:
        # Default path
        resolved_model_path = experiment_dir(ExperimentKey(
            task_name=task_name,
            algorithm=algorithm,
            reward_type=reward_type,
            seed=seed,
            variant="base_env",
        )) / "final_model"

    # Load without env
    if algorithm.upper() == 'TD3':
        model_class = TD3
    elif algorithm.upper() == 'SAC':
        model_class = SAC
    elif algorithm.upper() == 'PPO':
        model_class = PPO
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    model = model_class.load(str(resolved_model_path))
    return model

def deploy_loop(model, psm, step_size, max_steps=100, goal_vec7=None):
    """
    Minimal deployment loop.
    Assumes goal_vec7 is the desired goal in camera frame (7D: x,y,z,roll,pitch,yaw,jaw).
    Poses are assumed to be in camera frame.
    """
    if goal_vec7 is None:
        # For demo, use current pose as goal
        current_mat = psm.m_cp
        current_vec6 = frame_to_vector(current_mat)  # assume current_mat is Frame
        goal_vec7 = np.append(current_vec6, 0.0)  # assume jaw 0

    logger.info("Starting deployment with goal: %s", goal_vec7)

    for step in range(max_steps):
        # Get current observation
        current_mat = psm.m_cp
        if current_mat is None:
            logger.warning("No measured pose yet, skipping")
            time.sleep(0.1)
            continue
        current_vec6 = frame_to_vector(current_mat)  # direct, assuming camera frame
        current_jaw = 0.0  # assume
        achieved_vec7 = np.append(current_vec6, current_jaw)

        # Observation dict like in env
        obs = {
            'achieved_goal': achieved_vec7,
            'desired_goal': goal_vec7,
            'observation': achieved_vec7
        }

        # Predict action
        action, _ = model.predict(obs, deterministic=True)
        logger.info("Step %d: Action %s", step, action)

        # Scale action
        scaled_action = action * step_size

        # New pose
        new_pose_vec6 = achieved_vec7[:6] + scaled_action[:6]

        # Compute IK and publish
        new_frame = Frame(Rotation.RPY(new_pose_vec6[3], new_pose_vec6[4], new_pose_vec6[5]), Vector(new_pose_vec6[0], new_pose_vec6[1], new_pose_vec6[2]))
        ik_solution = psm._kd.compute_IK(new_frame)
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
    setup_logging()

    # Load model
    model = load_model_for_deploy(
        args.algorithm, args.task_name, args.reward_type, args.train_seed, args.model_path
    )
    step_size = default_step_size(trans_step=1.0e-3, angle_step_deg=3.0, jaw_step=0.05)

    # Initialize real PSM
    ral_instance = ral("deploy_psm")
    ral_instance.spin()
    psm = PSM(ral_instance, "PSM2")  # assume PSM2

    time.sleep(1)  # wait for init

    # Run deployment
    goal = np.array(args.goal) if args.goal else None
    deploy_loop(model, psm, step_size, args.max_steps, goal)

if __name__ == "__main__":
    main()