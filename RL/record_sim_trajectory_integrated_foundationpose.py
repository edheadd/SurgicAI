import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import gc

from stable_baselines3 import TD3, SAC, PPO
from PyKDL import Frame, Rotation, Vector

try:
    from RL.Approach_env import SRC_approach
    from RL.utils.utils import default_step_size, frame_to_vector
except ImportError:
    from Approach_env import SRC_approach
    from utils.utils import default_step_size, frame_to_vector


gc.collect()
torch.cuda.empty_cache()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Record AMBF/SRC simulation trajectory integrated with FoundationPose."
    )

    parser.add_argument("--algorithm", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="Approach")
    parser.add_argument("--reward_type", type=str, default="dense")
    parser.add_argument("--train_seed", type=int, default=10)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--max-steps", type=int, default=100)

    parser.add_argument("--trans-step", type=float, default=1.0e-3)
    parser.add_argument("--angle-step-deg", type=float, default=3.0)
    parser.add_argument("--jaw-step", type=float, default=0.05)

    parser.add_argument(
        "--psm-idx",
        type=int,
        default=2,
        help="Which simulated PSM to shift before recording. Usually 1 or 2.",
    )

    parser.add_argument("--start-dx", type=float, default=0.002)
    parser.add_argument("--start-dy", type=float, default=0.000)
    parser.add_argument("--start-dz", type=float, default=0.000)
    parser.add_argument("--start-droll", type=float, default=0.0)
    parser.add_argument("--start-dpitch", type=float, default=0.0)
    parser.add_argument("--start-dyaw", type=float, default=0.0)

    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Use small random perturbation instead of fixed offset.",
    )
    parser.add_argument("--random-trans-scale", type=float, default=0.002)
    parser.add_argument("--random-angle-scale-deg", type=float, default=5.0)

    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/xsun97/SurgicAI/RL/sim_trajectories",
    )

    return parser.parse_args()


def load_model(algorithm, model_path, env):
    algorithm = algorithm.upper()

    if algorithm == "TD3":
        model_class = TD3
    elif algorithm == "SAC":
        model_class = SAC
    elif algorithm == "PPO":
        model_class = PPO
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model = model_class.load(
        str(Path(model_path).expanduser()),
        env=env,
        custom_objects={
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        },
    )
    return model


def make_env(args):
    step_size = default_step_size(
        trans_step=args.trans_step,
        angle_step_deg=args.angle_step_deg,
        jaw_step=args.jaw_step,
    )

    try:
        return SRC_approach(
            step_size=step_size,
            reward_type=args.reward_type,
            train_seed=args.train_seed,
        )
    except TypeError:
        pass

    try:
        return SRC_approach(
            step_size=step_size,
            reward_type=args.reward_type,
        )
    except TypeError:
        pass

    try:
        return SRC_approach(step_size=step_size)
    except TypeError:
        pass

    env = SRC_approach()
    env.step_size = step_size
    return env


def reset_env(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out
    return out, {}


def step_env(env, action):
    out = env.step(action)

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, terminated or truncated, info

    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info

    raise RuntimeError(f"Unexpected env.step output length: {len(out)}")


def to_numpy_safe(x):
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        return x.astype(float).tolist()

    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=float).tolist()

    try:
        return np.asarray(x, dtype=float).tolist()
    except Exception:
        return str(x)


def flatten_obs(obs):
    if isinstance(obs, dict):
        return {k: to_numpy_safe(v) for k, v in obs.items()}
    return to_numpy_safe(obs)


def get_psm(env, psm_idx):
    return env.scene_manager.psm_list[psm_idx - 1]


def get_psm_pose_vec(env, psm_idx):
    try:
        psm = get_psm(env, psm_idx)
        T = psm.measured_cp()
        return frame_to_vector(T).astype(float).tolist()
    except Exception as e:
        return {"error": str(e)}


def get_psm_joint_vec(env, psm_idx):
    try:
        psm = get_psm(env, psm_idx)
        jp = getattr(psm, "_measured_jp", None)
        if jp is None:
            return None
        return np.asarray(jp, dtype=float).tolist()
    except Exception as e:
        return {"error": str(e)}


def shift_start_pose(env, args):
    """
    Move simulated PSM slightly before starting trajectory recording.
    This affects simulation only through psm.servo_cp().
    """
    psm = get_psm(env, args.psm_idx)

    T_current = psm.measured_cp()
    current_vec = frame_to_vector(T_current)

    if args.integrated_foundationpose_random:
        rng = np.random.default_rng(args.train_seed + int(time.time()) % 100000)

        dxyz = rng.uniform(
            low=-args.random_trans_scale,
            high=args.random_trans_scale,
            size=3,
        )

        angle_scale = np.deg2rad(args.random_angle_scale_deg)
        drpy = rng.uniform(
            low=-angle_scale,
            high=angle_scale,
            size=3,
        )
    else:
        dxyz = np.array([args.start_dx, args.start_dy, args.start_dz], dtype=float)
        drpy = np.array(
            [args.start_droll, args.start_dpitch, args.start_dyaw],
            dtype=float,
        )

    shifted_vec = current_vec.copy()
    shifted_vec[:3] += dxyz
    shifted_vec[3:6] += drpy

    T_shifted = Frame(
        Rotation.RPY(
            shifted_vec[3],
            shifted_vec[4],
            shifted_vec[5],
        ),
        Vector(
            shifted_vec[0],
            shifted_vec[1],
            shifted_vec[2],
        ),
    )

    print("[RECORD FOUNDATIONPOSE] Original start pose:", current_vec)
    print("[RECORD FOUNDATIONPOSE] Applied dxyz:", dxyz)
    print("[RECORD FOUNDATIONPOSE] Applied drpy:", drpy)
    print("[RECORD FOUNDATIONPOSE] Shifted start pose:", shifted_vec)

    psm.servo_cp(T_shifted)

    # Give AMBF time to apply the new simulated start command.
    time.sleep(1.0)

    return {
        "psm_idx": args.psm_idx,
        "original_start_pose": current_vec.astype(float).tolist(),
        "dxyz": dxyz.astype(float).tolist(),
        "drpy": drpy.astype(float).tolist(),
        "integrated_foundationpose_pose": shifted_vec.astype(float).tolist(),
    }


def main():
    args = parse_arguments()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1) * 1000):03d}"

    suffix = "integrated_foundationpose_random" if args.integrated_foundationpose_random else "integrated_foundationpose"
    json_path = out_dir / f"trajectory_{args.task_name}_{args.algorithm}_{suffix}_{timestamp}.json"
    npz_path = out_dir / f"trajectory_{args.task_name}_{args.algorithm}_{suffix}_{timestamp}.npz"
    csv_path = out_dir / f"trajectory_{args.task_name}_{args.algorithm}_{suffix}_{timestamp}.csv"

    print("[RECORD FOUNDATIONPOSE] Creating simulation environment...")
    env = make_env(args)

    print("[RECORD FOUNDATIONPOSE] Loading model...")
    model = load_model(args.algorithm, args.model_path, env)

    obs, info = reset_env(env)
    print("[RECORD FOUNDATIONPOSE] Reset complete.")

    time.sleep(1.0)

    start_info = shift_start_pose(env, args)

    # Refresh observation after moving start pose.
    obs, info = reset_env(env)
    time.sleep(1.0)

    # Apply shift again after reset, because some envs reset the PSM pose.
    start_info = shift_start_pose(env, args)

    records = []
    action_list = []
    reward_list = []
    done_list = []
    psm1_pose_list = []
    psm2_pose_list = []

    print("[RECORD FOUNDATIONPOSE] Starting rollout from FoundationPose-integrated simulated start.")

    for step in range(args.max_steps):
        action, _ = model.predict(obs, deterministic=True)

        psm1_pose_before = get_psm_pose_vec(env, 1)
        psm2_pose_before = get_psm_pose_vec(env, 2)
        psm1_joint_before = get_psm_joint_vec(env, 1)
        psm2_joint_before = get_psm_joint_vec(env, 2)

        next_obs, reward, done, info = step_env(env, action)

        psm1_pose_after = get_psm_pose_vec(env, 1)
        psm2_pose_after = get_psm_pose_vec(env, 2)
        psm1_joint_after = get_psm_joint_vec(env, 1)
        psm2_joint_after = get_psm_joint_vec(env, 2)

        record = {
            "step": step,
            "time_wall": time.time(),
            "start_info": start_info if step == 0 else None,
            "observation": flatten_obs(obs),
            "action": to_numpy_safe(action),
            "reward": float(reward),
            "done": bool(done),
            "next_observation": flatten_obs(next_obs),
            "info": str(info),
            "psm1_pose_before": psm1_pose_before,
            "psm2_pose_before": psm2_pose_before,
            "psm1_pose_after": psm1_pose_after,
            "psm2_pose_after": psm2_pose_after,
            "psm1_joint_before": psm1_joint_before,
            "psm2_joint_before": psm2_joint_before,
            "psm1_joint_after": psm1_joint_after,
            "psm2_joint_after": psm2_joint_after,
        }

        records.append(record)

        action_list.append(np.asarray(action, dtype=float))
        reward_list.append(float(reward))
        done_list.append(bool(done))

        if isinstance(psm1_pose_after, list):
            psm1_pose_list.append(np.asarray(psm1_pose_after, dtype=float))
        if isinstance(psm2_pose_after, list):
            psm2_pose_list.append(np.asarray(psm2_pose_after, dtype=float))

        print(
            f"[RECORD FOUNDATIONPOSE] step={step:04d}, "
            f"reward={float(reward):+.6f}, "
            f"done={done}, "
            f"action={np.asarray(action)}"
        )

        obs = next_obs

        if done:
            print(f"[RECORD FOUNDATIONPOSE] Episode finished at step {step}.")
            break

    with open(json_path, "w") as f:
        json.dump(
            {
                "start_info": start_info,
                "records": records,
            },
            f,
            indent=2,
        )

    np.savez(
        npz_path,
        actions=np.asarray(action_list, dtype=float),
        rewards=np.asarray(reward_list, dtype=float),
        dones=np.asarray(done_list, dtype=bool),
        psm1_poses=np.asarray(psm1_pose_list, dtype=float) if psm1_pose_list else np.empty((0, 6)),
        psm2_poses=np.asarray(psm2_pose_list, dtype=float) if psm2_pose_list else np.empty((0, 6)),
        start_original_pose=np.asarray(start_info["original_start_pose"], dtype=float),
        start_shifted_pose=np.asarray(start_info["integrated_foundationpose_pose"], dtype=float),
        start_dxyz=np.asarray(start_info["dxyz"], dtype=float),
        start_drpy=np.asarray(start_info["drpy"], dtype=float),
    )

    with open(csv_path, "w") as f:
        header = [
            "step",
            "reward",
            "done",
            "action_0",
            "action_1",
            "action_2",
            "action_3",
            "action_4",
            "action_5",
            "action_6",
            "psm1_x",
            "psm1_y",
            "psm1_z",
            "psm1_roll",
            "psm1_pitch",
            "psm1_yaw",
            "psm2_x",
            "psm2_y",
            "psm2_z",
            "psm2_roll",
            "psm2_pitch",
            "psm2_yaw",
        ]
        f.write(",".join(header) + "\n")

        for r in records:
            action = r["action"]
            if not isinstance(action, list):
                action = [np.nan] * 7
            action = action + [np.nan] * (7 - len(action))

            psm1 = r["psm1_pose_after"]
            psm2 = r["psm2_pose_after"]

            if not isinstance(psm1, list):
                psm1 = [np.nan] * 6
            if not isinstance(psm2, list):
                psm2 = [np.nan] * 6

            row = [
                r["step"],
                r["reward"],
                int(r["done"]),
                *action[:7],
                *psm1[:6],
                *psm2[:6],
            ]
            f.write(",".join(str(x) for x in row) + "\n")

    print("[RECORD FOUNDATIONPOSE] Saved trajectory:")
    print(f"  JSON: {json_path}")
    print(f"  NPZ : {npz_path}")
    print(f"  CSV : {csv_path}")


if __name__ == "__main__":
    main()
