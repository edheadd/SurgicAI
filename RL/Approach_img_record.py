import sys
# path_to_add = '/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2'
# sys.path.append(path_to_add)

import gymnasium as gym
import numpy as np
import time
import pickle
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ros_abstraction_layer import ral
import os
import cv2
from Domain_randomization.Domain_callback import DomainRandomizationCallback
from pathlib import Path

from rl_paths import data_dir, ensure_dir
from RL.utils.seed import seed_everything
from RL.utils.cli_args import add_experiment_variant_arg
from RL.utils.utils import experiment_variant, resolve_src_env


randomize_env = False
randomize_step_size = True

current_images = {}
image_received = {}
bridge = CvBridge()

ral_instance = ral("image_record_node")
ral_instance.spin()  # Start RAL spinning to process callbacks
time.sleep(0.5)  # Allow some time for RAL to initialize


def image_callback(msg, camera_id):
    """Callback to process and save images from different cameras."""
    global current_images, image_received
    try:
        current_images[camera_id] = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_received[camera_id] = True
    except Exception:
        # Keep recording robust even if a frame occasionally fails to decode.
        pass


# Subscribe to front and back camera images
# camera_topics = {
#     'front': '/ambf/env/cameras/cameraL/ImageData',
#     'back': '/ambf/env/cameras/normal_camera/ImageData'
# }

camera_topics = {
    "front": "/ambf/env/cameras/cameraL/ImageData",
}

for cam_id, topic in camera_topics.items():
    ral_instance.subscriber(topic, Image, lambda msg, cam_id=cam_id: image_callback(msg, cam_id), queue_size=10, latch=False)
    image_received[cam_id] = False


def wait_for_images():
    """Wait for all cameras to have received an image."""
    rate = ral_instance.create_rate(100)
    while not all(image_received.values()) and not ral_instance.is_shutdown():
        rate.sleep()
    for key in image_received:
        image_received[key] = False


# Standardized vis-DR output layout.
#
# Prefer `--variant` to keep folder naming consistent with train/eval scripts.
# You can still override the entire root with `SURGICAI_APPROACH_VISDR_DIR`.
import argparse

parser = argparse.ArgumentParser(description="Record expert transitions + images for Image-IL.")
add_experiment_variant_arg(parser)
parser.add_argument("--task_name", type=str, default="Approach", help="Task/environment name")
args, _unknown = parser.parse_known_args()

_VARIANT = experiment_variant(variant=args.variant)
_DEFAULT_VISDR_DIR = Path(data_dir()) / f"{args.task_name}_td3" / _VARIANT
_VISDR_DIR = Path(os.environ.get("SURGICAI_APPROACH_VISDR_DIR", str(_DEFAULT_VISDR_DIR))).expanduser().resolve()

_IMG_DIR = ensure_dir(_VISDR_DIR / "ImgData")
_EPISODES_DIR = ensure_dir(_VISDR_DIR / "TransitionEps")
_BATCH_DIR = ensure_dir(_VISDR_DIR / "TransitionBatch")


def save_images(episode, timestep, save_dir: Path = _IMG_DIR):
    """Save images with timestamps to a directory."""
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    for cam_id, img in current_images.items():
        img_path = save_dir / f"{cam_id}_image_ts{timestep}_ep{episode}.png"
        cv2.imwrite(str(img_path), img)
        print(f"Saved {cam_id} image for episode {episode}, timestep {timestep} to {img_path}")


def policy(obs, action_dim, time_step, env, noise=None):
    """
    Handcrafted expert for recording: moves in the sign direction of (goal - achieved).

    Notes:
    - We clip to [-1, 1] because the env expects normalized actions.
    - Noise is injected to avoid collecting a single deterministic trajectory.
    """
    obs_dict = obs
    current = obs_dict["achieved_goal"]
    goal = obs_dict["desired_goal"]
    dist_vec = np.array(goal - current, dtype=np.float32)

    # Compute direction: +1 if goal > current, -1 if goal < current
    direction = np.where(dist_vec > 0, 1.0, -1.0).astype(np.float32)

    # Scale action by env.step_size with large multiplier
    action = direction * env.step_size * 10000
    action = np.clip(action, -1, 1)

    if time_step % 1 == 0:
        noise = np.random.uniform(-0.1, 0.1, size=action.shape)
    if noise is not None:
        action = np.clip(action + noise, -1, 1)

    if np.linalg.norm(dist_vec[0:3]) > 0.1:
        action[-1] = 0.0

    return action, noise


seed = 60
seed_everything(seed)

max_episode_steps = 500

trans_step = 1.0e-3
angle_step = np.deg2rad(3)
jaw_step = 0.3
step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step], dtype=np.float32)

threshold = [3, np.deg2rad(30)]

step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step], dtype=np.float32)

threshold_expert = [0.1, np.deg2rad(60)]
SRC_approach = resolve_src_env(args.task_name)
gym.envs.register(id="SAC_HER_sparse", entry_point=SRC_approach, max_episode_steps=max_episode_steps)
env = gym.make(
    "SAC_HER_sparse",
    render_mode=None,
    reward_type="dense",
    seed=seed,
    threshold=threshold_expert,
    max_episode_steps=max_episode_steps,
    step_size=step_size,
    stepDR=randomize_step_size,
)

num_episodes = 50
episode_transitions = []
action_dim = 7
success = 0
max_action = [float("-inf")] * action_dim
min_action = [float("inf")] * action_dim
average_time_step = 0

visualDRCallback = DomainRandomizationCallback(env=env.unwrapped, randomization_args="0,0,1,1,1")

env.reset()

visualDRCallback.start_thread()

base_directory = str(_EPISODES_DIR)


def save_transitions_episode(transitions, episode_index, base_directory):
    # Ensure the directory exists
    filename = f"episode_{episode_index}.pkl"
    filepath = os.path.join(base_directory, filename)
    # create directory if it doesn't exist
    os.makedirs(base_directory, exist_ok=True)

    # Write the batch of transitions to a single pickle file
    with open(filepath, "wb") as file:
        pickle.dump(transitions, file)

    print(f"Saved episode {episode_index} to {filepath}")


episode = 0
while episode < num_episodes:
    obs, _ = env.reset()
    if randomize_env:
        print(f"Randomized environment for episode {episode}")
    time.sleep(0.5)
    noise = None
    episode_transitions = []

    for timestep in range(max_episode_steps):
        # If step limit exceeded, clean up and restart immediately.
        # This prevents “bad” episodes from polluting the dataset.
        if timestep > 100:
            print(f"Episode {episode} exceeded 100 steps at timestep {timestep}, deleting saved data and retrying...")
            # Delete episode pickle file (if any)
            episode_pkl = os.path.join(base_directory, f"episode_{episode}.pkl")
            if os.path.exists(episode_pkl):
                os.remove(episode_pkl)
            # Delete images for this episode
            img_dir = str(_IMG_DIR)
            for cam_id in camera_topics.keys():
                for ts in range(timestep + 1):
                    img_path = os.path.join(img_dir, f"{cam_id}_image_ts{ts}_ep{episode}.png")
                    if os.path.exists(img_path):
                        os.remove(img_path)
            # Restart episode (do not increment episode)
            break

        wait_for_images()
        save_images(episode, timestep)

        action, noise = policy(obs, action_dim, timestep, env.unwrapped, noise)
        print(action)
        next_obs, reward, done, _, info = env.step(action)

        time.sleep(0.01)

        transition = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": np.array(reward, dtype=np.float32),
            "done": np.array([done], dtype=np.float32),
            "info": info,
            "images": {cam_id: img for cam_id, img in current_images.items()},
        }
        episode_transitions.append(transition)

        obs = next_obs
        if done:
            print(timestep)
            average_time_step += timestep
            success += 1
            save_transitions_episode(episode_transitions, episode, base_directory)
            episode += 1
            break

average_time_step /= num_episodes
print(f"success rate is {success/num_episodes}")  # Make sure the success rate is 1 !
print(f"max action range is {max_action}")
print(f"min action range is {min_action}")
print(f"average time step is {average_time_step}")

episode_transitions[0]["images"].keys()

import os
import pickle


def save_transitions_batch(transitions, batch_index, base_directory):
    # Ensure the directory exists
    filename = f"batch_{batch_index}.pkl"
    filepath = os.path.join(base_directory, filename)

    # Write the batch of transitions to a single pickle file
    with open(filepath, "wb") as file:
        pickle.dump(transitions, file)

    print(f"Saved batch {batch_index} to {filepath}")


batch_size = 50
base_directory = str(_BATCH_DIR)

os.makedirs(base_directory, exist_ok=True)
current_batch = []
batch_num = 0
len_idx = len(episode_transitions)
for timestep, transition in enumerate(episode_transitions):
    current_batch.append(transition)
    if len(current_batch) >= batch_size or timestep >= len_idx - 1:
        batch_num += 1
        save_transitions_batch(current_batch, batch_num, base_directory)
        current_batch = []

filename = str(_VISDR_DIR / ("Expert_" + str(num_episodes) + ".pkl"))

# 使用 'wb' 模式打开文件以写入二进制数据
with open(filename, "wb") as file:
    # 使用 pickle.dump() 将对象序列化并保存到文件
    pickle.dump(episode_transitions, file)

data_dict = {
    "step_size": step_size,
    "threshold": threshold,
    "max_timestep": 3 * average_time_step,
}
pickle_file_path = str(_VISDR_DIR / "img_env_info.pkl")
with open(pickle_file_path, "wb") as file:
    pickle.dump(data_dict, file)

# Merge all episode_*.pkl files in the directory into one big list (no glob needed)
episodes_dir = str(_EPISODES_DIR)
merged_pickle_path = str(_VISDR_DIR / "all_episodes_merged.pkl")
all_transitions = []
episode_files = [f for f in os.listdir(episodes_dir) if f.startswith("episode_") and f.endswith(".pkl")]
episode_files = sorted(episode_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
for ep_file in episode_files:
    ep_path = os.path.join(episodes_dir, ep_file)
    with open(ep_path, "rb") as f:
        transitions = pickle.load(f)
        all_transitions.extend(transitions)
with open(merged_pickle_path, "wb") as f:
    pickle.dump(all_transitions, f)
print(f"Merged {len(episode_files)} episodes into {merged_pickle_path} with {len(all_transitions)} transitions.")

