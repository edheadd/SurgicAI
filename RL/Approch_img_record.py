import sys
path_to_add = '/home/jin/SRC-gym/gym-env/Hierachical_Learning_v2'
sys.path.append(path_to_add)

import gymnasium as gym
from Approach_env import SRC_approach
import numpy as np
from stable_baselines3.common.utils import set_random_seed
import time
import pickle
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rospy
import os
import cv2
from Domain_randomization.Domain_callback import DomainRandomizationCallback



randomize_env = False
randomize_step_size = False

current_images = {}
image_received = {}
bridge = CvBridge()

def image_callback(msg, camera_id):
    """Callback to process and save images from different cameras."""
    global current_images, image_received
    try:
        current_images[camera_id] = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_received[camera_id] = True
    except Exception as e:
        rospy.logerr(f"Failed to convert image from {camera_id}: {e}")

# Subscribe to front and back camera images
# camera_topics = {
#     'front': '/ambf/env/cameras/cameraL/ImageData',
#     'back': '/ambf/env/cameras/normal_camera/ImageData'
# }

camera_topics = {
    'front': '/ambf/env/cameras/cameraL/ImageData'
}

for cam_id, topic in camera_topics.items():
    rospy.Subscriber(topic, Image, image_callback, callback_args=(cam_id))
    image_received[cam_id] = False

def wait_for_images():
    """Wait for all cameras to have received an image."""
    rate = rospy.Rate(100)
    while not all(image_received.values()) and not rospy.is_shutdown():
        rate.sleep()
    for key in image_received:
        image_received[key] = False

def save_images(episode,timestep, save_dir=f'/home/exie3/SurgicAI/SurgicAI_Img_Data/Approach/SingeCam/visDR_{randomize_env}_stepDR_{randomize_step_size}/ImgData/'):
    """Save images with timestamps to a directory."""
    for cam_id, img in current_images.items():
        img_path = os.path.join(save_dir, f"{cam_id}_image_ts{timestep}_ep{episode}.png")
        cv2.imwrite(img_path, img)
        rospy.loginfo(f"Saved {img_path}")





def policy(obs,action_dim,time_step,env,noise=None):
    obs_dict = obs
    current = obs_dict['achieved_goal']
    goal = obs_dict['desired_goal']
    dist_vec = np.array(goal-current,dtype = np.float32)
    
    # Compute direction: +1 if goal > current, -1 if goal < current
    direction = np.where(dist_vec>0,1.0,-1.0).astype(np.float32)
    
    # Scale action by env.step_size with large multiplier
    action = direction * env.step_size * 100000
    action = np.clip(action, -1, 1)
    
    if time_step%1 == 0:
        noise = np.random.uniform(-0.1, 0.1, size=action.shape)
    if noise is not None:
        action = np.clip(action + noise, -1, 1)

    if np.linalg.norm(dist_vec[0:3])>0.1:
        action[-1] = 0.0

    return action,noise


seed = 60
set_random_seed(seed)

max_episode_steps=500

trans_step = 1.0e-3
angle_step = np.deg2rad(3)
jaw_step = 0.05
step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step, jaw_step], dtype=np.float32)

threshold = [3,np.deg2rad(30)]

step_size = np.array([trans_step,trans_step,trans_step,angle_step,angle_step,angle_step,jaw_step],dtype=np.float32) 


threshold_expert = [0.1,np.deg2rad(5)] 
gym.envs.register(id="SAC_HER_sparse", entry_point=SRC_approach, max_episode_steps=max_episode_steps)
env = gym.make("SAC_HER_sparse", render_mode=None,reward_type = "dense",seed = seed, threshold = threshold_expert,max_episode_steps=max_episode_steps,step_size=step_size,stepDR=randomize_step_size)


num_episodes = 50
episode_transitions = []
action_dim = 7
success = 0
max_action = [float('-inf')] * action_dim
min_action = [float('inf')] * action_dim
average_time_step = 0


visualDRCallback = DomainRandomizationCallback(env=env.unwrapped, randomization_args="1,1,1,1,1")

env.reset()



base_directory = f"/home/exie3/SurgicAI/SurgicAI_Img_Data/Approach/SingeCam/visDR_{randomize_env}_stepDR_{randomize_step_size}/TransitionEps/"

def save_transitions_episode(transitions, episode_index, base_directory):
    # Ensure the directory exists
    filename = f"episode_{episode_index}.pkl"
    filepath = os.path.join(base_directory, filename)
    # create directory if it doesn't exist
    os.makedirs(base_directory, exist_ok=True)

    # Write the batch of transitions to a single pickle file
    with open(filepath, 'wb') as file:
        pickle.dump(transitions, file)

    print(f"Saved episode {episode_index} to {filepath}")



episode = 0
while episode < num_episodes:
    obs,_ = env.reset()
    if randomize_env:
        visualDRCallback.randomize()
        print(f"Randomized environment for episode {episode}")
    time.sleep(0.5)
    noise = None
    episode_transitions = []

    for timestep in range(max_episode_steps):
        wait_for_images()
        save_images(episode,timestep)

        action,noise = policy(obs,action_dim,timestep,env.unwrapped,noise)
        next_obs, reward, done, _, info = env.step(action)
        
        time.sleep(0.01)

        transition = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": np.array([reward], dtype=np.float32),
            "done": np.array([done], dtype=np.float32),
            "info": info,
            "images": {cam_id: img for cam_id, img in current_images.items()}
        }
        episode_transitions.append(transition)
        
        obs = next_obs
        if done:
            print(timestep)

            if timestep > 200:
                print(f"Episode {episode} took too long ({timestep} steps), retrying...")
                break
            
            average_time_step += timestep
            success+=1
            save_transitions_episode(episode_transitions, episode, base_directory)
            episode += 1
            break




average_time_step /= num_episodes
print(f"success rate is {success/num_episodes}")    # Make sure the success rate is 1 !    
print(f"max action range is {max_action}")
print(f"min action range is {min_action}")
print(f"average time step is {average_time_step}")



episode_transitions[0]['images'].keys()




import os
import pickle

def save_transitions_batch(transitions, batch_index, base_directory):
    # Ensure the directory exists
    filename = f"batch_{batch_index}.pkl"
    filepath = os.path.join(base_directory, filename)
    
    # Write the batch of transitions to a single pickle file
    with open(filepath, 'wb') as file:
        pickle.dump(transitions, file)

    print(f"Saved batch {batch_index} to {filepath}")

batch_size = 50
# base_directory = "/home/jin/migoogledrive/SRC_img_data/Approach/Multi_view"
base_directory = f"/home/exie3/SurgicAI/SurgicAI_Img_Data/Approach/SingeCam/visDR_{randomize_env}_stepDR_{randomize_step_size}/TransitionBatch/"

os.makedirs(base_directory, exist_ok=True)
current_batch = []
batch_num = 0
len_idx = len(episode_transitions)
for timestep, transition in enumerate(episode_transitions):
    current_batch.append(transition)
    if len(current_batch) >= batch_size or timestep>=len_idx-1:
        batch_num += 1
        save_transitions_batch(current_batch, batch_num, base_directory)
        current_batch = []



filename = f"/home/exie3/SurgicAI/SurgicAI_Img_Data/Approach/SingeCam/visDR_{randomize_env}_stepDR_{randomize_step_size}/Expert_"+str(num_episodes)+".pkl"


# 使用 'wb' 模式打开文件以写入二进制数据
with open(filename, 'wb') as file:
    # 使用 pickle.dump() 将对象序列化并保存到文件
    pickle.dump(episode_transitions, file)


data_dict = {
    "step_size": step_size,
    "threshold": threshold,
    "max_timestep": 3*average_time_step
}
pickle_file_path = f"/home/exie3/SurgicAI/SurgicAI_Img_Data/Approach/SingeCam/visDR_{randomize_env}_stepDR_{randomize_step_size}/img_env_info.pkl"
with open(pickle_file_path, 'wb') as file:
    pickle.dump(data_dict, file)