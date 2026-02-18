import time
import gymnasium as gym
import numpy as np
from utils import scene_manager
from ros_abstraction_layer import ral
from utils.gym_manager import GymManager
from utils.scene_manager import SceneManager
from utils.utils import convert_mat_to_frame, convert_mat_to_vector

class SRC_subtask(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human'],"reward_type":['dense']}

    def __init__(self, seed=None, render_mode=None, reward_type="sparse", threshold=[0.5, np.deg2rad(30)], max_episode_step=200, step_size=None, stepDR=False):

        # Define action and observation space
        super(SRC_subtask, self).__init__()
        self.random_range = np.array([0.0003, 0.0002, np.pi / 6], dtype=np.float32)
        self.max_timestep = max_episode_step
        print(f"max episode length is {self.max_timestep}")
        self.base_step_size = step_size
        self.step_size = step_size
        print(f"step size is {self.step_size}")
        self.stepDR = stepDR
        if self.stepDR:
            print("State-Space Domain Randomization is enabled")

        if seed is not None:
            self.seed = seed 
            print("Set random seed")
        else:
            self.seed = None  # No seed was provided

        print(f"reward type is {reward_type}")
        self.reward_type = reward_type
        self.threshold_trans = threshold[0]
        self.threshold_angle = threshold[1]
        print(f"Translation threshold: {self.threshold_trans}, angle threshold: {self.threshold_angle}")
        
        # init ral
        ral_instance = ral("src_subtask_env")
        time.sleep(0.5)  # Allow some time for RAL to initialize
        
        ral_instance.spin()  # Start RAL spinning to process callbacks
            
        self.gym_manager = GymManager(self, reward_type=reward_type, threshold=[threshold])
        self.scene_manager = SceneManager(self, ral_instance)
        
        
        self.goal_obs = None
        self.obs = None
        self.info = None
        self.timestep = 0
        self.psm_idx = None
        self.last_goal_rotation = [None, None]  # Cache last goal rotation to avoid Euler wrapping
               
        print("Initialized!!!")
        return
    
    def set_seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)

    def step(self, action):
        """
        Step function, defines the system dynamic and updates the observation
        """
        self.timestep += 1
        
        # Only update goal if action is non-zero
        if np.any(action != 0):
            # Get current measured pose and convert to vector once
            measured_pose = self.scene_manager.psm_list[self.psm_idx-1].measured_cp()
            current_obs = convert_mat_to_vector(measured_pose)
            current_jaw = self.scene_manager.psm_list[self.psm_idx-1].get_jaw_angle()
            goal_vector = np.append(current_obs, current_jaw)
            
            # On first action, cache the rotation; on subsequent actions, reuse it
            if self.last_goal_rotation[self.psm_idx-1] is None:
                self.last_goal_rotation[self.psm_idx-1] = goal_vector[3:6].copy()
            else:
                goal_vector[3:6] = self.last_goal_rotation[self.psm_idx-1]
            
            # Apply action directly to vector 
            action_step = action * self.step_size
            goal_vector = goal_vector + action_step
            
            self.scene_manager.psm_goal_list[self.psm_idx-1] = goal_vector
        
        # Step and update observation
        self.scene_manager.step()
        return self.gym_manager.update_observation(self.scene_manager.psm_goal_list[self.psm_idx-1])

    def render(self, mode='human', close=False):
        pass
            
    def update_difficulty(self, difficulty_settings):
        self.threshold_trans = difficulty_settings['trans_tolerance']
        self.threshold_angle = difficulty_settings['angle_tolerance']
        self.random_range = difficulty_settings['random_range']

