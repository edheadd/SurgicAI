import gymnasium as gym
import numpy as np
from utils.gym_manager import GymManager
from utils.scene_manager import SceneManager

class SRC_subtask(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render_modes': ['human'],"reward_type":['dense']}

    def __init__(self, seed=None, render_mode=None, reward_type="sparse", threshold=[0.5, np.deg2rad(30)], max_episode_step=200, step_size=None):

        # Define action and observation space
        super(SRC_subtask, self).__init__()
        self.random_range = np.array([0.0003, 0.002, np.pi / 6], dtype=np.float32)
        self.max_timestep = max_episode_step
        print(f"max episode length is {self.max_timestep}")
        self.step_size = step_size
        print(f"step size is {self.step_size}")

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
            
        self.gym_manager = GymManager(self, reward_type=reward_type, threshold=[threshold])
        self.scene_manager = SceneManager(self)
        
        self.goal_obs = None
        self.obs = None
        self.info = None
        self.timestep = 0
        self.psm_idx = None
               
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
        action_step = action*self.step_size
        self.scene_manager.psm_goal_list[self.psm_idx-1] = self.scene_manager.psm_goal_list[self.psm_idx-1]+action_step
        self.scene_manager.step()

        return self.gym_manager.update_observation(self.scene_manager.psm_goal_list[self.psm_idx-1])

    def render(self, mode='human', close=False):
        pass
            
    def update_difficulty(self, difficulty_settings):
        self.threshold_trans = difficulty_settings['trans_tolerance']
        self.threshold_angle = difficulty_settings['angle_tolerance']
        self.random_range = difficulty_settings['random_range']

    def criteria(self):
        """
        Decide whether success criteria (Distance is lower than a threshold) is met.
        """
        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        distances_trans = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
        if (distances_trans<= self.threshold_trans) and (distances_angle <= self.threshold_angle):
            return True
        else:
            return False

