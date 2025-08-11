import numpy as np
from gymnasium import spaces

class GymManager:
    """Manages observation space, action space, and reward calculations"""
    
    def __init__(self, env, reward_type="sparse", threshold=[0.5, np.deg2rad(30)]):
        self.env = env
        self.reward_type = reward_type
        
        self._setup_spaces()
        
    def _setup_spaces(self):
        """Initialize observation and action spaces"""
        self.env.action_space = spaces.Box(
            low=np.array([-1, -1, -1, -1, -1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            shape=(7,), dtype=np.float32
        )
        
        self.env.observation_space = spaces.Dict({
            "observation": spaces.Box(
                low=np.array([-np.inf] * 21, dtype=np.float32),
                high=np.array([np.inf] * 21, dtype=np.float32),
                shape=(21,), dtype=np.float32),
            "achieved_goal": spaces.Box(
                low=np.array([-np.inf]*7, dtype=np.float32),
                high=np.array([np.inf]*7, dtype=np.float32),
                shape=(7,), dtype=np.float32),
            "desired_goal": spaces.Box(
                low=np.array([-np.inf]*7, dtype=np.float32),
                high=np.array([np.inf]*7, dtype=np.float32),
                shape=(7,), dtype=np.float32)              
        })
        
    def update_observation(self, current):
        """Update the environment observation"""
        goal_obs = self.env.goal_obs
        current = np.array(current, dtype=np.float32)
        goal_obs = np.array(goal_obs, dtype=np.float32)
        
        obs_array = np.concatenate((current, goal_obs, goal_obs-current), dtype=np.float32)
        obs_dict = {
            "observation": obs_array,
            "achieved_goal": current,
            "desired_goal": goal_obs
        }
        
        self.env.obs = self.normalize_observation(obs_dict)
        self.reward = self.compute_reward(self.env.obs["achieved_goal"], self.env.obs["desired_goal"])
        self.terminate = self.env.criteria()
        self.truncate = self.env.timestep >= self.env.max_timestep
        self.env.info = {"is_success": self.terminate}

        return self.env.obs, self.reward, self.terminate, self.truncate, self.env.info

    def normalize_observation(self, observation_dict):
        """Scale observations - translation into 'cm', orientation into 'rad'"""
        observation = observation_dict["observation"]
        achieved_goal = observation_dict["achieved_goal"]
        desired_goal = observation_dict["desired_goal"]

        multiplier = np.ones(21, dtype=np.float32)
        indices_to_multiply_100 = [0, 1, 2, 7, 8, 9, 14, 15, 16]
        multiplier[indices_to_multiply_100] = 100
        observation_dict["observation"] = np.array(observation * multiplier, dtype=np.float32)

        multiplier2 = np.array([100, 100, 100, 1, 1, 1, 1])
        observation_dict["achieved_goal"] = np.array(achieved_goal * multiplier2, dtype=np.float32)
        observation_dict["desired_goal"] = np.array(desired_goal * multiplier2, dtype=np.float32)
        
        return observation_dict
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Calculate reward based on current state"""
        goal_len = 7
        achieved_goal = np.array(achieved_goal).reshape(-1, goal_len)
        desired_goal = np.array(desired_goal).reshape(-1, goal_len)
        
        distances_trans = np.linalg.norm(achieved_goal[:, 0:3] - desired_goal[:, 0:3], axis=1)
        distances_angle = np.linalg.norm(achieved_goal[:, 3:6] - desired_goal[:, 3:6], axis=1)
        
        if self.reward_type == "dense":
            rewards = -(distances_trans/100 + distances_angle/10)
        else:  # sparse
            rewards = np.where(
                (distances_trans <= self.env.threshold_trans) & (distances_angle <= self.env.threshold_angle),
                0, -1
            )
        return float(rewards[0])
