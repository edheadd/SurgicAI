import time
import gymnasium as gym
import numpy as np
from utils import scene_manager
from ros_abstraction_layer import ral
from utils.gym_manager import GymManager
from utils.scene_manager import SceneManager
from utils.utils import convert_mat_to_frame, frame_to_vector, vector_to_frame

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
        self.ral_instance = ral("src_subtask_env")
        time.sleep(0.5)  # Allow some time for RAL to initialize
        
        self.ral_instance.spin()  # Start RAL spinning to process callbacks
            
        self.gym_manager = GymManager(self, reward_type=reward_type, threshold=[threshold])
        self.scene_manager = SceneManager(self, self.ral_instance)
        
        
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

    def _frame_to_vec6(self, frame):
        """Convert PyKDL.Frame to [x,y,z,roll,pitch,yaw]."""
        return frame_to_vector(frame)

    def _gripper_T_c(self, psm_idx):
        """Measured gripper pose expressed in camera frame."""
        psm = self.scene_manager.psm_list[psm_idx - 1]
        T_g_b = convert_mat_to_frame(psm.measured_cp())
        T_g_w = psm.get_T_b_w() * T_g_b
        T_w_c = self.scene_manager.ecm.get_T_w_c()
        return T_w_c * T_g_w

    def _goal_vec7_c_from_goal_vec7_b(self, goal_vec7_b, psm_idx):
        """Convert desired goal from PSM base frame to camera frame."""
        psm = self.scene_manager.psm_list[psm_idx - 1]
        T_goal_b = vector_to_frame(goal_vec7_b)
        T_goal_w = psm.get_T_b_w() * T_goal_b
        T_w_c = self.scene_manager.ecm.get_T_w_c()
        goal_vec6_c = self._frame_to_vec6(T_w_c * T_goal_w)
        return np.append(goal_vec6_c, float(np.asarray(goal_vec7_b)[6]))

    def _goal_vec7_b_from_goal_vec7_c(self, goal_vec7_c, psm_idx):
        """Convert desired goal from camera frame to PSM base frame (for commanding)."""
        psm = self.scene_manager.psm_list[psm_idx - 1]
        T_goal_c = vector_to_frame(goal_vec7_c)
        T_c_w = self.scene_manager.ecm.get_T_c_w()
        T_goal_w = T_c_w * T_goal_c
        T_goal_b = psm.get_T_w_b() * T_goal_w
        goal_vec6_b = self._frame_to_vec6(T_goal_b)
        return np.append(goal_vec6_b, float(np.asarray(goal_vec7_c)[6]))

    def step(self, action):
        """
        Step function, defines the system dynamic and updates the observation
        """
        self.timestep += 1
        
        # Only update goal if action is non-zero
        if np.any(action != 0):
            # Work in camera frame for RL: current measured gripper pose in camera coordinates
            T_g_c = self._gripper_T_c(self.psm_idx)
            current_obs_c = self._frame_to_vec6(T_g_c)
            current_jaw = self.scene_manager.psm_list[self.psm_idx - 1].get_jaw_angle()
            goal_vector_c = np.append(current_obs_c, current_jaw)
            
            # On first action, cache the rotation; on subsequent actions, reuse it
            if self.last_goal_rotation[self.psm_idx-1] is None:
                self.last_goal_rotation[self.psm_idx-1] = goal_vector_c[3:6].copy()
            else:
                goal_vector_c[3:6] = self.last_goal_rotation[self.psm_idx-1]
            
            # Apply action directly to vector 
            action_step = action * self.step_size
            goal_vector_c = goal_vector_c + action_step
            
            # Convert commanded goal back to PSM base frame (SceneManager expects base frame)
            goal_vector_b = self._goal_vec7_b_from_goal_vec7_c(goal_vector_c, self.psm_idx)
            self.scene_manager.psm_goal_list[self.psm_idx - 1] = goal_vector_b
        
        # Step and update observation
        self.scene_manager.step()

        # Emit observation in camera frame (both achieved and desired)
        achieved_vec7_c = np.append(self._frame_to_vec6(self._gripper_T_c(self.psm_idx)),
                                    float(self.scene_manager.psm_list[self.psm_idx - 1].get_jaw_angle()))

        # Gym reads desired_goal from env.goal_obs; keep it in camera frame for this env.
        if self.goal_obs is not None:
            self.goal_obs = self._goal_vec7_c_from_goal_vec7_b(self.goal_obs, self.psm_idx)

        return self.gym_manager.update_observation(achieved_vec7_c)

    def render(self, mode='human', close=False):
        pass
            
    def update_difficulty(self, difficulty_settings):
        self.threshold_trans = difficulty_settings['trans_tolerance']
        self.threshold_angle = difficulty_settings['angle_tolerance']
        self.random_range = difficulty_settings['random_range']


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
        return np.asarray(rewards[0])
    
    
    def criteria(self):
        """
        Decide whether success criteria (Distance is lower than a threshold) is met.
        """
        achieved_goal = self.env.obs["achieved_goal"]
        desired_goal = self.env.obs["desired_goal"]
        distances_trans = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
        print(f"Distance to goal - Translation: {distances_trans:.4f} cm, Rotation: {np.rad2deg(distances_angle):.2f} degrees")
        if (distances_trans<= self.env.threshold_trans) and (distances_angle <= self.env.threshold_angle):
            return True
        else:
            return False
