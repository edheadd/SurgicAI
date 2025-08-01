import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from PyKDL import Frame, Rotation, Vector
from surgical_robotics_challenge.kinematics.psmKinematics import *
from subtask_env import SRC_subtask

class SRC_regrasp(SRC_subtask):
    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=np.array([0.0005, 0.0005, 0.0005, np.deg2rad(2), np.deg2rad(2), np.deg2rad(2), 0.05])):

        super(SRC_regrasp, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 1


    def reset(self, seed = None,**kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.psm_goal_list[0] = np.copy(self.init_psm1)
        self.psm_goal_list[1] = np.copy(self.init_psm2)
        self.psm1.actuators[0].deactuate()
        self.psm2.actuators[0].deactuate()
        self.world_handle.reset()
        self.psm_step(self.psm_goal_list[0],1)
        self.psm_step(self.psm_goal_list[1],2)
        self.Camera_view_reset()
        time.sleep(0.5)
        self.world_handle.reset()
        time.sleep(0.5)

        # Approach and grasp the needle
        self.needle_obs = self.needle_random_grasping_evaluator(0.007)
        self.needle_obs = np.append(self.needle_obs,0.8)
        self.psm_step_move(self.needle_obs,2)
        time.sleep(0.8)
        self.needle_obs[-1] = 0.0
        self.psm_step(self.needle_obs,2)
        self.Camera_view_reset()
        time.sleep(0.3)
        self.psm2.actuators[0].actuate("Needle")
        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])

        # Place the needle at the entry
        self.entry_obs = self.entry_goal_evaluator(idx=2,dev_trans=[0,0,0.001],noise=False) # Close noise in this case
        self.psm_step_move(self.entry_obs,2,execute_time=0.5) 
        time.sleep(0.8)

        # Insert the needle
        self.insert_mid_obs =  self.entry_goal_evaluator(105,[0.001,0,0],-50)
        self.insert_obs = self.insert_goal_evaluator(90,[0.002,0,0])
        self.psm_step_move(self.insert_mid_obs,2,execute_time=0.3)
        time.sleep(0.6)
        self.psm_step_move(self.insert_obs,2,execute_time=0.4)
        self.psm_goal_list[1] = np.copy(self.insert_obs)
        time.sleep(0.8)

        self.goal_obs = self.needle_goal_evaluator(deg_angle=105,lift_height=0.005,psm_idx=1)

        self.init_obs_array = np.concatenate((self.psm_goal_list[self.psm_idx-1],self.goal_obs,self.goal_obs-self.psm_goal_list[self.psm_idx-1]),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":self.psm_goal_list[self.psm_idx-1],"desired_goal":self.goal_obs}
        
        self.obs = self.normalize_observation(self.init_obs_dict) 

        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0

        return self.obs, self.info
    
    def criteria(self):
        """
        Decide whether success criteria (Distance is lower than a threshold) is met.
        """
        achieved_goal = self.obs["achieved_goal"]
        desired_goal = self.obs["desired_goal"]
        distances_trans = np.linalg.norm(achieved_goal[0:3] - desired_goal[0:3])
        distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
        if (distances_trans<= self.threshold_trans) and (distances_angle <= self.threshold_angle and self.jaw_angle_list[self.psm_idx-1] <= 0.1):
            return True
        else:
            return False