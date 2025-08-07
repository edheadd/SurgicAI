import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time

from PyKDL import Frame, Rotation, Vector
from surgical_robotics_challenge.kinematics.psmKinematics import *
from subtask_env import SRC_subtask

class SRC_pullout(SRC_subtask):

    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=np.array([0.0005, 0.0005, 0.0005, np.deg2rad(2), np.deg2rad(2), np.deg2rad(2), 0.05])):

        super(SRC_pullout, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
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

        self.approach_and_grasp()
        self.place_at_entry()
        self.insert_needle()
        self.regrasp_needle()
        
        self.goal_obs = self.handover_goal_evaluator(idx=1)
        self.psm_goal_list[self.psm_idx-1] = np.copy(self.regrasp_obs)
        obs_array = np.concatenate((self.psm1_goal,self.goal_obs,self.goal_obs-self.psm1_goal), dtype=np.float32)
        
        self.init_obs_dict = {"observation":obs_array,"achieved_goal":self.psm1_goal,"desired_goal":self.goal_obs}
        self.obs = self.normalize_observation(self.init_obs_dict) 

        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info
    