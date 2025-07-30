import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import re

from PyKDL import Frame, Rotation, Vector
from surgical_robotics_challenge.kinematics.psmKinematics import *
from subtask_env import SRC_subtask


class SRC_place(SRC_subtask):

    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=None):
        # Define action and observation space
        super(SRC_place, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 2

    def reset(self, **kwargs):
        """ Reset the state of the environment to an initial state """
        self.psm2.actuators[0].deactuate()
        self.needle_randomization()
        time.sleep(1.0)
        self.psm_goal_list[self.psm_idx-1] = self.needle_random_grasping_evaluator(0.010)
        self.psm_step(self.psm_goal_list[self.psm_idx-1],2)
        self.world_handle.reset()
        self.Camera_view_reset()
        time.sleep(1.0)

        self.psm2.actuators[0].actuate("Needle")
        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])
        self.entry_obs = self.entry_goal_evaluator(self.psm_idx)
        self.goal_obs = self.entry_obs
        self.init_obs_array = np.concatenate((self.psm_goal_list[self.psm_idx-1],self.goal_obs,self.goal_obs-self.psm_goal_list[self.psm_idx-1]),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":self.psm_goal_list[self.psm_idx-1],"desired_goal":self.goal_obs}
        
        self.obs = self.normalize_observation(self.init_obs_dict)

        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info

    def entry_goal_evaluator(self,idx = 2):
        self.entry_w = self.scene.entry1_measured_cp()
        entry_pos = self.psm_list[idx-1].get_T_w_b()*self.entry_w
        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        T_tip_base = self.needle_kin.get_tip_pose()
        T_gripper_base = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        T_gripper_tip = T_tip_base.Inverse()*T_gripper_base

        T_insert = entry_pos
        T_insert.M *= rotation_entry
        T_insert = T_insert*T_gripper_tip
        array_insert = self.Frame2Vec(T_insert)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def needle_random_grasping_evaluator(self,lift_height):
        self.random_degree = np.random.uniform(10, 50)
        self.grasping_pos = self.needle_kin.get_random_grasp_point()
        needle_rot = self.grasping_pos.M
        needle_trans_lift = Vector(self.grasping_pos.p.x(),self.grasping_pos.p.y(),self.grasping_pos.p.z()+lift_height)
        needle_goal_lift = Frame(needle_rot, needle_trans_lift)

        T_calibrate = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_calibrate[:3, :3]

        rotation_calibrate = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        needle_goal_lift.M = needle_goal_lift.M * rotation_calibrate # To be tested
        
        psm_goal_lift = self.psm2.get_T_w_b()*needle_goal_lift

        T_goal = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_goal[:3, :3]

        rotation = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        psm_goal_lift.M = psm_goal_lift.M*rotation

        array_goal_base = self.Frame2Vec(psm_goal_lift)
        array_goal_base = np.append(array_goal_base,0.0)
        return array_goal_base
    

    def mid_goal_evaluator(self,height=0.025):
        mid_goal_rot = self.grasping_pos.M
        mid_goal_trans_lift = Vector(self.entry_w.p.x()-0.035,self.entry_w.p.y(),self.grasping_pos.p.z()+height)
        mid_goal_lift = Frame(mid_goal_rot, mid_goal_trans_lift)

        T_calibrate = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_calibrate[:3, :3]

        rotation_calibrate = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        mid_goal_lift.M = mid_goal_lift.M * rotation_calibrate # To be tested
        
        psm_goal_lift = self.psm2.get_T_w_b()*mid_goal_lift

        T_goal = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_goal[:3, :3]

        rotation = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        psm_goal_lift.M = psm_goal_lift.M*rotation

        array_goal_base = self.Frame2Vec(psm_goal_lift)
        array_goal_base = np.append(array_goal_base,0.0)
        return array_goal_base
