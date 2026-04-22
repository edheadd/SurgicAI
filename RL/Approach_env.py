# Training environment for approaching stage for (DDPG+HER)

import numpy as np
from RL.subtask_env import SRC_subtask


class SRC_approach(SRC_subtask):
    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=None, stepDR=None):

        # Define action and observation space
        super(SRC_approach, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size, stepDR)
        self.psm_idx = 2

    def reset(self, seed=None, **kwargs):
        """ Reset the state of the environment to an initial state """
        
        if seed is not None:
            self.set_seed(seed)

        self.scene_manager.env_reset()
        self.scene_manager.needle_randomization()
        
        self.min_angle = 5
        self.max_angle = 20
        self.grasp_angle = np.random.uniform(self.min_angle, self.max_angle)

        self.needle_obs = self.scene_manager.needle_goal_evaluator(0.007)
        self.goal_obs = self.needle_obs
        self.multigoal_obs = self.scene_manager.needle_multigoal_evaluator(lift_height=0.007,start_degree=self.min_angle,end_degree=self.max_angle)
        current_pos = self.scene_manager.psm_goal_list[self.psm_idx-1]
        self.init_obs_array = np.concatenate((current_pos,self.goal_obs,self.goal_obs-current_pos),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":current_pos,"desired_goal":self.goal_obs}

        self.obs = self.gym_manager.normalize_observation(self.init_obs_dict)

        # self.grasp_success = False
        # self.horizon = 0
        
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        
        return self.obs, self.info
    
    # Overridden step function
    def step(self, action):
        self.needle_obs = self.scene_manager.needle_goal_evaluator(lift_height=0.007,deg_angle=self.grasp_angle)
        self.multigoal_obs = self.scene_manager.needle_multigoal_evaluator(lift_height=0.007,start_degree=self.min_angle,end_degree=self.max_angle)
        self.goal_obs = self.needle_obs
        return super(SRC_approach, self).step(action)

    # Overridden criteria function
    def criteria(self):
        achieved_goal = self.obs["achieved_goal"]

        min_trans = np.Inf
        min_angle = np.Inf
        printed = False
        
        for idx, desired_goal in enumerate(self.multigoal_obs):
            desired_goal = desired_goal*np.array([100,100,100,1,1,1,1])
            
            distances_trans = np.linalg.norm(achieved_goal[:3] - desired_goal[:3])
            distances_angle = np.linalg.norm(achieved_goal[3:6] - desired_goal[3:6])
            
            if min_trans > distances_trans:
                min_trans = distances_trans
            if min_angle > distances_angle:
                min_angle = distances_angle

            # if self.timestep % 100 == 0 and not printed:
            #     print("distances_trans: ", distances_trans)
            #     print("distances_angle", distances_angle)
            #     print("jaw_angle", self.scene_manager.jaw_angle_list[self.psm_idx-1])
            #     printed = True

            if distances_trans <= self.threshold_trans and distances_angle <= self.threshold_angle: #and self.scene_manager.jaw_angle_list[self.psm_idx-1] <= 0.2:
                print(f"Matched degree is {self.min_angle + idx * (self.max_angle - self.min_angle) / len(self.multigoal_obs)}, distance_trans = {distances_trans}, distances_angle = {np.degrees(distances_angle)}")
                print("Attach the needle to the gripper")
                
                self.scene_manager.psm2.actuate("Needle")
                self.scene_manager.needle.release()

                return True

                # self.grasp_success = True
                # self.horizon = 20
        
        # if not self.horizon == 0:
        #     self.horizon = self.horizon-1
        #     if self.horizon == 0:
        #         return self.grasp_success
            
        self.min_trans = min_trans
        self.min_angle = min_angle    
        return False
    
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """Calculate reward based on current state"""
        goal_len = 7
        achieved_goal = np.array(achieved_goal).reshape(-1, goal_len)
        desired_goal = np.array(desired_goal).reshape(-1, goal_len)
        
        distances_trans = np.linalg.norm(achieved_goal[:, 0:3] - desired_goal[:, 0:3], axis=1)
        distances_angle = np.linalg.norm(achieved_goal[:, 3:6] - desired_goal[:, 3:6], axis=1)

        in_region = (distances_trans <= self.threshold_trans) & (distances_angle <= self.threshold_angle)
        grasp_zone_reward = np.where(in_region, 0.5, 0.0)

        
        if self.reward_type == "dense":
            rewards = -(distances_trans/100 + distances_angle/10) + grasp_zone_reward
        else:  # sparse
            rewards = np.where(
                (distances_trans <= self.env.threshold_trans) & (distances_angle <= self.env.threshold_angle),
                0, -1
            )
        return np.asarray(rewards, dtype=np.float32)

