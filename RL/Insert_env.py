import numpy as np
from subtask_env import SRC_subtask


class SRC_insert(SRC_subtask):
    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=None):

        # Define action and observation space
        super(SRC_insert, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 2
        self.action_lims_low = np.array([-0.1, -0.1, -0.25, np.deg2rad(-270), np.deg2rad(-80), np.deg2rad(-260), 0],dtype=np.float32)
        self.action_lims_high = np.array([0.1, 0.1, 0.05, np.deg2rad(-90), np.deg2rad(80), np.deg2rad(260), 1],dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.set_seed(seed)

        self.scene_manager.env_reset()

        self.scene_manager.approach_and_grasp()
        self.scene_manager.place_at_entry()

        self.scene_manager.psm_goal_list[self.psm_idx-1]  = self.scene_manager.entry_obs

        self.goal_obs = self.scene_manager.insert_goal_evaluator(100,[0.002,0,0],self.psm_idx)
        current = self.scene_manager.psm_goal_list[self.psm_idx-1]
        obs_array = np.concatenate((current ,self.goal_obs,self.goal_obs-current ), dtype=np.float32)

        self.init_obs_dict = {"observation":obs_array,"achieved_goal":current ,"desired_goal":self.goal_obs}
        self.obs = self.gym_manager.normalize_observation(self.init_obs_dict)
        
        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info

    # Overridden step function
    def step(self, action):
        action[-1]=0
        self.timestep += 1
        self.action = action
        current = self.scene_manager.psm_goal_list[self.psm_idx-1]
        action_step = action*self.step_size
        self.scene_manager.world_handle.update()


        self.scene_manager.psm_goal_list[self.psm_idx-1] = np.clip(current+action_step, self.action_lims_low[0:7], self.action_lims_high[0:7])
        self.scene_manager.psm_step(self.scene_manager.psm_goal_list[self.psm_idx-1] ,self.psm_idx)
        return self.gym_manager.update_observation(self.scene_manager.psm_goal_list[self.psm_idx-1])

