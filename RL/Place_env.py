import numpy as np
from subtask_env import SRC_subtask


class SRC_place(SRC_subtask):

    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=None):
        # Define action and observation space
        super(SRC_place, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 2

    def reset(self, seed=None, **kwargs):
        """ Reset the state of the environment to an initial state """
        
        if seed is not None:
            self.set_seed(seed)
        
        self.scene_manager.env_reset()
        self.scene_manager.needle_randomization()

        self.scene_manager.psm2.actuators[0].actuate("Needle")
        self.scene_manager.needle.needle.set_force([0.0,0.0,0.0])
        self.scene_manager.needle.needle.set_torque([0.0,0.0,0.0])
        self.entry_obs = self.scene_manager.place_entry_goal_evaluator(self.psm_idx)
        self.goal_obs = self.entry_obs
        current = self.scene_manager.psm_goal_list[self.psm_idx-1]
        self.init_obs_array = np.concatenate((current,self.goal_obs,self.goal_obs-current),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":current,"desired_goal":self.goal_obs}

        self.obs = self.gym_manager.normalize_observation(self.init_obs_dict)

        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info

