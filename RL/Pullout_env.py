import numpy as np
from subtask_env import SRC_subtask

class SRC_pullout(SRC_subtask):

    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=np.array([0.0005, 0.0005, 0.0005, np.deg2rad(2), np.deg2rad(2), np.deg2rad(2), 0.05])):

        super(SRC_pullout, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 1

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.set_seed(seed)

        self.scene_manager.env_reset()

        self.scene_manager.approach_and_grasp()
        self.scene_manager.place_at_entry()
        self.scene_manager.insert_needle()
        self.scene_manager.regrasp_needle()
        
        self.goal_obs = self.scene_manager.handover_goal_evaluator(idx=1)
        self.scene_manager.psm_goal_list[self.psm_idx-1] = np.copy(self.scene_manager.regrasp_obs)
        current = self.scene_manager.init_psm1
        obs_array = np.concatenate((current,self.goal_obs,self.goal_obs-current), dtype=np.float32)

        self.init_obs_dict = {"observation":obs_array,"achieved_goal":current,"desired_goal":self.goal_obs}
        self.obs = self.gym_manager.normalize_observation(self.init_obs_dict)

        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0
        return self.obs, self.info
    
