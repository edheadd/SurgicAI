import numpy as np
from subtask_env import SRC_subtask

class SRC_regrasp(SRC_subtask):
    def __init__(self,seed=None,render_mode = None,reward_type = "sparse",threshold = [0.5,np.deg2rad(30)],max_episode_step=200, step_size=np.array([0.0005, 0.0005, 0.0005, np.deg2rad(2), np.deg2rad(2), np.deg2rad(2), 0.05])):

        super(SRC_regrasp, self).__init__(seed,render_mode,reward_type,threshold,max_episode_step, step_size)
        self.psm_idx = 1


    def reset(self, seed=None, **kwargs):
        if seed is not None:
            self.set_seed(seed)
            
        self.src_manager.env_reset()

        self.src_manager.approach_and_grasp()
        self.src_manager.place_at_entry()
        self.src_manager.insert_needle()

        self.goal_obs = self.src_manager.needle_goal_evaluator(deg_angle=105,lift_height=0.005,psm_idx=1)

        current = self.src_manager.psm_goal_list[self.psm_idx-1]
        self.init_obs_array = np.concatenate((current,self.goal_obs,self.goal_obs-current),dtype=np.float32)
        self.init_obs_dict = {"observation":self.init_obs_array,"achieved_goal":current,"desired_goal":self.goal_obs}

        self.obs = self.gym_manager.normalize_observation(self.init_obs_dict)

        self.info = {"is_success":False}
        print("reset!!!")
        self.timestep = 0

        return self.obs, self.info