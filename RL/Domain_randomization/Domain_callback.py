from stable_baselines3.common.callbacks import BaseCallback

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, model, randomization_params, verbose=0):
        super().__init__(verbose)
        self.env = model.env
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        self.camera_randomization = self.randomization_params[0]
        self.light_randomization = self.randomization_params[1]
        self.psm_randomization = self.randomization_params[2]

    def _on_rollout_start(self):
        print("Randomizing AMBF (Rollout start)")
        self.randomize() 

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]):
                print("Randomizing AMBF (Step complete)")
                self.randomize()           
        return True
    
    def randomize(self):
        return;



