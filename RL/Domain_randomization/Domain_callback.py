from stable_baselines3.common.callbacks import BaseCallback
from Domain_randomization.Domain_randomization import DomainRandomization
import time

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, model, randomization_params, verbose=0):
        super().__init__(verbose)
        self.env = model.env
        self.randomizer = DomainRandomization(self.env, randomization_params)

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
        self.randomizer.randomize_environment()
        time.sleep(2)
        print("Randomized!")



