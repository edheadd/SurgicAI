from stable_baselines3.common.callbacks import BaseCallback
from Domain_randomization.Create_domain import DomainRandomization
import subprocess
import os
import time

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, model, ambf, headless_mode, randomization_params, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.env = model.env
        self.ambf = ambf
        self.headless_mode = headless_mode
        self.randomization_params = randomization_params
        self.randomizer = DomainRandomization(randomization_params)
        self.ambf_simulator = os.path.expanduser("~/ambf/bin/lin-x86_64/ambf_simulator")        

    def _on_rollout_start(self):
        self.ambf.terminate()
        time.sleep(5)
        print("Reinitializing AMBF (Rollout Start)")
        
        launch_file_path = self.get_randomized_path() 
        self.ambf = self.start_AMBF_env(launch_file_path)

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]):
                print("Restarting AMBF (Step complete)")
                self.ambf.terminate()
                print("AMBF terminated")
                time.sleep(5)
                
                launch_file_path = self.get_randomized_path() 
                
                self.ambf = self.start_AMBF_env(launch_file_path)
        return True
    
    def get_randomized_path(self):
        return self.randomizer.randomize_environment()
        
    def start_AMBF_env(self, launch_file_path):
        command = [
            self.ambf_simulator,
            '--launch_file', launch_file_path,
            '-l', '0,1,2,3,4,5',
            '-p', '200',
            '-t', '1',
            '--override_max_comm_freq', '120',
            '--override_min_comm_freq', '120'
        ]
        if self.headless_mode:
            command.append('-g')
            command.append('0')
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(25)
        self.env.reset()
        print("AMBF started")
        return process



