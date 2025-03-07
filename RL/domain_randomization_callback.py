from stable_baselines3.common.callbacks import BaseCallback
import subprocess
import os
import time

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, model, ambf, verbose=0):
        super().__init__(verbose)
        self.model = model
        self.env = model.env
        self.ambf = ambf
        self.launch_file_path = os.path.expanduser('~/surgical_robotics_challenge/launch.yaml')
        self.ambf_simulator = os.path.expanduser("~/ambf/bin/lin-x86_64/ambf_simulator")
        

    def _on_rollout_start(self):
        self.ambf.terminate()
        time.sleep(5)
        print("Reinitializing AMBF (Rollout Start)")
        self.ambf = self.start_AMBF_env(self.launch_file_path)

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]):
                print("Restarting AMBF (Step complete)")
                self.ambf.terminate()
                print("AMBF terminated")
                time.sleep(5)
                
                # TO BE IMPLEMENTED
                # self.launch_file_path = self.randomize_environment() 
                
                self.ambf = self.start_AMBF_env(self.launch_file_path)
        return True

    def randomize_environment(self):
        """ Modify the environment parameters dynamically here """
        # TO BE IMPLEMENTED
        
        
    def start_AMBF_env(self, launch_file_path):
        command = [
            self.ambf_simulator,
            '--launch_file', launch_file_path,
            '-l', '0,1,3,4,13,14',
            '-p', '200',
            '-t', '1',
            '--override_max_comm_freq', '120',
            '--override_min_comm_freq', '120'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(30)
        self.env.reset()
        print("AMBF started")
        return process



