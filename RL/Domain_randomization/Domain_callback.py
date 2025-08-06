from stable_baselines3.common.callbacks import BaseCallback
from world_randomization_msgs.msg import Randomization
from Domain_randomization.randomization_gui import GUI
import rospy

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_args, env, verbose=0):
        super().__init__(verbose)
                        
        self.randomization_params = [True if x == "1" else False for x in randomization_args.split(",")]
        self.env = env
        self.name_dict = {
            "gravity": "Gravity vector in the world frame.",
            "friction": "Friction of the objects in the scene.",
            "light_num": "Number of lights in the scene.",
            "light_color": "Color of the lights in the scene.",
            "light_attenuation": "Attenuation of the lights in the scene."
        }

        self.randomization_pub = rospy.Publisher('/ambf/env/world_randomization/randomization', Randomization, queue_size=1)


    def _on_rollout_start(self):
        self.randomize() 

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]):
                self.randomize()           
        return True
    
    def randomize(self):
        msg = Randomization()
        msg.gravity = self.randomization_params[0]
        msg.friction = self.randomization_params[1]
        msg.light_num = self.randomization_params[2]
        msg.light_color = self.randomization_params[3]
        msg.light_attenuation = self.randomization_params[4]
        self.randomization_pub.publish(msg)

    def start_gui(self, app):
            self.gui_window = GUI(self)
            self.gui_window.show()
        
    def update_randomization_params(self, idx, immediate_randomization):
        
        self.randomization_params[idx] = not self.randomization_params[idx]
        if immediate_randomization:
            self.randomize()
    
    def reset_env(self):
        self.env.unwrapped.reset()
        self.randomize()