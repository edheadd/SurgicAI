from stable_baselines3.common.callbacks import BaseCallback
import rospy
from ambf_msgs.msg import WorldCmd
import random

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_params, verbose=0):
        super().__init__(verbose)
        
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        self.enable_gravity_randomization = self.randomization_params[0]
        self.enable_light_color_randomization = self.randomization_params[1]
        
        self.pub = rospy.Publisher('/WorldRandomization/Commands/Command', WorldCmd, queue_size=10)
        
        self.prevMsg = WorldCmd()


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
        
        msg = WorldCmd()
        
        msg = self.randomize_gravity(msg, self.enable_gravity_randomization)
        msg = self.randomize_light_color(msg, self.enable_light_color_randomization)
                                
        self.pub.publish(msg)
        self.prevMsg = msg
        print("Published randomization command")
        
        
    def randomize_gravity(self, msg, randomize):
        msg.randomize_gravity = randomize
        msg.gravity.x = 0.0
        msg.gravity.y = 0.0

        if randomize:
            msg.gravity.z = random.choice([-9.81, 0.0])

        else:
            msg.gravity.z = -9.81

        return msg
    
    def randomize_light_color(self, msg, randomize):
        msg.randomize_light_color = randomize
        if randomize:
            msg.rgb.x = 0
            msg.rgb.y = 1
            msg.rgb.z = 0
        else:
            msg.rgb.x = 1
            msg.rgb.y = 1
            msg.rgb.z = 1

        return msg



