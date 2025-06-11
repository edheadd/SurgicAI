from stable_baselines3.common.callbacks import BaseCallback
import rospy
from ambf_msgs.msg import WorldCmd
import random

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_params, verbose=0):
        super().__init__(verbose)
        
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        
        self.gravity_randomization = self.randomization_params[0]
        self.light_num_randomization = self.randomization_params[1]
        self.light_color_randomization = self.randomization_params[2]
        
        self.pub = rospy.Publisher('/WorldRandomization/Commands/Command', WorldCmd, queue_size=10)
        
        self.msgID = 1;

    def _on_rollout_start(self):
        if any(self.randomization_params):
            print("Randomizing AMBF (Rollout start)")
            self.randomize() 

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]) and any(self.randomization_params):
                print("Randomizing AMBF (Step complete)")
                self.randomize()           
        return True
    
    def randomize(self):
        
        msg = WorldCmd()
        
        msg.msgID = self.msgID
        self.msgID += 1
        
        msg = self.randomize_gravity(msg, self.gravity_randomization)
        msg = self.randomize_light_num(msg, self.light_num_randomization)
        msg = self.randomize_light_color(msg, self.light_color_randomization)
                                
        self.pub.publish(msg)
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
    
    
    def randomize_light_num(self, msg, randomize):
        msg.randomize_light_num = randomize
        if randomize:
            msg.num_lights = random.randint(0, 3)
        else:
            msg.num_lights = 0

        return msg
    
    def randomize_light_color(self, msg, randomize):
        msg.randomize_light_color = randomize
        if randomize:
            msg.rgb.x = random.uniform(0, 1)
            msg.rgb.y = random.uniform(0, 1)
            msg.rgb.z = random.uniform(0, 1)
        else:
            msg.rgb.x = 1
            msg.rgb.y = 1
            msg.rgb.z = 1

        return msg



