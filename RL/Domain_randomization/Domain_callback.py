from stable_baselines3.common.callbacks import BaseCallback
import rospy
from ambf_msgs.msg import WorldCmd

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_params, verbose=0):
        super().__init__(verbose)
        
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        self.randomize_gravity = self.randomization_params[0]
        
        self.pub = rospy.Publisher('/WorldRandomization/Commands/Command', WorldCmd, queue_size=10)

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
        msg.randomize_gravity = self.randomize_gravity
        if self.randomize_gravity:
            msg.gravity.x = 0
            msg.gravity.y = 0
            msg.gravity.z = -9.81
        self.pub.publish(msg)
        print("Published randomization command")



