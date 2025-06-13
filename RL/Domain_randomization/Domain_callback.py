from stable_baselines3.common.callbacks import BaseCallback
import rospy
from world_randomization_msgs.msg import Gravity, LightColor, LightNum
import random

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_params, verbose=0):
        super().__init__(verbose)
        
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        
        self.gravity_randomization = self.randomization_params[0]
        self.light_num_randomization = self.randomization_params[1]
        self.light_color_randomization = self.randomization_params[2]
        
        self.gravity_pub = rospy.Publisher('/ambf/env/World/my_plugin_name/enable', Gravity, queue_size=1)
        self.light_num_pub = rospy.Publisher('/ambf/env/World/my_plugin_name/light_direction', LightNum, queue_size=1)
        self.light_color_pub = rospy.Publisher('/ambf/env/World/my_plugin_name/light_color', LightColor, queue_size=1)


    def _on_rollout_start(self):
        if any(self.randomization_params):
            print("Randomizing")
            self.randomize() 

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]) and any(self.randomization_params):
                print("Randomizing")
                self.randomize()           
        return True
    
    def randomize(self):
        
        if self.gravity_randomization:
            self.randomize_gravity()
        
        if self.light_num_randomization:
            self.randomize_light_num()
        
        if self.light_color_randomization:
            self.randomize_light_color()
        
        
    def randomize_gravity(self):
        msg = Gravity()
        msg.gravity.x = 0.0
        msg.gravity.y = 0.0
        msg.gravity.z = random.choice([-9.81, 0.0])
        self.gravity_pub.publish(msg)
    
    
    def randomize_light_num(self):
        msg = LightNum()
        msg.num_lights = random.randint(0, 3)
        self.light_num_pub.publish(msg)

    
    def randomize_light_color(self):
        msg = LightColor()
        msg.rgb.x = random.uniform(0, 1)
        msg.rgb.y = random.uniform(0, 1)
        msg.rgb.z = random.uniform(0, 1)
        self.light_color_pub.publish(msg)
