from stable_baselines3.common.callbacks import BaseCallback
import rospy
from world_randomization_msgs.msg import Gravity, LightColor, LightNum
from Domain_randomization.randomization_gui import ToggleApp
import random


class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_params, verbose=0):
        super().__init__(verbose)
        
        if not rospy.core.is_initialized():
            rospy.init_node('domain_randomization_callback', anonymous=True, disable_signals=True)
        
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        self.gravity_randomization = self.randomization_params[0]
        self.light_num_randomization = self.randomization_params[1]
        self.light_color_randomization = self.randomization_params[2]
                        
        self.name_list = ["gravity", "light_num", "light_color"]       
        
        self.gravity_pub = rospy.Publisher('/ambf/env/world_randomization/gravity', Gravity, queue_size=1)
        self.light_num_pub = rospy.Publisher('/ambf/env/world_randomization/light_num', LightNum, queue_size=1)
        self.light_color_pub = rospy.Publisher('/ambf/env/world_randomization/light_color', LightColor, queue_size=1)

        rospy.sleep(1.0)

    def _on_rollout_start(self):
        self.randomize() 

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]):
                self.randomize()           
        return True
    
    def randomize(self):
                
        self.gravity_randomization = self.randomization_params[0]
        self.light_num_randomization = self.randomization_params[1]
        self.light_color_randomization = self.randomization_params[2]
        
        self.randomize_gravity()
        self.randomize_light_num()
        self.randomize_light_color()
            
    def update_randomization_params(self, idx):
        self.randomization_params[idx] = not self.randomization_params[idx]
            
    from PyQt5.QtCore import QTimer

    def start_gui(self, app):
            self.gui_window = ToggleApp(self)
            self.gui_window.show()

    def randomize_gravity(self):
        msg = Gravity()
        msg.gravity.x = 0.0
        msg.gravity.y = 0.0
        msg.gravity.z = random.choice([-9.81, 0.0]) if self.gravity_randomization else -9.81
        self.gravity_pub.publish(msg)
    
    
    def randomize_light_num(self):
        msg = LightNum()
        msg.num_lights = random.randint(0, 3) if self.light_num_randomization else 0
        self.light_num_pub.publish(msg)

    
    def randomize_light_color(self):
        msg = LightColor()
        msg.rgb.r = random.uniform(0, 1) if self.light_color_randomization else 1
        msg.rgb.g = random.uniform(0, 1) if self.light_color_randomization else 1
        msg.rgb.b = random.uniform(0, 1) if self.light_color_randomization else 1
        msg.rgb.a = 1
        self.light_color_pub.publish(msg)

