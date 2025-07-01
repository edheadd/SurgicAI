from stable_baselines3.common.callbacks import BaseCallback
import rospy
from world_randomization_msgs.msg import Gravity, LightColor, LightNum, LightAttenuation
from Domain_randomization.randomization_gui import GUI
import random


class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_params, env, verbose=0):
        super().__init__(verbose)
        
        if not rospy.core.is_initialized():
            rospy.init_node('domain_randomization_callback', anonymous=True, disable_signals=True)
            
        self.env = env
        
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
                        
        self.dict = {"gravity": self.randomization_params[0],
                     "light_num": self.randomization_params[1],
                     "light_color": self.randomization_params[2],
                     "light_attenuation": self.randomization_params[3]}     
        
        self.gravity_pub = rospy.Publisher('/ambf/env/world_randomization/gravity', Gravity, queue_size=1)
        self.light_num_pub = rospy.Publisher('/ambf/env/world_randomization/light_num', LightNum, queue_size=1)
        self.light_color_pub = rospy.Publisher('/ambf/env/world_randomization/light_color', LightColor, queue_size=1)
        self.light_attenuation_pub = rospy.Publisher('/ambf/env/world_randomization/light_attenuation', LightAttenuation, queue_size=1)
        
        self.randomization_functions = [self.randomize_gravity,
                                        self.randomize_light_num,
                                        self.randomize_light_color,
                                        self.randomize_light_attenuation]

        rospy.sleep(1.0)

    def _on_rollout_start(self):
        self.randomize() 

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]):
                self.randomize()           
        return True
    
    def randomize(self):
                
        self.update_dict()
        
        for idx, func in enumerate(self.randomization_functions):
            func()
            
    def update_dict(self):
        for idx, key in enumerate(self.dict.keys()):
            self.dict[key] = self.randomization_params[idx]
        
    def update_randomization_params(self, idx, immediate_randomization):
        self.randomization_params[idx] = not self.randomization_params[idx]
        self.update_dict()
        
        if immediate_randomization:
            self.randomization_functions[idx]()

    def start_gui(self, app):
            self.gui_window = GUI(self)
            self.gui_window.show()

    def randomize_gravity(self):
        msg = Gravity()
        msg.gravity.x = 0.0
        msg.gravity.y = 0.0
        msg.gravity.z = random.uniform(-9.9, -9.7) if self.dict["gravity"] else -9.81
        self.gravity_pub.publish(msg)
    
    
    def randomize_light_num(self):
        msg = LightNum()
        msg.num_lights = random.randint(0, 3) if self.dict["light_num"] else 0
        self.light_num_pub.publish(msg)

    
    def randomize_light_color(self):
        msg = LightColor()
        if self.dict["light_num"]: 
            msg.rgb.r = random.uniform(0.9, 1.0)
            msg.rgb.g = random.uniform(0.85, 1.0)
            msg.rgb.b = random.uniform(0.75, 1.0)
        else:
            msg.rgb.r = msg.rgb.g = msg.rgb.b = 1.0          
        msg.rgb.a = 1
        self.light_color_pub.publish(msg)
        
    def randomize_light_attenuation(self):
        msg = LightAttenuation()
        if self.dict["light_attenuation"]:
            msg.constant = random.uniform(0.5, 1.0)
            msg.linear = random.uniform(0.005, 0.05)
            msg.quadratic = random.uniform(0.0001, 0.01)
        else:
            msg.constant = 1.0
            msg.linear = 0.0
            msg.quadratic = 0.0
        self.light_attenuation_pub.publish(msg)

