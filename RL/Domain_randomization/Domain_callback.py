from stable_baselines3.common.callbacks import BaseCallback
import rospy
from world_randomization_msgs.msg import Gravity, LightColor, LightNum, LightAttenuation, Friction
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
                     "light_attenuation": self.randomization_params[3],
                     "friction": self.randomization_params[4]}
        
        self.holdout = {"gravity": [-9.89, -9.87, -9.85, -9.83, -9.8, -9.78, -9.75, -9.73, -9.7],
                              "light_num": [2],
                              "light_color": [
                                (0.95, 0.90, 0.90),  
                                (0.90, 0.95, 0.90), 
                                (0.90, 0.90, 0.95), 
                                (0.95, 0.95, 0.85),
                                (0.92, 0.88, 0.95),  
                                (0.97, 0.97, 0.97), 
                                ],  
                              "light_attenuation": [
                                (0.6, 0.01, 0.001), 
                                (0.75, 0.025, 0.005),
                                (0.9, 0.045, 0.009), 
                                (0.55, 0.005, 0.0005),  
                                (0.8, 0.03, 0.002),
                                (0.95, 0.015, 0.0075),
                              ],
                              "friction": [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                            }

        self.gravity_pub = rospy.Publisher('/ambf/env/world_randomization/gravity', Gravity, queue_size=1)
        self.light_num_pub = rospy.Publisher('/ambf/env/world_randomization/light_num', LightNum, queue_size=1)
        self.light_color_pub = rospy.Publisher('/ambf/env/world_randomization/light_color', LightColor, queue_size=1)
        self.light_attenuation_pub = rospy.Publisher('/ambf/env/world_randomization/light_attenuation', LightAttenuation, queue_size=1)
        self.friction_pub = rospy.Publisher('/ambf/env/world_randomization/friction', Friction, queue_size=1)
        
        self.randomization_functions = [self.randomize_gravity,
                                        self.randomize_light_num,
                                        self.randomize_light_color,
                                        self.randomize_light_attenuation,
                                        self.randomize_friction]

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

        if self.dict.get("gravity"):
            holdouts = self.holdout.get("gravity", [])
            max_retries = 100
            for _ in range(max_retries):
                z = random.uniform(-9.9, -9.7)
                if all(abs(z - h) > 1e-3 for h in holdouts):
                    msg.gravity.z = z
                    break
            else:
                msg.gravity.z = -9.81
        else:
            msg.gravity.z = -9.81

        self.gravity_pub.publish(msg)

    def randomize_light_num(self):
        msg = LightNum()
        if self.dict.get("light_num"):
            holdouts = self.holdout.get("light_num", [])
            options = [i for i in range(4) if i not in holdouts]
            msg.num_lights = random.choice(options) if options else 0
        else:
            msg.num_lights = 0

        self.light_num_pub.publish(msg)

    def randomize_light_color(self):
        msg = LightColor()
        holdouts = self.holdout.get("light_color", [])
        max_retries = 100

        if self.dict.get("light_num"):
            for _ in range(max_retries):
                r = random.uniform(0.9, 1.0)
                g = random.uniform(0.85, 1.0)
                b = random.uniform(0.75, 1.0)
                if all(
                    abs(r - h[0]) > 1e-3 or
                    abs(g - h[1]) > 1e-3 or
                    abs(b - h[2]) > 1e-3
                    for h in holdouts
                ):
                    msg.rgb.r = r
                    msg.rgb.g = g
                    msg.rgb.b = b
                    break
            else:
                # fallback if stuck: pick midpoint
                msg.rgb.r = 0.95
                msg.rgb.g = 0.925
                msg.rgb.b = 0.875
        else:
            msg.rgb.r = msg.rgb.g = msg.rgb.b = 1.0

        msg.rgb.a = 1.0
        self.light_color_pub.publish(msg)

    def randomize_light_attenuation(self):
        msg = LightAttenuation()
        holdouts = self.holdout.get("light_attenuation", [])
        max_retries = 100

        if self.dict.get("light_attenuation"):
            for _ in range(max_retries):
                c = random.uniform(0.5, 1.0)
                l = random.uniform(0.005, 0.05)
                q = random.uniform(0.0001, 0.01)
                if all(
                    abs(c - h[0]) > 1e-3 or
                    abs(l - h[1]) > 1e-4 or
                    abs(q - h[2]) > 1e-5
                    for h in holdouts
                ):
                    msg.constant = c
                    msg.linear = l
                    msg.quadratic = q
                    break
            else:
                # fallback: average value
                msg.constant = 0.75
                msg.linear = 0.025
                msg.quadratic = 0.005
        else:
            msg.constant = 1.0
            msg.linear = 0.0
            msg.quadratic = 0.0

        self.light_attenuation_pub.publish(msg)

    def randomize_friction(self):
        msg = Friction()
        holdouts = self.holdout.get("friction", [])
        max_retries = 100

        if self.dict.get("friction"):
            for _ in range(max_retries):
                friction = random.uniform(0.1, 0.5)
                if all(
                    abs(friction - h) > 1e-3
                    for h in holdouts
                ):
                    msg.friction = friction
                    break
            else:
                msg.friction = 0.25
        else:
            msg.friction = 0.5

        self.friction_pub.publish(msg)