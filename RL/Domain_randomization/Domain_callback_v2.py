from stable_baselines3.common.callbacks import BaseCallback
import rospy
import world_randomization_msgs.msg
from Domain_randomization.randomization_gui import GUI
import random
import yaml

class DomainRandomizationCallback(BaseCallback):
    def __init__(self, randomization_args, env, verbose=0):
        super().__init__(verbose)
        
                
        self.randomization_params = [True if x == "1" else False for x in randomization_args.split(",")]
        self.env = env       
        with open("Domain_randomization/randomization_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.func_dict = {}
        
        i = 0
        for key, value in self.config.items():
            
            function = self.unpack_config(value)
            status = value.get("status", "False")
            arg_status = self.randomization_params[i]
            if arg_status != status:
                print(f"{key} status override, using {arg_status} instead of {status}")  
                status = arg_status        
                      
            key_dict = {
                "status": status,
                "description": value["description"],
                "function": function
            }            
            self.func_dict[key] = key_dict
            i += 1            

    def _on_rollout_start(self):
        self.randomize() 

    def _on_step(self):
        if "dones" in self.locals:
            if any(self.locals["dones"]):
                self.randomize()           
        return True
    
    def randomize(self):        
        for key, value in self.func_dict.items():
            value["function"](value["status"])
            
    def unpack_config(self, cfg):
        topic_name = f"/ambf/env/world_randomization/{cfg['namespace']}"
        msg_class = getattr(world_randomization_msgs.msg, cfg['message'])
        topic_publisher = rospy.Publisher(topic_name, msg_class, queue_size=1)

        def randomization_function(randomize):
            msg = msg_class()
            data = {}

            for field, field_cfg in cfg["fields"].items():
                if isinstance(field_cfg, dict) and "type" not in field_cfg:
                    sub_data = {}
                    for sub_field, sub_cfg in field_cfg.items():
                        value = self.get_randomized_value(sub_cfg, randomize)
                        sub_data[sub_field] = value
                    data[field] = sub_data
                else:
                    value = self.get_randomized_value(field_cfg, randomize)
                    data[field] = value

            self.set_message_fields(msg, data)
            topic_publisher.publish(msg)

        return randomization_function
    
    def get_randomized_value(self, cfg, randomize):
        if not randomize:
            return cfg["default"]
        
        field_type = cfg.get("type", "float")
        holdouts = cfg.get("holdouts", [])
        value_range = cfg.get("range", [0, 0])
        
        max_retries = 100
        for _ in range(max_retries):
            if field_type == "float":
                value = random.uniform(*value_range)
            elif field_type == "int":
                value = random.randint(*value_range)
            else:
                raise ValueError(f"Unsupported type: {field_type}")
            if value not in holdouts:
                return value
            
        return (sum(value_range) / 2) if field_type == "float" else int(sum(value_range) / 2)


    def set_message_fields(self, msg, data):
        for attr, value in data.items():
            if isinstance(value, dict):
                sub_msg = getattr(msg, attr)
                for sub_attr, sub_value in value.items():
                    setattr(sub_msg, sub_attr, sub_value)
            else:
                setattr(msg, attr, value)
                
    def start_gui(self, app):
            self.gui_window = GUI(self)
            self.gui_window.show()
        
    def update_randomization_params(self, name, immediate_randomization):
        
        self.func_dict[name]["status"] = not self.func_dict[name]["status"]
        if immediate_randomization:
            self.func_dict[name]["function"](self.func_dict[name]["status"])
    
    def reset_env(self):
        self.env.unwrapped.reset()
        self.randomize()



