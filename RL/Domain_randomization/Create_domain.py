import yaml
import random
import os

class DomainRandomization():
    def __init__(self, randomization_params):
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        self.camera_randomization = self.randomization_params[0]
        self.light_randomization = self.randomization_params[1]
        self.PSM_randomization = self.randomization_params[2]
        self.phantom_randomization = self.randomization_params[3]
    
        self.world_configs, self.psm1_configs, self.psm2_configs, self.phantom_configs = self.load_random_options()
        self.config_options = {
            'world': self.world_configs,
            'psm1': self.psm1_configs,
            'psm2': self.psm2_configs,
            'phantom': self.phantom_configs            
        }
        
        
    def randomize_environment(self):
        """ Create new launch.yaml file"""
        
        default_path = os.path.expanduser("~/SurgicAI/RL/Launch_files/launch.yaml")
        with open(default_path, 'r') as default_launch:
            launch_dict = yaml.safe_load(default_launch)
        
        launch_dict['world config'] = self.randomize_world()
        launch_dict['multibody configs'] = self.randomize_PSMs(launch_dict['multibody configs'])
        launch_dict['multibody configs'] = self.randomize_phantom(launch_dict['multibody configs'])
        
        
        #TODO: generate unique path name based on random attributes
        output_path = os.path.expanduser("~/SurgicAI/RL/Launch_files/result.yaml")
        with open(output_path, 'w') as yaml_file:
            yaml.dump(launch_dict, yaml_file, default_flow_style=False)
        
        return output_path
    
    def load_random_options(self):
        options_path = os.path.expanduser("~/SurgicAI/RL/Domain_randomization/random_options.yaml")
        with open(options_path, "r") as options_file:
            data = yaml.safe_load(options_file)

        world_configs = data["world configs"]
        psm1_configs = data["PSM1 configs"]
        psm2_configs = data["PSM2 configs"]
        phantom_configs = data["phantom configs"] 
    
        return world_configs, psm1_configs, psm2_configs, phantom_configs
    
    def pick_random_option(self, config_name):
        return random.choice(self.config_options[config_name])
    
    def randomize_world(self):
        #TODO: implement randomization for world, seperating lights and camera
        #      - use similar load - > modify - > dump process as in psms + phantom
        
        return 'ADF/world/world_stereo_varylight.yaml'
    
    def randomize_PSMs(self, multibody_configs):
            
        if self.PSM_randomization:
            psm1_config = self.pick_random_option('psm1')
            psm2_config = self.pick_random_option('psm2')         
            
            multibody_configs[0] = psm1_config
            multibody_configs[1] = psm2_config
            
        print("Set PSM1 config to ", multibody_configs[0])
        print("Set PSM2 config to ", multibody_configs[1])
                        
        return multibody_configs
            
    def randomize_phantom(self, multibody_configs):
        
        if self.phantom_randomization:
            phantom_config = self.pick_random_option('phantom')
            multibody_configs[2] = phantom_config
        
        print("Set phantom config to", multibody_configs[2])
        
        return multibody_configs
        