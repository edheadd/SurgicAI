from PyKDL import Vector, Rotation, Frame
import numpy as np
from surgical_robotics_challenge.camera import Camera

class DomainRandomization():
    
    def __init__(self, env, randomization_params):
        self.randomization_params = [True if x == "1" else False for x in randomization_params.split(",")]
        print("Randomization params: ", self.randomization_params)
        self.camera_randomization = self.randomization_params[0]
        self.psm_randomization = self.randomization_params[1]
        self.light_randomization = self.randomization_params[2]
        
        self.env = env.envs[0].unwrapped
        self.simulation_manager = self.env.simulation_manager
        
        
        self.cameraL = Camera(self.simulation_manager, "/ambf/env/cameras/cameraL")
        self.cameraR = Camera(self.simulation_manager, "cameraR")
        
        self.psm1 = self.env.psm1
        self.psm2 = self.env.psm2
        
        self.light = self.simulation_manager.get_obj_handle("/ambf/env/lights/light2")
              
        
    def randomize_environment(self):
        self.env.reset()
        
        # RANDOMIZE CAMERA POSITION
        self.camera_view_reset(self.camera_randomization)
        
        # RANDOMIZE PSM POSITION
        self.psm_reset(self.psm_randomization)
        
        # RANDOMIZE LIGHT POSITION
        self.light_reset(self.light_randomization)
        
        return
    
    def camera_view_reset(self, randomize):
        
        # create a frame that points at the position
        
        if randomize:
            xrand = np.random.uniform(-0.03, 0.03)
            yrand = np.random.uniform(-0.03, 0.03)
            zrand = np.random.uniform(-0.05, 0.01)
            
            L_goal_pos = Vector(xrand-0.002, yrand, zrand)
            L_goal_rpy = self.calculate_RPY(L_goal_pos)
            
            R_goal_pos = Vector(xrand+0.002, yrand, zrand)
            R_goal_rpy = self.calculate_RPY(R_goal_pos)
            
            # add noise to camera rotation
            
            print("Randomized camera positions: ", xrand, yrand, zrand)
            
            L_goal = Frame(L_goal_rpy, L_goal_pos)
            R_goal = Frame(R_goal_rpy, R_goal_pos)
            self.cameraL.move_cp(L_goal)
            self.cameraR.move_cp(R_goal)
        
        else:
            print("Using default camera positions")
            
    def psm_reset(self, randomize):
        
        if randomize:
            pass
        
    def light_reset(self, randomize):
        
        if randomize:
            pass
               
    def calculate_RPY(from_pos):
        look_at = Vector(0.0, 0.0, -1.0) 
        
        direction = np.array([
            look_at.x() - from_pos.x(),
            look_at.y() - from_pos.y(),
            look_at.z() - from_pos.z()
        ])
        direction = direction / np.linalg.norm(direction)

        yaw = np.arctan2(direction[1], direction[0]) - np.pi / 2
        pitch = np.arcsin(np.clip(direction[2], -1.0, 1.0))
        roll = 0.0 

        return Rotation.RPY(roll, pitch, yaw)


        
        
        
        
        
        
    