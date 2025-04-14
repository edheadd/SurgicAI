from surgical_robotics_challenge.kinematics.psmIK import *
import time


class Light:
    def __init__(self, simulation_manager, name):
        self.simulation_manager = simulation_manager
        self.name = name
        self.light_handle = self.simulation_manager.get_obj_handle(name)
        time.sleep(0.1)

    def move_cp(self, T_c_w):
        self.light_handle.set_pose(T_c_w)

    def get_rpy(self):
        return self.light_handle.get_rotation()