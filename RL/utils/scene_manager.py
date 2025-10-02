import time
import numpy as np
from PyKDL import Frame, Rotation, Vector
from surgical_robotics_challenge.psm_arm import PSM
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.utils.task3_init import NeedleInitialization
from utils.needle_kinematics import NeedleKinematics
from surgical_robotics_challenge.kinematics.psmKinematics import *


class SceneManager:
    """Manages simulation initialization, PSM arms, and needle handling"""
    
    def __init__(self, env):
        self.env = env
        self.psm_list = []
        self.psm_goal_list = []
        self.jaw_angle_list = []
        
        # Initialize simulation components
        self.simulation_manager = SimulationManager('src_client')
        self.world_handle = self.simulation_manager.get_world_handle()
        self.world_handle.reset()
        time.sleep(0.5)
        self.scene = Scene(self.simulation_manager)
        
        # Initialize arms
        self.psm1 = PSM(self.simulation_manager, 'psm1', add_joint_errors=False)
        self.psm2 = PSM(self.simulation_manager, 'psm2', add_joint_errors=False)
        self.psm_list = [self.psm1, self.psm2]
        self.ecm = ECM(self.simulation_manager, 'CameraFrame')
        self.camera_view_reset()

        # Initialize needle
        self.needle = NeedleInitialization(self.simulation_manager)
        self.needle_kin = NeedleKinematics()
        self.needle_init_pos = self.needle.needle.get_pos()
        
        # Move PSM baselinks
        self.psm1.base.set_pos(Vector(0.14, 0.34, 0.8))
        self.psm2.base.set_pos(Vector(-0.08, 0.34, 0.8))

        # Set initial positions
        self.init_psm1 = np.array([ 0.04629208,0.00752399,-0.08173992,-3.598019,-0.05762508,1.2738742,0.8],dtype=np.float32)
        self.init_psm2 = np.array([-0.03721037,  0.01213105, -0.08036895, -2.7039163, 0.07693613, 2.0361109, 0.8],dtype=np.float32)


        self.psm_goal_list = [self.init_psm1.copy(), self.init_psm2.copy()]
        
        # Move to initial positions
        self.psm_step(self.init_psm1, 1)
        time.sleep(0.5)
        self.psm_step(self.init_psm2, 2)
        time.sleep(0.5)
        
    def step(self):
        """Perform a simulation step"""

        jaw1_angle = self.psm_goal_list[0][-1]
        jaw2_angle = self.psm_goal_list[1][-1]
        self.jaw_angle_list = [jaw1_angle, jaw2_angle]
        self.world_handle.update()
        self.psm_step(self.psm_goal_list[self.env.psm_idx-1], self.env.psm_idx)
    
    def env_reset(self):
        """Reset the simulation state"""
        self.psm1.actuators[0].deactuate()
        self.psm2.actuators[0].deactuate()
        self.psm_goal_list[0] = np.copy(self.init_psm1)
        self.psm_goal_list[1] = np.copy(self.init_psm2)
        self.psm_step(self.psm_goal_list[0], 1)
        self.psm_step(self.psm_goal_list[1], 2)
        self.world_handle.reset()
        self.camera_view_reset()
        time.sleep(1.0)
    
    def camera_view_reset(self, reset_noise=False):
        """Reset camera view"""
        camera_pose = self.ecm.camera_handle.get_pose()
        rotation = camera_pose.M
        base_vec = camera_pose.p

        if reset_noise:
            vector = base_vec + Vector(
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02),
                np.random.uniform(-0.02, 0.02)
            )
        else:
            vector = base_vec
        
        ecm_pos_origin = Frame(rotation, vector)
        self.ecm.servo_cp(ecm_pos_origin)
    
    def psm_step(self, obs, psm_idx):
        """Move PSM to specified position"""
        X = obs[0]
        Y = obs[1]
        Z = obs[2]
        Roll = obs[3]
        Pitch = obs[4]
        Yaw = obs[5]
        Jaw_angle = obs[6]
        
        T_goal = Frame(Rotation.RPY(Roll, Pitch, Yaw), Vector(X, Y, Z))
        self.psm_list[psm_idx-1].servo_cp(T_goal)
        self.psm_list[psm_idx-1].set_jaw_angle(Jaw_angle)
    
    def psm_step_move(self, obs, psm_idx, execute_time=0.5):
        """Smoothly move PSM to position"""
        X = obs[0]
        Y = obs[1]
        Z = obs[2]
        Roll = obs[3]
        Pitch = obs[4]
        Yaw = obs[5]
        Jaw_angle = obs[6]
        
        T_goal = Frame(Rotation.RPY(Roll, Pitch, Yaw), Vector(X, Y, Z))
        self.psm_list[psm_idx-1].move_cp(T_goal, execute_time)
        self.psm_list[psm_idx-1].set_jaw_angle(Jaw_angle)
            
    def needle_randomization(self):
        """
        Initialize needle at random positions in the world
        """

        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])

        origin_p = self.needle_init_pos
        origin_rz = 0.0

        random_range = self.env.random_range
        random_x = np.random.uniform(-random_range[0],random_range[0])
        random_y = np.random.uniform(-random_range[1],random_range[1])
        random_rz = np.random.uniform(-random_range[2],random_range[2])

        origin_p[0] += random_x
        origin_p[1] += random_y
        origin_rz += random_rz

        new_rot = Rotation(np.cos(origin_rz),-np.sin(origin_rz),0,
                            np.sin(origin_rz),np.cos(origin_rz),0,
                            0.0,0.0,1.0)
                
        needle_pos_new = Frame(new_rot,origin_p)
        self.needle.needle.set_pose(needle_pos_new)
        
    def entry_goal_evaluator(self,deg=120,dev_trans=[0,0,0],dev_Yangle = 0.0,idx=2,noise=False):
        rotation_noise = Rotation.RotY(np.deg2rad(dev_Yangle))
        translation_noise = Vector(0, 0, 0)
        noise_in_entry = Frame(rotation_noise, translation_noise)

        entry_in_world = self.scene.entry1_measured_cp()
        entry_in_world.p[1] -= 0.005
        entry_in_world.p[2] += 0.006
        noise_in_world = entry_in_world*noise_in_entry
        entry_in_base = self.psm_list[idx-1].get_T_w_b()*noise_in_world # entry with angle deviation

        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_tip_in_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_tip_in_entry = Vector(dev_trans[0],dev_trans[1],dev_trans[2])
        tip_in_entry = Frame(rotation_tip_in_entry,trans_tip_in_entry)


        tip_in_world = self.needle_kin.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_tip = tip_in_world.Inverse()*gripper_in_world

        gripper_in_base = entry_in_base*tip_in_entry*gripper_in_tip
        array_insert = self.Frame2Vec(gripper_in_base)
        array_insert = np.append(array_insert,0.0)
        if noise:
            ranges = np.array([0.001, 0.001, 0.001, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5), 0])
            random_noise = np.random.uniform(-ranges, ranges)
            array_insert += random_noise
        return array_insert
    
    def insert_goal_evaluator(self,deg=120,dev=[0,0,0],idx=2):
        exit_in_world = self.scene.exit1_measured_cp()
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*exit_in_world

        # entry_pos.Inverse() to obtain the inverse transformation matrix
        # rotation_matrix = np.array([[0,-1,0],[0,0,1],[-1,0,0]]).astype(np.float32)
        rotation_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]]).astype(np.float32)
        rotation_front_in_exit = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_front_in_exit = Vector(dev[0],dev[1],dev[2])
        front_in_exit = Frame(rotation_front_in_exit,trans_front_in_exit)

        front_in_world = self.needle_kin.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_front = front_in_world.Inverse()*gripper_in_world

        gripper_in_base = exit_in_base*front_in_exit*gripper_in_front
        array_insert = self.Frame2Vec(gripper_in_base)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def handover_goal_evaluator(self,deg=110,dev=[0,0,0],idx=1):
        exit_in_world = self.scene.exit1_measured_cp()
        rotation_decrease_y = Rotation.RotY(-np.deg2rad(50))
        new_rotation = exit_in_world.M * rotation_decrease_y
        handover_in_world = Frame(new_rotation, Vector(exit_in_world.p[0] + 0.03, exit_in_world.p[1], exit_in_world.p[2] + 0.03))
        exit_in_base = self.psm_list[idx-1].get_T_w_b()*handover_in_world

        rotation_matrix = np.array([[-1,0,0],[0,0,1],[0,1,0]]).astype(np.float32)
        rotation_front_in_exit = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])
        trans_front_in_exit = Vector(dev[0],dev[1],dev[2])
        front_in_exit = Frame(rotation_front_in_exit,trans_front_in_exit)

        front_in_world = self.needle_kin.get_pose_angle(deg)
        gripper_in_world = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        gripper_in_front = front_in_world.Inverse()*gripper_in_world

        gripper_in_base = exit_in_base*front_in_exit*gripper_in_front
        array_handover = self.Frame2Vec(gripper_in_base)
        array_handover = np.append(array_handover,0.0)
        return array_handover
    
    # Overridden entry goal evaluator for place subtask
    def place_entry_goal_evaluator(self,idx = 2):
        self.entry_w = self.scene.entry1_measured_cp()
        entry_pos = self.psm_list[idx-1].get_T_w_b()*self.entry_w
        rotation_matrix = np.array([[1,0,0],[0,0,1],[0,-1,0]]).astype(np.float32)
        rotation_entry = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        T_tip_base = self.needle_kin.get_tip_pose()
        T_gripper_base = self.psm_list[idx-1].get_T_b_w()*convert_mat_to_frame(self.psm_list[idx-1].measured_cp())
        T_gripper_tip = T_tip_base.Inverse()*T_gripper_base

        T_insert = entry_pos
        T_insert.M *= rotation_entry
        T_insert = T_insert*T_gripper_tip
        array_insert = self.Frame2Vec(T_insert)
        array_insert = np.append(array_insert,0.0)
        return array_insert
    
    def needle_random_grasping_evaluator(self,lift_height):
        self.random_degree = np.random.uniform(12, 15)
        self.grasping_pos = self.needle_kin.get_random_grasp_point()
        needle_rot = self.grasping_pos.M
        needle_trans_lift = Vector(self.grasping_pos.p.x(),self.grasping_pos.p.y(),self.grasping_pos.p.z()+lift_height)
        needle_goal_lift = Frame(needle_rot, needle_trans_lift)

        T_calibrate = np.array([[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_calibrate[:3, :3]

        rotation_calibrate = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        needle_goal_lift.M = needle_goal_lift.M * rotation_calibrate # To be tested
        
        psm_goal_lift = self.psm2.get_T_w_b()*needle_goal_lift

        T_goal = np.array([[0,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]]).astype(np.float32)
        rotation_matrix = T_goal[:3, :3]

        rotation = Rotation(rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
                            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
                            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2])

        psm_goal_lift.M = psm_goal_lift.M*rotation

        array_goal_base = self.Frame2Vec(psm_goal_lift)
        array_goal_base = np.append(array_goal_base,0.0)
        return array_goal_base
    
    def needle_goal_evaluator(self,lift_height=0.007, psm_idx=2, deg_angle = None):
        '''
        Evaluate the target goal for needle grasping in Robot frame.
        '''

        if deg_angle is None:
            grasp_in_World = self.needle_kin.get_bm_pose()

        else:
            grasp_in_World = self.needle_kin.get_pose_angle(deg_angle)

        lift_in_grasp_rot = Rotation(1, 0, 0,
                                    0, 1, 0,
                                    0, 0, 1)    
        lift_in_grasp_trans = Vector(0,0,lift_height)
        lift_in_grasp = Frame(lift_in_grasp_rot,lift_in_grasp_trans)

        if psm_idx == 2:
            gripper_in_lift_rot = Rotation(0, -1, 0,
                                            -1, 0, 0,
                                            0, 0, -1)
        else:
            gripper_in_lift_rot = Rotation(0, 1, 0,
                                            1, 0, 0,
                                            0, 0, -1)           

        gripper_in_lift_trans = Vector(0.0,0.0,0.0)
        gripper_in_lift = Frame(gripper_in_lift_rot,gripper_in_lift_trans)

        gripper_in_world = grasp_in_World*lift_in_grasp*gripper_in_lift
        gripper_in_base = self.psm_list[psm_idx-1].get_T_w_b()*gripper_in_world
        

        array_goal_base = self.Frame2Vec(gripper_in_base)
        array_goal_base = np.append(array_goal_base,0.0)
        return array_goal_base
    
    def needle_multigoal_evaluator(self, lift_height=0.007, psm_idx=2, start_degree=5, end_degree=30, num_points=25):
        """
        Evaluate the multiple allowed goal grasping points.
        """
        interpolated_transforms = self.needle_kin.get_interpolated_transforms(start_degree, end_degree, num_points)
        goals = []

        for transform in interpolated_transforms:
            grasp_in_World = transform

            lift_in_grasp_rot = Rotation(1, 0, 0,
                                         0, 1, 0,
                                         0, 0, 1)
            lift_in_grasp_trans = Vector(0, 0, lift_height)
            lift_in_grasp = Frame(lift_in_grasp_rot, lift_in_grasp_trans)

            if psm_idx == 2:
                gripper_in_lift_rot = Rotation(0, -1, 0,
                                               -1, 0, 0,
                                               0, 0, -1)
            else:
                gripper_in_lift_rot = Rotation(0, 1, 0,
                                               1, 0, 0,
                                               0, 0, -1)

            gripper_in_lift_trans = Vector(0.0, 0.0, 0.0)
            gripper_in_lift = Frame(gripper_in_lift_rot, gripper_in_lift_trans)

            gripper_in_world = grasp_in_World * lift_in_grasp * gripper_in_lift
            gripper_in_base = self.psm_list[psm_idx - 1].get_T_w_b() * gripper_in_world

            array_goal_base = self.Frame2Vec(gripper_in_base)
            array_goal_base = np.append(array_goal_base, 0.0)
            goals.append(array_goal_base)

        return goals


    def Frame2Vec(self,goal_frame,bound = True):
        """
        Convert Frame variables into vector forms.
        """
        X_goal = goal_frame.p.x()
        Y_goal = goal_frame.p.y()
        Z_goal = goal_frame.p.z()
        rot_goal = goal_frame.M
        roll_goal,pitch_goal,yaw_goal  = rot_goal.GetRPY()
        if bound:
            if (roll_goal <= np.deg2rad(-360)):
                roll_goal += 2*np.pi
            elif (roll_goal > np.deg2rad(0)):
                roll_goal -= 2*np.pi
        array_goal = np.array([X_goal,Y_goal,Z_goal,roll_goal,pitch_goal,yaw_goal],dtype=np.float32)
        return array_goal
    
    def approach_and_grasp(self):
        # Approach and grasp the needle
        self.needle_obs = self.needle_random_grasping_evaluator(0.0007)
        self.needle_obs = np.append(self.needle_obs,0.8)
        self.psm_step_move(self.needle_obs,2)
        time.sleep(0.6)
        self.needle_obs[-1] = 0.0
        self.psm_step(self.needle_obs,2)
        time.sleep(0.5)
        self.psm2.actuators[0].actuate("Needle")
        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])
    
    def place_at_entry(self):
        # Place the needle at the entry
        self.entry_obs = self.entry_goal_evaluator(idx=2,dev_trans=[0,0,0.001],noise=False) # Close noise in this case
        self.adjusted_entry_obs = np.copy(self.entry_obs)
        self.adjusted_entry_obs[1] = self.adjusted_entry_obs[1] + 0.003
        self.adjusted_entry_obs[2] = self.adjusted_entry_obs[2] + 0.003
        self.psm_step_move(self.adjusted_entry_obs,2,execute_time=1.2)
        time.sleep(1.4) 
        self.psm_step_move(self.entry_obs,2,execute_time=1)
        time.sleep(1.8)

    def insert_needle(self):
        # Insert the needle
        self.insert_obs = self.insert_goal_evaluator(90,[0.002,0,0])
        self.psm_step_move(self.insert_obs,2,execute_time=0.7)
        self.psm_goal_list[1] = np.copy(self.insert_obs)
        time.sleep(1)

    def regrasp_needle(self):        
        # Regrasp the needle
        self.regrasp_obs = self.needle_goal_evaluator(deg_angle=105,lift_height=0.005,psm_idx=1)
        self.regrasp_obs[-1] = 0.8
        self.psm_step_move(self.regrasp_obs,1,execute_time=0.8)
        time.sleep(1)
        self.regrasp_obs[-1] = 0.0
        self.psm_step(self.regrasp_obs,1)
        time.sleep(0.4)
        self.psm1.actuators[0].actuate("Needle")
        self.needle.needle.set_force([0.0,0.0,0.0])
        self.needle.needle.set_torque([0.0,0.0,0.0])
        self.psm_goal_list[1][-1] = 0.8
        self.psm_step(self.psm_goal_list[1],2)
        time.sleep(0.3)
            
