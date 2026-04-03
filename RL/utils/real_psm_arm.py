import time
import numpy as np
from threading import Thread, Lock, RLock
from ambf_msgs.msg import RigidBodyCmd, RigidBodyState, ActuatorCmd, GhostObjectState
from geometry_msgs.msg import Pose
from PyKDL import Vector, Frame, Rotation
from utils.kinematics.kinematics import PSMKinematicSolver
from utils.kinematics.DH import enforce_limits
from utils.utils import convert_mat_to_frame
from utils.interpolation import Interpolation
from transforms3d.euler import quat2euler

_psm_global_compute_lock = RLock()

class PSM:
    def __init__(self, ral_instance, name, env = "ambf/env"):
        self.name = name
        self.env = env
        self.ral_instance = ral_instance       
        
        self._kd = PSMKinematicSolver()
        self.interpolater = Interpolation()        
        self._thread_lock = Lock()
        self._cmd_lock = Lock()
        self._actuator_cmd_lock = Lock()
        self._state_lock = Lock()
        
        self._T_b_w = None 
        self._T_w_b = None 
        self._measured_jp = None
        self._measured_jv = None
        self.orientation = None
        self._cmd = None
        self._actuator_cmd = None
        
        self.jaw_angle = 0.5
        self.grasp_actuation_jaw_angle = 0.05
        self.graspable_objs_prefix = ["Needle", "Thread", "Puzzle"]
        self.grasped = [False]*len(self.graspable_objs_prefix) 
        self.grasped_obj_name = None
        self.left_finger_sensed_objs = []
        self.right_finger_sensed_objs = []
        
        self._setup_ros_interface()
        
        cmd_thread = Thread(target=self.send_cmds, daemon=True)
        cmd_thread.start()

    def _setup_ros_interface(self):
        topic = f'{self.env}/{self.name}'
        self.psm_pub = self.ral_instance.publisher(f'/{topic}/baselink/Command', RigidBodyCmd, queue_size=1)
        self.psm_sub = self.ral_instance.subscriber(f'/{topic}/baselink/State', RigidBodyState, self._state_callback)
        #self.actuator_pub = self.ral_instance.publisher(f'/{self.env}/ghosts/{self.name}/Actuator0/Command', ActuatorCmd, queue_size=1)
        #self.left_finger_ghost = self.ral_instance.subscriber(f'/{self.env}/ghosts/{self.name}/left_finger_ghost/State', GhostObjectState, self._left_finger_callback)
        #self.right_finger_ghost = self.ral_instance.subscriber(f'/{self.env}/ghosts/{self.name}/right_finger_ghost/State', GhostObjectState, self._right_finger_callback)

    def _state_callback(self, msg):
        v = Vector(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        quat = msg.pose.orientation
        rpy = quat2euler([quat.w, quat.x, quat.y, quat.z])
        r = Rotation.RPY(rpy[0], rpy[1], rpy[2])
        with self._state_lock:
            self._T_b_w = Frame(r, v)
            self._T_w_b = self._T_b_w.Inverse()
            self._measured_jp = list(msg.joint_positions)[:6]
            self._measured_jv = list(msg.joint_velocities)[:6]
            self.orientation = msg.pose.orientation
        if self._cmd is None:
            self.servo_jp(self._measured_jp)

    def get_T_b_w(self):
        with self._state_lock:
            if self._T_b_w is None:
                return Frame()
            return self._T_b_w

    def get_T_w_b(self):
        with self._state_lock:
            if self._T_w_b is None:
                return Frame()
            return self._T_w_b

    def servo_cp(self, T_t_b):
        with _psm_global_compute_lock, self._state_lock:
            if type(T_t_b) in [np.matrix, np.ndarray]:
                T_t_b = convert_mat_to_frame(T_t_b)
            self.ik_solution = self._kd.compute_IK(T_t_b)
            self._ik_solution = enforce_limits(self.ik_solution, self._kd.JOINT_LIMITS_LOWER, self._kd.JOINT_LIMITS_UPPER)
            self.servo_jp(self._ik_solution)
    
    def servo_jp(self, jp_vec):
        with self._cmd_lock:
            msg = self._cmd if self._cmd is not None else RigidBodyCmd()
            # Ensure message has at least 8 joints (6 arm + 2 jaw)
            if len(msg.joint_cmds) < 8:
                msg.joint_cmds = list(msg.joint_cmds) + [0.0] * (8 - len(msg.joint_cmds))
            if len(msg.joint_cmds_types) < 8:
                msg.joint_cmds_types = list(msg.joint_cmds_types) + [RigidBodyCmd.TYPE_POSITION] * (8 - len(msg.joint_cmds_types))
            # Set arm joints 0-5
            for i in range(6):
                msg.joint_cmds[i] = float(jp_vec[i])
                msg.joint_cmds_types[i] = RigidBodyCmd.TYPE_POSITION
            # Set jaw joints to fixed 0.5
            msg.joint_cmds[6] = 0.5
            msg.joint_cmds[7] = 0.5
            msg.joint_cmds_types[6] = RigidBodyCmd.TYPE_POSITION
            msg.joint_cmds_types[7] = RigidBodyCmd.TYPE_POSITION
            self._cmd = msg
        
    def measured_jp(self):
        """Get all measured joint positions from the current RigidBodyState message"""
        with self._state_lock:
            return self._measured_jp
    
    def measured_jv(self):
        """Get all measured joint velocities from the current RigidBodyState message"""
        with self._state_lock:
            return self._measured_jv
    
    def measured_cp(self):
        with self._state_lock:
            jp = list(self._measured_jp)
        jp.append(0.0)
        return self._kd.compute_FK(jp[:7], 7)
        
    def send_cmds(self):
        while True:
            with self._cmd_lock:
                if self._cmd is not None:
                    self.psm_pub.publish(self._cmd)
            time.sleep(0.05)
                    
    def get_lower_limits(self):
        return self._kd.JOINT_LIMITS_LOWER
    
    def get_upper_limits(self):
        return self._kd.JOINT_LIMITS_UPPER