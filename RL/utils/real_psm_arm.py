import time
import numpy as np
from threading import Thread, Lock, RLock
from geometry_msgs.msg import PoseStamped
from transforms3d.euler import quat2euler, euler2quat

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
        
        self.measured_cp = None
        self._cmd = None
        
        self._setup_ros_interface()
        
        cmd_thread = Thread(target=self.send_cmds, daemon=True)
        cmd_thread.start()

    def _setup_ros_interface(self):
        self.psm_pub = self.ral_instance.publisher('/PSM2/servo_cp', PoseStamped, queue_size=1)
        self.psm_sub = self.ral_instance.subscriber('/PSM2/measured_cp', PoseStamped, self._state_callback)

    def _state_callback(self, msg):
        # updated measured pose
        x,y,z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        qx, qy, qz, qw = msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w
        rpy = quat2euler([qw, qx, qy, qz])
        self.measured_cp = np.array([x, y, z, rpy[0], rpy[1], rpy[2]])

    def servo_cp(self, goal_cp):
        with _psm_global_compute_lock, self._state_lock:
            x,y,z = goal_cp[0], goal_cp[1], goal_cp[2]
            rpy = goal_cp[3:6]
            quat = euler2quat(rpy[0], rpy[1], rpy[2])
            qw, qx, qy, qz = quat
            _cmd = PoseStamped()
            _cmd.pose.position.x = x
            _cmd.pose.position.y = y
            _cmd.pose.position.z = z
            _cmd.pose.orientation.x = qx
            _cmd.pose.orientation.y = qy
            _cmd.pose.orientation.z = qz
            _cmd.pose.orientation.w = qw

    def measured_cp(self):
        return self.measured_cp
        
    def send_cmds(self):
        while True:
            with self._cmd_lock:
                if self._cmd is not None:
                    self.psm_pub.publish(self._cmd)
            time.sleep(0.05)