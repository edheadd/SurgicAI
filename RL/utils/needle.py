from PyKDL import Frame, Rotation, Vector
import numpy as np
import time
from threading import Lock
from ros_abstraction_layer import ral
from ambf_msgs.msg import RigidBodyState, RigidBodyCmd
from geometry_msgs.msg import Wrench


def pose_msg_to_frame(msg):
    """Convert geometry_msgs/Pose to PyKDL Frame"""
    p = Vector(msg.position.x, msg.position.y, msg.position.z)
    R = Rotation.Quaternion(
        msg.orientation.x,
        msg.orientation.y,
        msg.orientation.z,
        msg.orientation.w
    )
    return Frame(R, p)


class Needle:
    """Unified Needle class combining kinematics and control via RAL"""
    
    # Needle geometry constants
    Radius = 0.1018
    T_bINn = Frame(Rotation.RPY(0., 0., 0.), Vector(-Radius, 0., 0.) / 10.0)
    T_mINn = Frame(
        Rotation.RPY(0., 0., -np.pi / 3),
        Vector(-Radius * np.cos(np.pi / 3), Radius * np.sin(np.pi / 3), 0.) / 10.0
    )
    T_tINn = Frame(
        Rotation.RPY(0., 0., -np.pi / 3 * 2),
        Vector(-Radius * np.cos(np.pi / 3 * 2), Radius * np.sin(np.pi / 3 * 2), 0.) / 10.0
    )
    T_bmINn = Frame(
        Rotation.RPY(0., 0., -np.pi / 6),
        Vector(-Radius * np.cos(np.pi / 6), Radius * np.sin(np.pi / 6), 0.) / 10.0
    )

    def __init__(self, ral_instance):
        """
        Initialize Needle with RAL instance.
        
        :param ral_instance: RAL instance for ROS communication
        """
        self.ral = ral_instance
        self._state_lock = Lock()
        
        # Needle pose in world
        self._T_nINw = Frame()
        self._needle_pos = Vector(0, 0, 0)
        
        # Setup ROS interface
        self._setup_ros_interface()
        time.sleep(0.5)

    def _setup_ros_interface(self):
        """Setup ROS subscribers and publishers for needle"""
        # Subscribe to needle state
        self.ral.subscriber(
            '/ambf/env/phantom/Needle/State',
            RigidBodyState,
            self._needle_state_cb,
            queue_size=1
        )
        
        # Publisher for wrench commands
        self._wrench_pub = self.ral.publisher(
            '/ambf/env/phantom/Needle/Command/Wrench',
            Wrench,
            queue_size=1
        )

    def _needle_state_cb(self, msg):
        """Callback for needle state updates"""
        with self._state_lock:
            self._T_nINw = pose_msg_to_frame(msg.pose)
            self._needle_pos = self._T_nINw.p

    def get_pos(self):
        """Get needle position as Vector"""
        with self._state_lock:
            return Vector(self._needle_pos.x(), self._needle_pos.y(), self._needle_pos.z())

    def get_pose(self):
        """Get needle pose as Frame"""
        with self._state_lock:
            return Frame(self._T_nINw.M, self._T_nINw.p)

    def set_pose(self, pose_frame):
        """Set needle pose (command to simulator)"""
        # Note: Direct pose setting may require special simulator support
        # This is a placeholder for simulation control
        with self._state_lock:
            self._T_nINw = pose_frame
            self._needle_pos = pose_frame.p

    def release(self):
        """Release needle by publishing zero forces and torques in single message"""
        wrench_msg = Wrench()
        wrench_msg.force.x = 0.0
        wrench_msg.force.y = 0.0
        wrench_msg.force.z = 0.0
        wrench_msg.torque.x = 0.0
        wrench_msg.torque.y = 0.0
        wrench_msg.torque.z = 0.0
        self._wrench_pub.publish(wrench_msg)

    # Kinematics methods
    def get_tip_pose(self):
        """Get tip pose in world frame"""
        with self._state_lock:
            return self._T_nINw * self.T_tINn

    def get_base_pose(self):
        """Get base pose in world frame"""
        with self._state_lock:
            return self._T_nINw * self.T_bINn

    def get_mid_pose(self):
        """Get mid pose in world frame"""
        with self._state_lock:
            return self._T_nINw * self.T_mINn

    def get_bm_pose(self):
        """Get base-mid center pose in world frame"""
        with self._state_lock:
            return self._T_nINw * self.T_bmINn

    def get_pose_angle(self, angle_degree):
        """Get pose at specified angle on needle"""
        angle_rad = np.deg2rad(angle_degree)
        T_angle = Frame(
            Rotation.RPY(0., 0., -angle_rad),
            Vector(-self.Radius * np.cos(angle_rad), self.Radius * np.sin(angle_rad), 0.) / 10.0
        )
        with self._state_lock:
            return self._T_nINw * T_angle

    def get_interpolated_transforms(self, start_degree=5, end_degree=25, num_points=25):
        """Get interpolated transforms along needle arc"""
        angles = np.linspace(start_degree, end_degree, num_points)
        transforms = []
        with self._state_lock:
            for angle in angles:
                angle_rad = np.deg2rad(angle)
                T_angle = Frame(
                    Rotation.RPY(0., 0., -angle_rad),
                    Vector(-self.Radius * np.cos(angle_rad), self.Radius * np.sin(angle_rad), 0.) / 10.0
                )
                transforms.append(self._T_nINw * T_angle)
        return transforms

    def get_random_grasp_point(self, random_degree=None):
        """Get random grasping point on needle"""
        min_degree = 10
        max_degree = 15
        if random_degree is None:
            random_degree = np.random.uniform(min_degree, max_degree)
        else:
            if not (min_degree <= random_degree <= max_degree):
                raise ValueError("random_degree out of range. Must be between 10 and 15 degrees.")
        
        random_radian = np.deg2rad(random_degree)
        T_randomINn = Frame(
            Rotation.RPY(0., 0., -random_radian),
            Vector(-self.Radius * np.cos(random_radian), self.Radius * np.sin(random_radian), 0.) / 10.0
        )
        
        with self._state_lock:
            return self._T_nINw * T_randomINn


