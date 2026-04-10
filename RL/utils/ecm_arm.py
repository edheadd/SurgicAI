#!/usr/bin/env python
# ECM Arm Controller using RAL
import numpy as np
import time
from threading import Thread, Lock
from PyKDL import Frame, Rotation, Vector, Twist
from ambf_msgs.msg import RigidBodyState
from copy import deepcopy


class ECM:
    """ECM Arm controller using RAL for ROS communication"""
    
    def __init__(self, ral_instance, name):
        self.ral = ral_instance
        self.name = name
        self._state_lock = Lock()
        
        # Pose tracking (stored as PyKDL Frames)
        self._T_c_w = Frame()
        self._T_c_w_init = None
        self._measured_cp = Frame()
        self._max_vel = 0.002
        
        # Command state
        self._T_cmd = Frame()
        self._force_exit_thread = False
        self._thread_busy = False
        
        # Setup ROS interface
        self._setup_ros_interface()
        time.sleep(0.5)
        
        # Store initial pose
        self._T_c_w_init = self.measured_cp()

    def _setup_ros_interface(self):
        """Setup ROS subscribers and publishers for camera"""
        self.ral.subscriber(
            f'/ambf/env/cameras/{self.name}/State',
            RigidBodyState,
            self._state_callback,
            queue_size=1
        )

    def _state_callback(self, msg):
        """Callback for state updates from AMBF"""
        with self._state_lock:
            if msg.pose is not None:
                self._T_c_w = self._pose_msg_to_frame(msg.pose)

    def _pose_msg_to_frame(self, pose_msg):
        """Convert geometry_msgs/Pose to PyKDL Frame"""
        if pose_msg is None:
            return Frame()
        
        p = Vector(pose_msg.position.x, pose_msg.position.y, pose_msg.position.z)
        R = Rotation.Quaternion(
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        )
        return Frame(R, p)

    def measured_cp(self):
        """Get the measured Cartesian pose"""
        with self._state_lock:
            return deepcopy(self._T_c_w)

    def get_T_c_w(self):
        """Get Transform of Camera in World"""
        return self.measured_cp()

    def get_T_w_c(self):
        """Get Transform of World in Camera (inverse)"""
        return self.measured_cp().Inverse()

    def servo_cp(self, T_c_w):
        """Servo to a Cartesian pose (direct command)"""
        if isinstance(T_c_w, (np.ndarray, np.matrix)):
            T_c_w = self._numpy_to_frame(T_c_w)
        
        with self._state_lock:
            self._T_cmd = T_c_w

    def _numpy_to_frame(self, mat):
        """Convert numpy matrix to PyKDL Frame"""
        frame = Frame(Rotation.RPY(0, 0, 0), Vector(0, 0, 0))
        for i in range(3):
            for j in range(3):
                frame.M[i, j] = mat[i, j]
        for i in range(3):
            frame.p[i] = mat[i, 3]
        return frame

