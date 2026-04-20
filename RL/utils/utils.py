import re
import numpy as np
from scipy.spatial.transform import Rotation as R
from PyKDL import Frame, Rotation, Vector, dot
import numpy as np
import json
import math

def frame_to_vector(frame):
    """
    Convert a PyKDL.Frame to a 6D vector [x, y, z, roll, pitch, yaw]
    """
    if frame is None:
        return np.zeros(6, dtype=np.float32)
    x = frame.p[0]
    y = frame.p[1]
    z = frame.p[2]
    
    # GetRPY returns (roll, pitch, yaw) - be explicit
    rpy = frame.M.GetRPY()
    roll = rpy[0]
    pitch = rpy[1]
    yaw = rpy[2]
    
    return np.array([x, y, z, roll, pitch, yaw], dtype=np.float32)

def vector_to_frame(vec):
    """
    Convert a 6D vector [x, y, z, roll, pitch, yaw] to a PyKDL.Frame.
    Extra entries (e.g. jaw) are ignored.
    """
    if vec is None:
        return Frame()
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.size < 6:
        raise ValueError(f"vector_to_frame expects at least 6 values, got {v.size}")
    return Frame(
        Rotation.RPY(float(v[3]), float(v[4]), float(v[5])),
        Vector(float(v[0]), float(v[1]), float(v[2]))
    )

def convert_mat_to_frame(mat):
    frame = Frame(Rotation.RPY(0, 0, 0), Vector(0, 0, 0))
    for i in range(3):
        for j in range(3):
            frame[(i, j)] = mat[i, j]

    for i in range(3):
        frame.p[i] = mat[i, 3]

    return frame

def convert_mat_to_vector(mat):
    frame = convert_mat_to_frame(mat)
    return frame_to_vector(frame)

def get_angle(vec_a, vec_b, up_vector=None):
    vec_a.Normalize()
    vec_b.Normalize()
    cross_ab = vec_a * vec_b
    vdot = dot(vec_a, vec_b)
    # print('VDOT', vdot, vec_a, vec_b)
    # Check if the vectors are in the same direction
    if 1.0 - vdot < 0.000001:
        angle = 0.0
        # Or in the opposite direction
    elif 1.0 + vdot < 0.000001:
        angle = np.pi
    else:
        angle = math.acos(vdot)

    if up_vector is not None:
        same_dir = np.sign(dot(cross_ab, up_vector))
        if same_dir < 0.0:
            angle = -angle

    return angle

def load_json_dvrk(file_path:str)->dict:
    '''
    Load json files from dVRK repository
    :param file_path: json file path
    :return: a dictionary with loaded json file content
    '''
    with open(file_path) as f:
        data = f.read()
        data = re.sub("//.*?\n", "", data)
        data = re.sub("/\\*.*?\\*/", "", data)
        obj = data[data.find('{'): data.rfind('}') + 1]
        jsonObj = json.loads(obj)
    return jsonObj