import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from cv_bridge import CvBridge
from ros_abstraction_layer import ral
from sensor_msgs.msg import Image as RosImage
import time
from r3m import load_r3m
import os
import sys
from PyKDL import Vector, Frame, Rotation
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../RL"))
from utils.real_psm_arm import PSM
from utils.utils import convert_mat_to_vector

task_name = "Approach"
view_name = "front"

ral_instance = ral("run_model_node")
ral_instance.spin()  # Start RAL spinning to process callbacks
time.sleep(0.5)  # Allow some time for RAL to initialize


if task_name == "Approach" or task_name == "Regrasp":
    is_grasp = False
else:
    is_grasp = True

print(f"is grasp: {is_grasp}")

r3m_model = load_r3m("resnet50")
r3m_model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
r3m_model.to(device)

class BehaviorCloningModel(nn.Module):
    def __init__(self, r3m):
        super(BehaviorCloningModel, self).__init__()
        self.r3m = r3m
        self.regressor = nn.Sequential(
            nn.BatchNorm1d(2048 + 7),
            nn.Linear(2048 + 7, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
            nn.Tanh()
        ).to(device)

    def forward(self, x, proprioceptive_data):
        with torch.no_grad():
            visual_features = self.r3m(x)
        combined_input = torch.cat((visual_features, proprioceptive_data), dim=1)
        return self.regressor(combined_input)

def load_r3m_model(model_path, r3m):
    model = BehaviorCloningModel(r3m).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_action(model, image_np, proprio_data):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image_np).unsqueeze(0).to(device)
    proprioceptive_tensor = torch.tensor(proprio_data, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predicted_action = model(image, proprioceptive_tensor)
    return predicted_action.cpu().numpy().astype(np.float64)

current_images = {}
image_received = {}
bridge = CvBridge()

def image_callback(msg, camera_id="front"):
    global current_images, image_received
    try:
        current_images[camera_id] = bridge.imgmsg_to_cv2(msg, "bgr8")
        image_received[camera_id] = True
    except Exception as e:
        pass

camera_topics = {
    view_name: f'/jhu_daVinci/right/image_raw' }

for cam_id, topic in camera_topics.items():
    ral_instance.subscriber(topic, RosImage, image_callback)
    image_received[cam_id] = False

def wait_for_images():
    rate = ral_instance.create_rate(100)
    while not all(image_received.values()) and not ral_instance.is_shutdown():
        rate.sleep()
    for key in image_received:
        image_received[key] = False

model_path = f'/home/xsun97/SurgicAI/Image_IL/Approach/base_env/Model/model_final.pth'
model = load_r3m_model(model_path, r3m_model)

trans_step = 1.0e-3
angle_step = np.deg2rad(3)
step_size = np.array([trans_step, trans_step, trans_step, angle_step, angle_step, angle_step,0.0], dtype=np.float32)

max_timesteps = 100

psm_idx = 2
psm = PSM(ral_instance, f"psm_{psm_idx}", env="dVRK")
goal_rotation_cache = None
goal_vector = np.zeros(6) 

for t in range(max_timesteps):

    wait_for_images()
    
    # Get proprioceptive data from PSM
    while psm.measured_cp() is None and not ral_instance.is_shutdown():
        print("Waiting for /PSM2/measured_cp ...")
        time.sleep(0.05)
    measured_pose = psm.measured_cp()
    measured_pose = np.append(measured_pose,0.0).astype(np.float64)
    print(f"measured pose: {measured_pose}")
    action = predict_action(model, current_images['front'], measured_pose).squeeze()
    action[0:3] = action[0:3] + np.random.uniform(-0.1, 0.1, size=action[0:3].shape)

    print(f"Predicted action: {action}")

    # Only update goal if action is non-zero
    if np.any(action != 0):
        goal_vector = measured_pose.copy()
        
        # On first action, cache the rotation; on subsequent actions, reuse it
        if goal_rotation_cache is None:
            goal_rotation_cache = goal_vector[3:6].copy()
        else:
            goal_vector[3:6] = goal_rotation_cache
        
        # Apply action directly to vector 
        action_step = action * step_size
        goal_vector = goal_vector + action_step
        
    X = goal_vector[0]
    Y = goal_vector[1]
    Z = goal_vector[2]
    Roll = goal_vector[3]
    Pitch = goal_vector[4]
    Yaw = goal_vector[5]
    
    goal_cp = np.array([X, Y, Z, Roll, Pitch, Yaw])
    print(f"Goal CP: {goal_cp}")
    # psm.servo_cp(goal_cp)
    
    time.sleep(0.1)  # Small delay between steps
    
    