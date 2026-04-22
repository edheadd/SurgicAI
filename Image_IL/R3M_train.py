import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pickle
import os
import re
import wandb
import argparse
from r3m import load_r3m
import gc
from pathlib import Path
import sys

from RL.utils.cli_args import add_common_logging_args
from RL.utils.logging_utils import get_logger, setup_logging
from RL.utils.seed import seed_everything

parser = argparse.ArgumentParser(description='Behavior Cloning Training')
parser.add_argument('--task_name', type=str, required=True, help='Name of the task')
parser.add_argument('--view_name', type=str, required=True, help='Name of the view')
parser.add_argument('--seed', type=int, default=10, help='Random seed')
parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
parser.add_argument('--wandb_project', type=str, default="behavior_cloning_v2", help='W&B project name')
parser.add_argument('--wandb_name', type=str, default=None, help='Optional W&B run name override')
add_common_logging_args(parser)
args = parser.parse_args()

task_name = args.task_name
view_name = args.view_name

logger = get_logger(__name__)
setup_logging(level=args.log_level, log_file=args.log_file)
seed_everything(args.seed)

gc.collect()
torch.cuda.empty_cache()
visDR=True
stepDR=True

# Resolve repo-local paths (no hardcoded /home/...).
_REPO_ROOT = Path(os.environ.get("SURGICAI_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
_RL_DIR = _REPO_ROOT / "RL"

# Data location can be overridden externally.
_IL_DATA_DIR = Path(os.environ.get("SURGICAI_IL_DATA_DIR", "")).expanduser().resolve() if os.environ.get("SURGICAI_IL_DATA_DIR") else None
data_dir = str(_IL_DATA_DIR / task_name) if _IL_DATA_DIR is not None else str(_RL_DIR / f"{task_name}_td3" / "vis_dr" / "TransitionEps")

# Model outputs live under the repo by default; override with SURGICAI_IL_OUT_DIR.
_IL_OUT_DIR = Path(os.environ.get("SURGICAI_IL_OUT_DIR", "")).expanduser().resolve() if os.environ.get("SURGICAI_IL_OUT_DIR") else (_REPO_ROOT / "Image_IL")
model_save_dir = str(_IL_OUT_DIR / task_name / "vis_dr" / "Model")

os.makedirs(model_save_dir, exist_ok=True)

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
        #with torch.no_grad():
        visual_features = self.r3m(x)
        combined_input = torch.cat((visual_features, proprioceptive_data), dim=1)
        return self.regressor(combined_input)

bc_model = BehaviorCloningModel(r3m_model)

class PickleDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.1, scale=(0.02,0.1)),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        
        for f in os.listdir(data_dir):
            if re.match(r'episode_\d+\.pkl$', f):
                file_path = os.path.join(data_dir, f)
                with open(file_path, 'rb') as file:
                    trajectory = pickle.load(file)
                    self.data.extend(trajectory)
        
        print(f"Total number of data points: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        step = self.data[idx]
        
        image_data = step['images'][view_name]
        img = Image.fromarray(image_data.astype('uint8'), 'RGB') if isinstance(image_data, np.ndarray) else Image.open(image_data).convert('RGB')
        img = self.transform(img)
        
        proprioceptive = torch.tensor(step['obs']['observation'][0:7], dtype=torch.float32)
        action = torch.tensor(step['action'], dtype=torch.float32)
        
        return img, action, proprioceptive

dataset = PickleDataset(data_dir)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=4)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(bc_model.parameters(), lr=0.0001)

wandb.init(project="behavior_cloning_v2", name=f"{task_name}_{view_name}_view")
if args.use_wandb:
    wandb.init(project=args.wandb_project, name=(args.wandb_name or f"{task_name}_{view_name}_view"))
    wandb.config.update(vars(args))

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for i, (images, actions, proprio_data) in enumerate(train_loader):
            images, actions, proprio_data = images.to(device), actions.to(device), proprio_data.to(device)
            optimizer.zero_grad()
            outputs = model(images, proprio_data)
            loss = criterion(outputs, actions)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        if args.use_wandb:
            wandb.log({"Train Loss": avg_train_loss}, step=epoch)
        
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for images, actions, proprio_data in test_loader:
                images, actions, proprio_data = images.to(device), actions.to(device), proprio_data.to(device)
                outputs = model(images, proprio_data)
                loss = criterion(outputs, actions)
                total_test_loss += loss.item()
            avg_test_loss = total_test_loss / len(test_loader)
            if args.use_wandb:
                wandb.log({"Test Loss": avg_test_loss}, step=epoch)
            logger.info("Epoch %s: train=%0.4f test=%0.4f", epoch + 1, avg_train_loss, avg_test_loss)

        if (epoch + 1) % checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            torch.save(model.state_dict(), os.path.join(model_save_dir, f'model_epoch_{epoch+1}.pth'))
            logger.info("Saved model checkpoint at epoch %s", epoch + 1)

    torch.save(model.state_dict(), os.path.join(model_save_dir, 'model_final.pth'))
    logger.info("Final model saved")

num_epochs = 40
checkpoint_interval = 20
train_and_evaluate(bc_model, train_loader, test_loader, criterion, optimizer, num_epochs, checkpoint_interval)

if args.use_wandb:
    wandb.finish()