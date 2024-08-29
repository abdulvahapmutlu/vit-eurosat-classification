import torch
import timm
import torch.nn as nn
import torch.optim as optim
import tqdm as tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
import wandb
import warnings
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
# Initialize Weights & Biases
wandb.init(project="visiontransformer")# Data Augmentation and Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Dataset
dataset = datasets.ImageFolder(root=r'C:\Users\offic\OneDrive\Masaüstü\datasets\EuroSAT', transform=transform)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Mixup Augmentation
mixup = Mixup(mixup_alpha=0.2)  # Adjust alpha as needed
