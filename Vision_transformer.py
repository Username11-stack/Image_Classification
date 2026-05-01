try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[1]) >= 12 or int(torch.__version__.split(".")[0]) == 2, "torch version should be 1.12+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U",
                           "torch", "torchvision", "torchaudio",
                           "--index-url", "https://download.pytorch.org/whl/cu118"])
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

import multiprocessing
import os
from pathlib import Path
from timeit import default_timer as timer

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary

import Config

device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = Path("C:/Users/anchi/GitHub/Repos/Image_Classification/Data_Refined")
train_dir = data_path / "Train"
test_dir = data_path / "Test"

manual_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])

BATCH_SIZE = 32 # this is lower than the ViT paper but it's because we're starting small

# Create data loaders
train_dataloader, test_dataloader, class_names = Config.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=manual_transforms, # use manually created transforms
    batch_size=BATCH_SIZE
)

print(train_dataloader, test_dataloader, class_names)