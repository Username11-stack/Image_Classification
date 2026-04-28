import torch
from torch import nn
import requests
import zipfile
import os
import pandas as pd
import tabulate as tb
import sklearn

    
device = "cuda" if torch.cuda.is_available() else "cpu"

data_path = "C:/Users/anchi/GitHub/Repos/Image_Classification/Data/Training_set.csv"
#image = data_path / "images"

df = pd.read_csv(data_path)
print(tb.tabulate(df.head(100), headers='keys', tablefmt='psql'))
print(df["label"].nunique())

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0 
])
