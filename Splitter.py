import os
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
import tabulate
import shutil

base_path = "C:/Users/anchi/GitHub/Repos/Image_Classification"
data_path = os.path.join(base_path, "Data/Training_set.csv")
df = pd.read_csv(data_path)
print(tabulate.tabulate(df.head(), headers='keys', tablefmt='psql'))

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

from shutil import copyfile

for index, row in train_df.iterrows():
    image_name = row["filename"]
    label = row["label"]
    source_path = os.path.join(base_path, "Data/train", image_name)
    target_dir = os.path.join(base_path, "Data_Refined/Train", label)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, image_name)
    copyfile(source_path, target_path)

for index, row in test_df.iterrows():
    image_name = row["filename"]
    label = row["label"]
    source_path = os.path.join(base_path, "Data/train", image_name)
    target_dir = os.path.join(base_path, "Data_Refined/Test", label)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, image_name)
    copyfile(source_path, target_path)