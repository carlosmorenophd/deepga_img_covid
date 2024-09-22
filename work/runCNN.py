import torch
import torch.nn as nn
import torch.nn.functional as F          # adds some efficiency
from torch.utils.data import Dataset, DataLoader  # lets us load data in batches
from torchvision import datasets, transforms
import h5py


import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix  # for evaluating results
import matplotlib.pyplot as plt
import os
# %matplotlib inline


class CustomDataset(Dataset):
    def __init__(self, base_dir, h5_file, transform=None):
        self.h5_file = h5py.File(os.path.join(base_dir, h5_file), 'r')
        self.transform = transform

    def __len__(self):
        return len(self.h5_file['images'])

    def __getitem__(self, idx):
        image = self.h5_file['images'][idx]
        label = self.h5_file['labels'][idx]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.ToTensor()

print(os.getcwd())
base_path = 'work/Data'


train_dataset = CustomDataset(
    base_dir=base_path, h5_file='test_training_covid_img.h5', transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True)

test_dataset = CustomDataset(
    base_dir=base_path, h5_file='test_testing_covid_img.h5', transform=transform)
test_loader = DataLoader(test_dataset, shuffle=True)
train_dataset
conv1 = nn.Conv2d(1, 6, 3, 1)
conv2 = nn.Conv2d(6, 16, 3, 1)

train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img[0,:,:])
plt.show()
print(f"Label: {label}")
