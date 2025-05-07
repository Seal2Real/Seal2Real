import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision.transforms as transforms
import random

class RealDataset(Dataset):
    def __init__(self, dir_train, transform=None):
        self.dir_train = "/placeholder/path/to/real_data"
        self.transform = transform
        self.image_paths = []
        for root, _, files in os.walk(self.dir_train):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            image = transform(image)
        return image

class FakeDataset(Dataset):
    def __init__(self, dir_train="/placeholder/path/to/fake_data", transform=None):
        self.dir_train = dir_train
        self.transform = transform
        self.image_paths = []
        self.label = []
        for root, _, files in os.walk(self.dir_train):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
                    if "real" in root:
                        self.label.append(torch.tensor(1))
                    else:
                        self.label.append(torch.tensor(0))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((512,512)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            image = transform(image)
        return image, label


