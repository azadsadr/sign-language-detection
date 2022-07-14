import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset 
from torchvision import transforms


class SignLanguageDataset(Dataset):

    def __init__(self, images, labels=None, transform=None):
        """
        Args:
            images (string): Path to the csv file with annotations.
            labels (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()
        
        x = self.images[index, :]
        x = x.reshape(28, 28, 1)
        if self.transform:
            x = self.transform(x)
        
        y = self.labels[index]

        if self.labels is not None:
            return(x, y)
        else:
            return x

def read_from_csv(path):
    train_pd = pd.read_csv(path)
    data_np = np.array(train_pd, dtype=np.float32)
    m, n = data_np.shape
    labels_np = data_np[:, 0]
    images_np = data_np[:, 1:n]
    images_np = images_np / 255.

    return images_np, labels_np


def get_data(path):
    images_np, labels_np = read_from_csv(path)

    my_transforms = transforms.Compose([
        transforms.ToPILImage(mode='F'), 
        transforms.Resize((32,32)), 
        transforms.RandomCrop((28,28)), 
        #transforms.ColorJitter(brightness=0.5), 
        transforms.RandomRotation(degrees=45), 
        transforms.RandomHorizontalFlip(p=0.5), 
        #transforms.RandomVerticalFlip(p=0.05), 
        #transforms.RandomGrayscale(p=0.2), 
        transforms.ToTensor(), 
        #transforms.Normalize(mean=0.5, std=0.5), # (value - mean) / std
    ])

    data = SignLanguageDataset(images_np, labels_np, transform=my_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=data, batch_size=64, shuffle=True)
    return train_loader