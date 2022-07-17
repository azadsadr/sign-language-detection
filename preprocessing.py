import numpy as np
import torch
from torch.utils.data import Dataset 
#from torchvision import transforms


class SignDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        
        xy = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, skiprows=1)
        self.images = xy[:, 1:]
        self.labels = xy[:, 0]

        change = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        for i in range(self.images.shape[0]):
            if self.labels[i] in change:
                self.labels[i] -= 1
        
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        
        if torch.is_tensor(index):
            index = index.tolist()
        
        x = self.images[index, :]
        x = x.reshape(1, 28, 28)
        y = self.labels[index]
        y = np.array(y)
        sample = x, y

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class ToTensor:
    def __call__(self, sample):
        images, labels = sample
        return torch.from_numpy(images), torch.from_numpy(labels).type(torch.LongTensor)
