import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import torchvision


class ASLDataset(Dataset):

    def __init__(self, csv_file, transform=None, target_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform and target_trandform (callable, optional): Optional transform to be applied on a data.

        Note: ToTensor() Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a 
        torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
        if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or 
        if the numpy.ndarray has dtype = np.uint8

        """

        # read as pandas dataframe and convert it to numpy -------------------------------
        data = pd.read_csv(csv_file) # pandas dataframe
        #self.images = data.iloc[:, 1:].to_numpy(dtype = 'float32')
        self.images = data.iloc[:, 1:].to_numpy(dtype = 'uint8')
        self.labels = data['label'].values

        # read as numpy array
        #data = np.loadtxt(csv_file, delimiter=",", dtype=np.float32, skiprows=1)
        #self.images = data[:, 1:]
        #self.labels = data[:, 0]

        change = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        for i in range(self.images.shape[0]):
            if self.labels[i] in change:
                self.labels[i] -= 1

        # convert to tensor --------------------------------------------------------------
        #self.images = torch.from_numpy(images)
        #self.labels = torch.from_numpy(labels)
        #self.labels = torch.from_numpy(labels).type(torch.LongTensor)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.images[idx, :]
        image = image.reshape(28, 28, 1)    # ToTensor() Converts (H x W x C) to the shape of (C x H x W)
        label = self.labels[idx]

        #label = np.array(self.labels[idx]).astype('long')
        #image = torch.from_numpy(image)
        #label = torch.from_numpy(label).type(torch.LongTensor)

        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            label = self.target_transform(label)

        return image, label



'''
#---------------------------------------------------------------------
class ToTensor:
    def __call__(self, sample):
        images, labels = sample
        return torch.from_numpy(images), torch.from_numpy(labels).type(torch.LongTensor)


class Normalize:
    def __call__(self, sample):
        images, labels = sample
        mean = torch.mean(images)
        std = torch.std(images)
        images = (images - mean) / std
        return images, labels
#---------------------------------------------------------------------
'''


def make_dataset(train_path, test_path):

    #train_path = r'data/sign_mnist_train.csv'
    #test_path = r'data/sign_mnist_test.csv'

    images_transforms = torchvision.transforms.Compose([
        #preprocessing.ToTensor(), 
        #preprocessing.Normalize(), 
        torchvision.transforms.ToTensor(), 
        torchvision.transforms.RandomHorizontalFlip(), 
        torchvision.transforms.RandomRotation(10), 
        #torchvision.transforms.Normalize(110, 110), 
    ])

    labels_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), 
    ])

    train_data_full = ASLDataset(
        csv_file=train_path, 
        transform=images_transforms, 
        target_transform=None
        )

    test_data = ASLDataset(
        csv_file=test_path, 
        transform=images_transforms, 
        target_transform=None
        )

    val_size = 7455
    train_size = len(train_data_full) - val_size

    train_data, val_data = random_split(train_data_full, [train_size, val_size])

    return train_data, val_data, test_data