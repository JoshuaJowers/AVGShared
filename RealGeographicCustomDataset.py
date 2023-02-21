import random

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, transform
import os
from sklearn.preprocessing import MinMaxScaler
import sys
from torch.utils.data import Dataset, random_split
from torch.utils.data import DataLoader
from torch.nn.functional import relu
from torch import sigmoid
from torch import flatten


scalexparam = 2.9249173710842667e-05
scaleyparam = 2.923720141508055e-05

def random_crop_single(img, random_crop_size, img_real_size):
    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]

    dy, dx = random_crop_size
    # Choose a random point in the valid area to be the new location, then crop
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :], (float(x) / img.shape[1]) * img_real_size, \
                                           (-float(y) / img.shape[0]) * img_real_size


def image_crop_map(image, labels):
    image_crop, x_shift, y_shift = random_crop_single(image, (200, 200), 3000)
    # Combine shifts into one numpy array
    # We must scale the shifts to match the scale of the labels
    x_shift *= scalexparam
    # The scaling is inverted in y direction since north is positive in the coordinate system but going up is negative
    y_shift *= -scaleyparam
    test1 = 0.99
    test2 = 0.99
    return_labels = torch.tensor([0.9,0.5])
    # return_labels[0] = torch.add(labels[0], x_shift)
    # return_labels[1] = torch.add(labels[1], y_shift)
    return_labels = torch.reshape(return_labels, [2, ])

    return image_crop, return_labels
def show_image(image):
    """Show image with landmarks"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


class AerialImageDataset(Dataset):
    """Image Location Dataset."""

    def __init__(self, csv_file, root_dir, transform):
        """
        Args:
            csv_file (string): Path to the csv file with Filenames and x and y locations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.image_data[['x', 'y']] = scaler.fit_transform(self.image_data[['x', 'y']])
        scalexparam = scaler.scale_[0]
        scaleyparam = scaler.scale_[1]

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.image_data.iloc[idx, 0])
        image = io.imread(img_name)


        xCoord = self.image_data.iloc[idx, 1]
        yCoord = self.image_data.iloc[idx, 2]
        if self.transform:
            image = self.transform(image)
        image = np.transpose(image, (1, 2, 0))
        image, labels = image_crop_map(image, torch.tensor([xCoord, yCoord]))
        image = np.transpose(image, (2,0,1))
        xCoord = labels[0]
        yCoord = labels[1]
        number = random.randint(0,1)
        sample = (image, number)
        #sample = (image, xCoord, yCoord) Commented out until I know how to use the x and y coordinates
        return sample

csv_file = 'C:/Users/jower/miniconda3/envs/AVG/Images/multiplesourcelabelsmain.csv'
root_dir = 'C:/Users/jower/miniconda3/envs/AVG/Images/jpgs'
