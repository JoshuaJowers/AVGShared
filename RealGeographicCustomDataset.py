#This is the custom dataset that I wrote based on Rafael's Code and the Pytorch custom datsets tutorial
#imports the image_crop_map method from Cropping Methods
import random
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import os
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, random_split
from torchvision.transforms import transforms

from CroppingMethods import image_crop_map


#Basic method to print out an image, used only for debugging purposes, works on numpy.ndarray image
def show_image(image):
    plt.imshow(image)
    plt.pause(0.001)


class AerialImageDataset(Dataset):
    #Init function takes in csv_file with headers filename, x, y, root_dir is the filepath to the folder containg the
    #images, transform is used to convert image to tensor
    def __init__(self, csv_file, root_dir, transform=transforms.ToTensor()):

        self.image_data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        scaler = MinMaxScaler(feature_range=(0, 1))
        self.image_data[['x', 'y']] = scaler.fit_transform(self.image_data[['x', 'y']])
        #These were used to determine scale size to be applied when adjust for cropping
        # scalexparam = scaler.scale_[0]
        # scaleyparam = scaler.scale_[1]

    def __len__(self):
        return len(self.image_data)

    #Gets and returns image with x and y labels, but currently doesn't do x and y labels, and instead a random binary
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
        #These are the new, adjusted x and y labels after the crop is performed
        xCoord = labels[0]
        yCoord = labels[1]
        number = random.randint(0,1)
        sample = (image, number)
        #sample = (image, xCoord, yCoord) Commented out until I know how to use the x and y coordinates
        return sample

csv_file = 'C:/Users/jower/miniconda3/envs/AVG/Images/multiplesourcelabelsmain.csv'
root_dir = 'C:/Users/jower/miniconda3/envs/AVG/Images/jpgs'

