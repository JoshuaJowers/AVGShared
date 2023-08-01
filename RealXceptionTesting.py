import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from Main.RealGeographicCustomDataset import AerialImageDataset
from torch.utils.data import random_split
from XceptionModel import Xception
from math import sqrt

# All of these number are currently arbitrary
batch_size = 5
validation_ratio = 0.1
random_seed = 10
csv_file = 'C:/Users/jower/miniconda3/envs/AVG/Images/multiplesourcelabels.csv'
root_dir = 'C:/Users/jower/miniconda3/envs/AVG/Images/jpgs'
full_dataset = AerialImageDataset(csv_file, root_dir)
dataset_size = full_dataset.__len__()
train_length = round(dataset_size / 2)
valid_length = round(dataset_size / 4)
print(dataset_size)
test_length = dataset_size - train_length - valid_length

# Splits dataset into 3 sets, with the training datset currently
train_dataset, test_dataset, valid_dataset = random_split(full_dataset, [train_length, valid_length, test_length])
# Instantiate loader objects to facilitate processing
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

#Change to RMSE
# Loss function that takes in the means and standard deviations
import torch
from torch.distributions import Normal

import torch


import torch

def nll_loss(predictions, targets):
    mean_x = predictions[:,0].clone().requires_grad_(True)
    loss = torch.abs((targets[:, 0] - mean_x))
    print(loss)
    print(torch.mean(loss))
    return torch.mean(loss)

def straight_line_loss(predictions, targets):
    mean_x = predictions[:, 0].clone().requires_grad_(True)
    mean_y = predictions[:, 2].clone().requires_grad_(True)
    std_x = torch.abs(predictions[:, 1].clone().requires_grad_(True))
    std_y = torch.abs(predictions[:, 3].clone().requires_grad_(True))

    # Define normal distributions for x and y
    dist_x = Normal(mean_x, std_x)
    dist_y = Normal(mean_y, std_y)

    # Reparameterization trick for sampling
    epsilon_x = torch.randn_like(std_x)
    epsilon_y = torch.randn_like(std_y)
    pred_x = mean_x + std_x * epsilon_x
    pred_y = mean_y + std_y * epsilon_y

    target_x = targets[:, 0]
    target_y = targets[:, 1]

    loss_x = target_x - pred_x
    loss_y = target_y - pred_y

    squared_sum = torch.pow(loss_x, 2) + torch.pow(loss_y, 2)
    loss = torch.sqrt(squared_sum).mean()
    return loss

def nonsamplingxyloss(predictions, targets):
    mean_x = predictions[:, 0].clone().requires_grad_(True)
    mean_y = predictions[:, 2].clone().requires_grad_(True)
    std_x = torch.abs(predictions[:, 1].clone().requires_grad_(True))
    std_y = torch.abs(predictions[:, 3].clone().requires_grad_(True))

    target_x = targets[:, 0]
    target_y = targets[:, 1]

    loss_x = target_x - mean_x
    loss_y = target_y - mean_y

    squared_sum = torch.pow(loss_x, 2) + torch.pow(loss_y, 2)
    loss = torch.sqrt(squared_sum).mean()

    return loss
# These are the classes defined for having it as a classification CNN
classes = ('ADOP2006', 'ADOP2017')
epochs = 1000
# Arbitrary

net = Xception(3, 4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
# Defines Loss function and optimizer
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# Runs Training Loop
i = 0
for epoch in range(epochs):
    running_loss = 0.0
    i = i + 1
    print("Epoch Number")
    print(i)
    print(train_loader.__len__())
    for batch_idx, (images, targets) in enumerate(train_loader):
        # Zero the gradients


        # Forward pass
        output = net(images)
        mean_x = output[0].clone().requires_grad_(True)

        #loss = nll_loss(mean_x.squeeze(), targets)
        loss = nonsamplingxyloss(output, targets)
        # print("Targets 1 = ")
        # print(targets)
        # print("Targets = ")
        # print(targets)
        # print("Output = ")
        # print(output)
        #
        # print("Loss = ")
        # print(loss)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        # for name, param in net.named_parameters():
        #     if param.grad is not None:
        #         print(f'{name}: {param.grad.norm()}')
        #     else:
        #         print(f'{name}: None')

        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 1300 == 1299:  # Print every 20 batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 1300))
            running_loss = 0.0
            # if epoch % 5 == 1:
            #     print("Predicted X = " + str(mean_x))
            #     print("Predicted Y = " + str(mean_y))

    # Validation loop
    with torch.no_grad():
        net.eval()
        val_loss = 0.0
        for images, targets in valid_loader:
            output = net(images)
            loss = nonsamplingxyloss(output, targets)
            val_loss += loss.item()

        # Report validation loss
        print('Validation Loss: %.3f' % (val_loss / len(valid_loader)))

    net.train()

print('Finished Training')
