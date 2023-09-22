import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from RealGeographicCustomDataset import AerialImageDataset
from torch.utils.data import random_split, DataLoader
from XceptionModel import Xception
from math import sqrt
from torch.distributions import Normal
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomDatasetOnDevice(AerialImageDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        return image.to(device), target.to(device)


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
# All of these number are currently arbitrary
batch_size = 5
validation_ratio = 0.1
random_seed = 10
import os

# Get the current working directory
current_directory = os.getcwd()

# Define the path to the "Images" folder relative to the current directory
image_folder = os.path.join(current_directory, "Images")

# Specify the file name within the "Images" folder
file_name = "multiplesourcelabels.csv"

# Combine the folder path and file name to create the complete file path
csv_file = os.path.join(image_folder, file_name)

# Now, you can use the 'file_path' variable to access the file
jpgs = "jpgs"
root_dir = os.path.join(image_folder, jpgs)
csv_file2 = 'C:/Users/jower/miniconda3.1/envs/pythonProject1/Images/multiplesourcelabels.csv'
root_dir2 = 'C:/Users/jower/miniconda3.1/envs/pythonProject1/Images/jpgs'
full_dataset = CustomDatasetOnDevice(csv_file, root_dir)
dataset_size = full_dataset.__len__()
train_length = round(dataset_size / 2)
valid_length = round(dataset_size / 4)
test_length = dataset_size - train_length - valid_length

train_dataset, test_dataset, valid_dataset = random_split(full_dataset, [train_length, valid_length, test_length])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

#Change to RMSE


# These are the classes defined for having it as a classification CNN
classes = ('ADOP2006', 'ADOP2017')
epochs = 10
# Arbitrary

net = Xception(3, 4)
checkpoint_path = 'checkpoint3_epoch_10.pth'
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint)

net.to(device)
# Defines Loss function and optimizer
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
# Runs Training Loop
i = 0
for epoch in range(epochs):
    if (epoch + 1) % 5 == 0:
        # Save the model's parameters
        checkpoint_path = f'checkpoint3_epoch_{epoch + 1}.pth'
        torch.save(net.state_dict(), checkpoint_path)
        print(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}')

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
