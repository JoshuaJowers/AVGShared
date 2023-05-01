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
batch_size = 10
validation_ratio = 0.1
random_seed = 10
csv_file = 'C:/Users/jower/miniconda3/envs/AVG/Images/multiplesourcelabelsmain.csv'
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

def nll_loss(mean_x, std_x, mean_y, std_y, target):
    dist_x = Normal(mean_x, std_x)
    dist_y = Normal(mean_y, std_y)
    diff_x = target[:, 0] - mean_x
    diff_y = target[:, 1] - mean_y
    loss = torch.sqrt(diff_x ** 2 + diff_y ** 2)
    return torch.mean(loss)



# These are the classes defined for having it as a classification CNN
classes = ('ADOP2006', 'ADOP2017')
epochs = 100
# Arbitrary
initial_lr = 0.045

net = Xception(3, 4)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
# Defines Loss function and optimizer
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
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
        optimizer.zero_grad()

        # Forward pass
        output = net(images)
        mean_x = output[0].clone().detach().requires_grad_(True)
        std_x = output[1].clone().detach().requires_grad_(True)
        mean_y = output[2].clone().detach().requires_grad_(True)
        std_y = output[3].clone().detach().requires_grad_(True)

        loss = nll_loss(mean_x.squeeze(), std_x.squeeze(), mean_y.squeeze(), std_y.squeeze(), targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch_idx % 20 == 19:  # Print every 20 batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / 20))
            running_loss = 0.0
            if epoch % 5 == 1:
                print("Predicted X = " + str(mean_x))
                print("Predicted Y = " + str(mean_y))

    # Validation loop
    with torch.no_grad():
        net.eval()
        val_loss = 0.0
        for images, targets in valid_loader:
            output = net(images)
            mean_x = output[0].clone().detach().requires_grad_(True)
            std_x = output[1].clone().detach().requires_grad_(True)
            mean_y = output[2].clone().detach().requires_grad_(True)
            std_y = output[3].clone().detach().requires_grad_(True)
            loss = nll_loss(mean_x.squeeze(), std_x.squeeze(), mean_y.squeeze(), std_y.squeeze(), targets)
            val_loss += loss.item()

        # Report validation loss
        print('Validation Loss: %.3f' % (val_loss / len(valid_loader)))

    net.train()

print('Finished Training')
