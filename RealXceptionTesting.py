import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from Main.RealGeographicCustomDataset import AerialImageDataset
from torch.utils.data import random_split
from XceptionModel import Xception

#All of these number are currently arbitrary
batch_size = 16
validation_ratio = 0.1
random_seed = 10
csv_file = 'C:/Users/jower/miniconda3/envs/AVG/Images/multiplesourcelabelsmain.csv'
root_dir = 'C:/Users/jower/miniconda3/envs/AVG/Images/jpgs'
full_dataset = AerialImageDataset(csv_file, root_dir)
dataset_size = full_dataset.__len__()
train_length = round(dataset_size / 2)
valid_length = round(dataset_size / 4)
test_length = dataset_size - train_length - valid_length

#Splits dataset into 3 sets, with the training datset currently
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




#These are the classes defined for having it as a classification CNN
classes = ('ADOP2006', 'ADOP2017')
epochs = 3
#Arbitrary
initial_lr = 0.045

net = Xception(3, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net.to(device)
#Defines Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)

#Runs Training Loop
for epoch in range(epochs):
    print("Starting epoch: ")
    print(epoch)
    if epoch == 0:
        lr = initial_lr
    elif epoch % 2 == 0 and epoch != 0:
        lr *= 0.94
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    running_loss = 0.0
    print(len(train_loader))
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        show_period = 250
        if i % show_period == show_period-1:    # print every "show_period" mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss / show_period))
            running_loss = 0.0

    # Validation
    correct = 0
    total = 0
    for j, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('[%d epoch] Accuracy of the network on the validation images: %d %%' %
          (epoch, 100 * correct / total)
          )

print('Finished Training')