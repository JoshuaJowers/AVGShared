import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import glob
import PIL
from PIL import Image
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
import random
from RealGeographicCustomDataset import AerialImageDataset
from torch.utils.data import random_split


batch_size = 16
validation_ratio = 0.1
random_seed = 10
csv_file = 'C:/Users/jower/miniconda3/envs/AVG/Images/multiplesourcelabelsmain.csv'
root_dir = 'C:/Users/jower/miniconda3/envs/AVG/Images/jpgs'
all_transforms = transforms.Compose([transforms.ToTensor()])

full_dataset = AerialImageDataset(csv_file, root_dir, all_transforms)
train_dataset, test_dataset, valid_dataset = random_split(full_dataset, [2000,1000,1000])
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

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                          shuffle=True, num_workers=0)

num_train = len(train_dataset)
indices = list(range(num_train))
split = int(np.floor(validation_ratio * num_train))
np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)



classes = ('ADOP2006', 'ADOP2017')

initial_lr = 0.045

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class Xception(nn.Module):
    def __init__(self, input_channel, num_classes=2):
        super(Xception, self).__init__()

        # Entry Flow
        self.entry_flow_1 = nn.Sequential(

            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.entry_flow_2 = nn.Sequential(
            depthwise_separable_conv(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            depthwise_separable_conv(128, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_2_residual = nn.Conv2d(64, 128, kernel_size=1, stride=2, padding=0)

        self.entry_flow_3 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(128, 256, 3, 1),
            nn.BatchNorm2d(256),

            nn.ReLU(True),
            depthwise_separable_conv(256, 256, 3, 1),
            nn.BatchNorm2d(256),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_3_residual = nn.Conv2d(128, 256, kernel_size=1, stride=2, padding=0)

        self.entry_flow_4 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(256, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.entry_flow_4_residual = nn.Conv2d(256, 728, kernel_size=1, stride=2, padding=0)

        # Middle Flow
        self.middle_flow = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728)
        )

        # Exit Flow
        self.exit_flow_1 = nn.Sequential(
            nn.ReLU(True),
            depthwise_separable_conv(728, 728, 3, 1),
            nn.BatchNorm2d(728),

            nn.ReLU(True),
            depthwise_separable_conv(728, 1024, 3, 1),
            nn.BatchNorm2d(1024),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.exit_flow_1_residual = nn.Conv2d(728, 1024, kernel_size=1, stride=2, padding=0)
        self.exit_flow_2 = nn.Sequential(
            depthwise_separable_conv(1024, 1536, 3, 1),
            nn.BatchNorm2d(1536),
            nn.ReLU(True),

            depthwise_separable_conv(1536, 2048, 3, 1),
            nn.BatchNorm2d(2048),
            nn.ReLU(True)
        )

        self.linear = nn.Linear(2048, num_classes)

    def forward(self, x):
        entry_out1 = self.entry_flow_1(x)
        entry_out2 = self.entry_flow_2(entry_out1) + self.entry_flow_2_residual(entry_out1)
        entry_out3 = self.entry_flow_3(entry_out2) + self.entry_flow_3_residual(entry_out2)
        entry_out = self.entry_flow_4(entry_out3) + self.entry_flow_4_residual(entry_out3)

        middle_out = self.middle_flow(entry_out) + entry_out

        for i in range(7):
            middle_out = self.middle_flow(middle_out) + middle_out

        exit_out1 = self.exit_flow_1(middle_out) + self.exit_flow_1_residual(middle_out)
        exit_out2 = self.exit_flow_2(exit_out1)

        exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)

        output = self.linear(exit_avg_pool_flat)

        return output


net = Xception(3, 2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=0.9)

for epoch in range(3):
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
        """
        running_loss += loss.item()

        show_period = 250
        if i % show_period == show_period-1:    # print every "show_period" mini-batches
            print('[%d, %5d] loss: %.7f' %
                  (epoch + 1, i + 1, running_loss / show_period))
            running_loss = 0.0
        """

    # validation part
    correct = 0
    total = 0
    print(len(valid_loader))
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