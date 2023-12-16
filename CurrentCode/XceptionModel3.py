import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from RealGeographicCustomDataset3 import AerialImageDataset
from torch.utils.data import random_split


#Num Classes = 4 because independent normal distribution outputs st dev and mean for x and y locations
#Defining depthwise separable layer to be using within the Xception architecture
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

# Main xception method, layers taken from xception diagram. Here is the documentation https://arxiv.org/abs/1610.02357
# I have not messed with any of the input values for any of the layers
class Xception(nn.Module):
    #Num classes = 4 because of 2 mean and 2 standard deviations for bivariate independent normal distribution
    def __init__(self, input_channel, num_classes=4):
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

        self.linear = nn.Linear(2048, 4)

    def forward(self, x):
        batch_size = x.size(0)
        entry_out1 = self.entry_flow_1(x)
        entry_out2 = self.entry_flow_2(entry_out1) + self.entry_flow_2_residual(entry_out1)
        entry_out3 = self.entry_flow_3(entry_out2) + self.entry_flow_3_residual(entry_out2)
        entry_out = self.entry_flow_4(entry_out3) + self.entry_flow_4_residual(entry_out3)

        middle_out = self.middle_flow(entry_out) + entry_out

        for i in range(7):
            middle_out = self.middle_flow(middle_out) + middle_out

        # exit_out1 = self.exit_flow_1(middle_out) + self.exit_flow_1_residual(middle_out)
        # exit_out2 = self.exit_flow_2(exit_out1)
        #
        # exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))
        # exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)
        #
        # output = self.linear(exit_avg_pool_flat)
        # mean_x = output[:, :1]
        # std_x = nn.functional.softplus(output[:, 2])  # Use softplus activation to ensure positive standard deviation
        # mean_y = output[:, 2:3]
        # std_y = nn.functional.softplus(output[:, 3])
        # return mean_x, std_x, mean_y, std_y
        exit_out1 = self.exit_flow_1(middle_out) + self.exit_flow_1_residual(middle_out)
        exit_out2 = self.exit_flow_2(exit_out1)

        exit_avg_pool = F.adaptive_avg_pool2d(exit_out2, (1, 1))
        exit_avg_pool_flat = exit_avg_pool.view(exit_avg_pool.size(0), -1)

        output = self.linear(exit_avg_pool_flat)
        return output
