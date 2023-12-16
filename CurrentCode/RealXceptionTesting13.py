#In Wint's Code, the images are cropped to 1.5kmx1.5km which is 500x500 pixels which is then downsampled to 224x224(compressed)
#Cache images to save time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from torch.utils import data as D
from torch.utils.data.sampler import SubsetRandomSampler
from RealGeographicCustomDataset3 import AerialImageDataset
from torch.utils.data import random_split, DataLoader
from XceptionModel3 import Xception
from math import sqrt
from torch.distributions import Normal
import math
import time


max_grad_norm = 1.5
start_time = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
output_file_name = "output14.txt"

with open(output_file_name, 'w') as output_file:
    output_file.write("Start Time = " + str(start_time) + "\n")

class CustomDatasetOnDevice(AerialImageDataset):
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)


   def __getitem__(self, index):
       image, target = super().__getitem__(index)
       return image.to(device), target.to(device)



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


# All of these number are currently arbitrary
batch_size = 5
import os

l1_param = 0.0
l2_param = 0.0
# Get the current working directory
current_directory = os.getcwd()


image_folder = os.path.join(current_directory, "Images")


file_name = "multiplesourcelabels.csv"


csv_file = os.path.join(image_folder, file_name)


jpgs = "jpgs"
root_dir = os.path.join(image_folder, jpgs)
csv_file2 = 'C:/Users/jower/miniconda3.1/envs/pythonProject1/Images/multiplesourcelabels.csv'
root_dir2 = 'C:/Users/jower/miniconda3.1/envs/pythonProject1/Images/jpgs'
full_dataset = CustomDatasetOnDevice(csv_file, root_dir)
dataset_size = full_dataset.__len__()
train_length = round(dataset_size * 0.8)
valid_length = round(dataset_size * 0.1)
test_length = dataset_size - train_length - valid_length

print("Dataset Size = ", dataset_size)
train_dataset, test_dataset, valid_dataset = random_split(full_dataset, [train_length, valid_length, test_length])


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)




epochs = 200


net = Xception(3, 4)
checkpoint_path = 'checkpoint5_epoch_200.pth'
print("Device = ", device)
# if(device == 'cpu'):
#     checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
# else:
#     checkpoint = torch.load(checkpoint_path)
# net.load_state_dict(checkpoint)


net.to(device)
# Defines Loss function and optimizer
#criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=l2_param)
# Runs Training Loop
i = 0
for epoch in range(epochs):
   if (epoch + 1) % 5 == 0:
       # Save the model's parameters
       checkpoint_path = f'checkpoint5_epoch_{epoch + 1}.pth'
       torch.save(net.state_dict(), checkpoint_path)
       print(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}')
       with open(output_file_name, 'a') as output_file:
           output_file.write(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}' + "\n")
           print(f'Saved checkpoint at epoch {epoch + 1} to {checkpoint_path}')

   running_loss = 0.0
   i = i + 1
   with open(output_file_name, 'a') as output_file:
        output_file.write("Epoch Number" + str(i) + "\n")
   print("Epoch Number")
   print(i)
   print(train_loader.__len__())
   for batch_idx, (images, targets) in enumerate(train_loader):
       # Zero the gradients




       # Forward pass
       output = net(images)
       l1_loss = torch.tensor(0.0, dtype=torch.float).to(device)
       for param in net.parameters():
            l1_loss += torch.norm(param, p=1)
       mean_x = output[0].clone().requires_grad_(True)
       l1_loss *= l1_param

       loss = straight_line_loss(output,targets)
       total_loss = l1_loss + loss
       # print("Targets 1 = ")



       # Backward pass and optimization
       optimizer.zero_grad()
       total_loss.backward()

       #torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)



       optimizer.step()


       running_loss += loss.item()
       if batch_idx % 1040 == 1039:
       
           print('[%d, %5d] loss: %.3f' %
                 (epoch + 1, batch_idx + 1, running_loss / 1040))
           with open(output_file_name, 'a') as output_file:
               output_file.write('[%d, %5d] loss: %.3f' %
               (epoch + 1, batch_idx + 1, running_loss / 1040) + "\n")
           running_loss = 0.0



   # Validation loop
   with torch.no_grad():
       net.eval()
       val_loss = 0.0
       val_loss2 = 0.0
       if epoch % 10 < 10:
           for images, targets in valid_loader:
               output = net(images)
               loss = straight_line_loss(output, targets)
               val_loss += loss.item()
       else:
            for images, targets in valid_loader:
               output = net(images)
               loss = straight_line_loss(output, targets)
               val_loss += loss.item()


       # Report validation loss
       print('Validation NLL Loss: %.3f' % (val_loss / len(valid_loader)))
       with open(output_file_name, 'a') as output_file:
            output_file.write('Validation SL Loss: %.3f' % (val_loss / len(valid_loader)) + "\n")

   net.train()


print('Finished Training')


