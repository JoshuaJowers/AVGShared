# These are the methods used within the Custom Dataset (RealGeographicCustomDataset.py) to randomly crop
# a single image and adjust the x and y coordinates based on the crop


import numpy as np
import torch

#These values were found from running the MinMaxScaler on the given dataset. It is the factor that the values
# (in meters) are multiplied by to get them between 0 and 1 after being shifted
scalexparam = 2.9249173710842667e-05
scaleyparam = 2.923720141508055e-05

#This method randomly crops an image taken in as a numpy array of shape (1000,1000,3)
#Ouputs the cropped image and the change in y and x used to get the crop
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

# This method takes in the image and x and y labels and uses the random_crop_single method to get the cropped image
# and shifts for x and y. Then, it adjusts the x and y labels based on the x and y shift and returns the cropped image
# and the adjusted x and y labels
def image_crop_map(image, labels):
    image_crop, x_shift, y_shift = random_crop_single(image, (200, 200), 3000)
    # We must scale the shifts to match the scale of the labels
    x_shift *= scalexparam
    # The scaling is inverted in y direction since north is positive in the coordinate system but going up is negative
    y_shift *= -scaleyparam
    return_labels = torch.tensor([0.0, 0.0])
    return_labels[0] = torch.add(labels[0], x_shift)
    return_labels[1] = torch.add(labels[1], y_shift)
    return_labels = torch.reshape(return_labels, [2, ])

    return image_crop, return_labels
