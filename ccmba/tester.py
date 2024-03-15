import ccmba_ext_transforms as et
from torch.utils import data
from dataset_voc import VOCSegmentation

import torchvision.transforms.functional as TF
import PIL
import os
import subprocess

# Run this in terminal to remove .DS_Store files:
# find ccmba/blur_kernels_levelwise -name '.DS_Store' -type f -delete

train_transform = et.ExtCompose([
                et.ExtRandomScale((0.5, 2.0)),
                et.ExtRandomCrop(size=(256,256), pad_if_needed=True),
                et.ExtCCMBA(kerneldirectory='blur_kernels_levelwise'),
                et.ExtRandomHorizontalFlip(),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])



# @FIX THIS: datasets/data path not found!!
train_dst = VOCSegmentation(root='datasets/data', year='2012' ,image_set='train',download=False, transform=train_transform)
train_loader = data.DataLoader(train_dst, batch_size=4, shuffle=True, num_workers=0,drop_last=True) 

for image, label in train_loader:
    print(image.shape,label.shape)
