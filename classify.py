from __future__ import print_function, division
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import glob
import cv2
import time
import PIL
from PIL import Image
import pickle
import matplotlib.pyplot as plt


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):

        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        return {'image': img, 'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, lbl = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'label': lbl } #torch.from_numpy(lbl)}

if __name__ == "__main__":

    startTime = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        Rescale(224),
        ToTensor(),
         ] )

    #validationFNames = pd.read_csv('trainingFileNames.txt',header=None)
    validationFNames = pd.read_csv('validationFileNames.txt',header=None)

    n = validationFNames.shape
    print(n)

    correct = 0

    with open('myModel.pkl', 'rb') as fid:
        model_ft = pickle.load(fid)

    for ii in range(n[0]):

        fName = validationFNames.iloc[ii,0]

        splitFName = fName.split('/')

        lbl = splitFName[3] 

        labelID = 0
        if lbl == 'horses':
            labelID = 0
        else:
            labelID = 1

        image = io.imread(fName)
        image = image[:,:,:3]

        sample = {'image': image, 'label': labelID }
        sample = data_transforms(sample)

        inputs = sample['image'].to(device)
        inputs = inputs.unsqueeze(0)   

        outputs             = model_ft(inputs.float())
        tempVar, preds      = torch.max(outputs, 1)

        print(labelID,outputs) #preds)

        if labelID == preds.data.numpy()[0]:
            correct += 1

    print(correct/n[0])

    endTime = time.time()

    timeTaken = endTime - startTime
    doneStr = '[Done! Time Taken: %.2f]' % (timeTaken)
    print(doneStr)