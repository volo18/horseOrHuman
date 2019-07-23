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

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

#plt.ion()   # interactive mode


class HumanHorseData(Dataset):

    def __init__(self, transform = None):
        #dataPth = './DATA/'
        self.trainingNames      = pd.read_csv('trainingFileNames.txt',header=None)
        self.transform          = transform

    def __len__(self):
        return len(self.trainingNames)

    def __getitem__(self, id):

        fName = self.trainingNames.iloc[id,0]

        #image = Image.open(fName)

        image = io.imread(fName)
        image = image[:,:,:3]

        #image = cv2.imread(fName,cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        splitFName = fName.split('/')

        lbl = splitFName[3] 

        labelID = 0
        if lbl == 'horses':
            labelID = 0
        else:
            labelID = 1


        sample = {'image': image, 'label': labelID }

        if self.transform:
            sample = self.transform(sample)

        return sample

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

        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image), 'label': lbl } 

class RandomFlip(object):
    """Horiz flip image"""

    def __call__(self, sample):
        image, lbl = sample['image'], sample['label']

        p = np.random.random_sample()

        if p > 0.5:
            #cv2.imshow('Original Image', image)
            image = image[..., ::-1,:] - np.zeros_like(image)

            #cv2.imshow('New Image', image)
            #cv2.waitKey(0)

        return {'image': image, 'label': lbl}

class RandomCrop(object):
    """Crop image"""

    def __call__(self, sample):
        
        image, lbl = sample['image'], sample['label']

        h, w = image.shape[:2]
        centreH = np.round(h / 2).astype(int)
        centreW = np.round(w / 2).astype(int)

        p = np.random.random_sample()

        if p > 0.8:
            h *= p
            w *= p
            
            h = np.round(h / 2).astype(int)
            w = np.round(w / 2).astype(int)

            image = image[centreH-h:centreH+h,centreW-w:centreW+w,:] #- np.zeros_like(image)
            #image -= np.zeros_like(image)

        return {'image': image, 'label': lbl}


if __name__ == "__main__":    
    startTime = time.time()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose([
        RandomCrop(),
        Rescale(224),
        RandomFlip(),
        ToTensor(),
        #transforms.Resize(224),        
        #transforms.RandomHorizontalFlip(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        #transforms.ToPILImage(),
        #transforms.ToTensor(),
         ] )

    myDataSet   = HumanHorseData( transform = data_transforms ) 

    myDataloader  = DataLoader(myDataSet, batch_size=20, shuffle=True) #, num_workers=4)

    #for i_batch, sample_batched in enumerate(dataloader):
        #print(i_batch, sample_batched['image'].size() ) #, sample_batched['label'])

    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    #criterion = nn.BCELoss()  #CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft.train()
    model_ft = model_ft.float()

    num_epochs = 15

    e_losses = []

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        running_loss = 0.0

        for i_batch, sample_batched in enumerate(myDataloader):

            inputs = sample_batched['image'].to(device)
            labels = sample_batched['label'].to(device)

            optimizer_ft.zero_grad()
            
            outputs     = model_ft(inputs.float())
            _, preds    = torch.max(outputs, 1)
            loss        = criterion(outputs, labels)

            loss.backward()
            optimizer_ft.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(myDataSet)

        e_losses.append(epoch_loss)

        print('Loss: {:.4f}'.format(epoch_loss) )
        with open('myModel.pkl', 'wb') as fid:
            pickle.dump(model_ft, fid) 

        #plt.plot(e_losses)

        #print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))


    endTime = time.time()
    timeTaken = endTime - startTime
    doneStr = '[Done! Time Taken: %.2f]' % (timeTaken)
    print(doneStr)