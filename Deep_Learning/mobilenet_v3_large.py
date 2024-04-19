


# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 21:03:36 2024

@author: micha
"""


#
#   Michael Gugala
#   02/12/2023
#   Image recognition
#   Master 4th year project
#   Univeristy of Bristol
#

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import torchvision
from torchvision import datasets#
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchmetrics
from torchvision.models import resnet50, ResNet50_Weights
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.utils import Bunch

from PIL import Image

import requests
import random
import shutil
from pathlib import Path
import os

import wandb
# import cv2
from timeit import default_timer as timer
from tqdm.auto import tqdm
from trainLibTorch import *

# check imports
print(torch.__version__)
print(torchvision.__version__)

#agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



# test with no hyperparameter sweeping
torch.manual_seed(42)
torch.cuda.manual_seed(42)

BATCH_SIZE = 16

mean=[0.5, 0.5, 0.5]
std=[0.001, 0.001, 0.001]
tranform = transforms.Compose([
    expMapping(4),
    softThreshold(30, 255),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
]) 

PATH_TRAIN = Path("extended_raw_train")
PATH_TEST =  Path("extended_raw_test")


train_dataloader, class_names, _ = create_dataset(path=PATH_TRAIN,
                                                  batchsize=BATCH_SIZE,
                                                  preprocess=tranform)
print(class_names)
test_dataloader, _ , _ = create_dataset(path=PATH_TEST,
                                        batchsize=BATCH_SIZE,
                                        preprocess=tranform)

print(enumerate(train_dataloader))
print(test_dataloader)



weight = list(torchvision.models.get_model_weights('mobilenet_v3_large'))[-1]
model = torch.hub.load('pytorch/vision', 'mobilenet_v3_large', weight).to(device)
model.classifier[3] = nn.Linear(1280 , len(class_names), bias=True).to(device)

def train_model_googlenet(train_dataloader, test_dataloader, lr, optimizer, batchsize, epochs, class_names, model):
    print(device)

    model = model.to(device)
    loss_fn = get_lossFn()
    optimizer = create_optiimizer(model=model,
                                    optimizer=optimizer,
                                    lr=lr
    )

    prev_acc = 0
    metric = torchmetrics.classification.Accuracy(
        task="multiclass",
        num_classes=len(class_names)
    ).to(device)
    
    
    train_time_start = timer()
    for epoch in tqdm(range(epochs)):
        print([g['lr'] for g in optimizer.param_groups])

        ave_batch_train_loss, ave_batch_train_acc = train_step(
            model=model,
            metric=metric,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_loader=train_dataloader,
            device=device,
            debug=True,
            wnb=False
        )
        ave_batch_test_loss, ave_batch_test_acc = test_step(
            model=model,
            metric=metric,
            loss_fn=loss_fn,
            data_loader=test_dataloader,
            device=device,
            debug=True,
            wnb=False
        )
  
        curr_acc = ave_batch_train_acc

        if ((curr_acc-prev_acc)<0.02) and (curr_acc>0.65):
            for g in optimizer.param_groups:
              g['lr'] = 0.001
        prev_acc = curr_acc

    train_time_end = timer()

    return model

newModel = train_model_googlenet(train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 lr=0.01,
                                 optimizer='sgd',
                                 batchsize=BATCH_SIZE,
                                 epochs=7,
                                 class_names=class_names,
                                 model=model).cpu()

torch.save(obj=newModel.state_dict(), f='mobilenet_v3_large_test_10_classes.pth')



