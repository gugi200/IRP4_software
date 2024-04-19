# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 21:56:53 2024

@author: micha
"""
"""
# compare different normalizations methods
# 6 runs 5 epochs each, 10% dataset

mean = [0.25, 0.5, 0.75]

std = [0.1, 0.01, 0.001]


"""

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
import json

# check imports
print(torch.__version__)
print(torchvision.__version__)

#agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# test with no hyperparameter sweeping
torch.manual_seed(42)
torch.cuda.manual_seed(42)

PATH_TRAIN = Path("../pytorch_models/short_train")
PATH_TEST =  Path("../pytorch_models/short_test")
BATCH_SIZE = 16

def expFilter(coeff):
    def func(data):
            data = np.asarray(data)
            img_scaled = np.round(np.exp(-(coeff/((data*4)+0.1)))*1023)
            img_scaled = np.where(img_scaled>255, 255, img_scaled)
            mapped = img_scaled.astype(np.uint8)
            return Image.fromarray(mapped)
    return func


def trans(mean, std):
    tranform = transforms.Compose([
        expFilter(25),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 
    return tranform

transforms = ([
    (trans(mean=0.25, std=0.1), 'mean=0.25_std=0.1'),
    (trans(mean=0.25, std=0.01), 'mean=0.25_std=0.01'),
    (trans(mean=0.25, std=0.001), 'mean=0.25_std=0.001'),
    (trans(mean=0.5, std=0.1), 'mean=0.5_std=0.1'),
    (trans(mean=0.5, std=0.01), 'mean=0.5_std=0.01'),
    (trans(mean=0.5, std=0.001), 'mean=0.5_std=0.001'),
    (trans(mean=0.75, std=0.1), 'mean=0.75_std=0.1'),
    (trans(mean=0.75, std=0.01), 'mean=0.75_std=0.01'),
    (trans(mean=0.75, std=0.001), 'mean=0.75_std=0.001')],
    'normalization_10epochs')


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
    
    list_train_loss, list_train_acc, list_test_loss, list_test_acc = [], [], [], []
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
        list_train_loss.append(ave_batch_train_loss.cpu().item())
        list_train_acc.append(ave_batch_train_acc.cpu().item())
        list_test_loss.append(ave_batch_test_loss.cpu().item())
        list_test_acc.append(ave_batch_test_acc.cpu().item())

    train_time_end = timer()

    return list_train_loss, list_train_acc, list_test_loss, list_test_acc


def compare_transformations(transforms, trans_name, path_train, path_test):
    results = {}
    for transfrom, name in transforms:
        train_dataloader, class_names, _ = create_dataset(path=path_train,
                                                          batchsize=BATCH_SIZE,
                                                          preprocess=transfrom)
        
        test_dataloader, _ , _ = create_dataset(path=path_test,
                                                batchsize=BATCH_SIZE,
                                                preprocess=transfrom)
        
        weight = list(torchvision.models.get_model_weights('mobilenet_v3_large'))[-1]
        model = torch.hub.load('pytorch/vision', 'mobilenet_v3_large', weight).to(device)
        model.classifier[3] = nn.Linear(1280 , len(class_names), bias=True).to(device)
        
        
        b_tr_loss, b_tr_acc, b_te_loss, b_te_acc = train_model_googlenet(train_dataloader=train_dataloader,
                                         test_dataloader=test_dataloader,
                                         lr=0.01,
                                         optimizer='sgd',
                                         batchsize=BATCH_SIZE,
                                         epochs=10,
                                         class_names=class_names,
                                         model=model)
        results[str(name)+'_train_loss'] =  b_tr_loss
        results[str(name)+'_test_loss'] =  b_te_loss
        results[str(name)+'_train_acc'] =  b_tr_acc
        results[str(name)+'_test_acc'] =  b_te_acc
        
    return {f'{trans_name}': results}

def runPipe(tran_list):
    for trans_stack, name in tran_list:
        dic = compare_transformations(trans_stack, name, PATH_TRAIN, PATH_TEST)
        print(dic)
        file_path = str(name)+".json"
        with open(file_path, "w") as json_file:
            json.dump(dic, json_file)
        
runPipe([transforms])     
        
def runScaled():
    mean=[0.5, 0.5, 0.5]
    std=[0.001, 0.001, 0.001]
    tranform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
    ]) 
    trans_stack = [(tranform, 'arduinoNotNorm')]
    dic = compare_transformations(trans_stack, 'arduinoNotNorm', 
                                  Path('extended_arduinoScaled_train'), 
                                  Path('extended_arduinoScaled_test'))
    with open('arduinoNotNorm.json', "w") as json_file:
            json.dump(dic, json_file)

def runRaw():
    mean=[0.5, 0.5, 0.5]
    std=[0.001, 0.001, 0.001]
    tranform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 
    trans_stack = [(tranform, 'raw')]
    dic = compare_transformations(trans_stack, 'raw', PATH_TRAIN, PATH_TEST)
    with open('raw.json', "w") as json_file:
            json.dump(dic, json_file)

# runScaled()
