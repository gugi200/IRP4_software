# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:22:27 2024

@author: micha
"""



import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os
import random
from trainLibTorch import *

alpha_coeff = 25
th_lower = 20
th_upper = 90


pureDataPath = Path("dataset_pressure_sensor/dataCollection1_sensor")
mappedDataPath = Path("dataset_pressure_sensor/dataCollection1_sensor")

pure_classes = os.listdir(pureDataPath)
mapped_classses = os.listdir(mappedDataPath)




def displayEachClass(path, title, transfroms=None):
    dirs = os.listdir(path)
    plt.figure(figsize=(9, 4.85))
    plt.suptitle(title)
    subplot_i = 1
    for _dir in dirs:
        file = random.choice(os.listdir(path/_dir))
        img = Image.open(path/_dir/file)
        img_np = np.asarray(img) 
        if transfroms:
            for transfrom in transfroms:
                img_np = transfrom(img_np)        
        ax = plt.subplot(2, 4, subplot_i)
        ax.set_title(_dir + ', \nvmax=' + str(np.max(img_np).astype(np.uint8)))
        ax.imshow(img_np, vmax = np.max(img_np))
        ax.axis(False)
        subplot_i += 1
       
        
def display4with4mappings(path):
        alphas = [0, 2, 4, 6, 8, 10, 12, 14]
        nrows = 4
        plt.figure(figsize=(15, 9))
        subplot_i = 1
        for i in range(1, 1+nrows):
            files = os.listdir(path)
            file = random.choice(files)
            img = Image.open(path/file)
            img_np = np.asarray(img) 
            for alpha in alphas:
                img_mapped = expMappingGeneric(img_np, alpha)
                ax = plt.subplot(nrows, len(alphas), subplot_i)
                ax.set_title(str((os.path.split(path))[-1])+
                             '\nvmax=' + 
                             str(np.max(img_np).astype(np.uint8))+
                             '\nalpha='+str(alpha))
                ax.imshow(img_mapped, vmax = np.max(img_mapped))
                ax.axis(False)
                subplot_i += 1   
                
                
def displayMultipleSameDir(path, title, transfroms=None):
    files = os.listdir(path)
    plt.figure(figsize=(15, 9))
    plt.suptitle(title)
    nrows = 4
    ncols = 6
    subplot_i = 1
    for i in range(1, 1+nrows):
        for j in range(ncols):
            file = random.choice(files)
            img = Image.open(path/file)
            img_np = np.asarray(img) 
            if transfroms:
                for transfrom in transfroms:
                    img_np = transfrom(img_np)        
                    
            ax = plt.subplot(nrows, ncols, subplot_i)
            ax.set_title(str((os.path.split(path))[-1])+
                         '\nvmax=' + 
                         str(np.max(img_np).astype(np.uint8)))
            ax.imshow(img_np, vmax = np.max(img_np))
            ax.axis(False)
            subplot_i += 1   

def expMappingGeneric(data, alpha):
    if alpha==0:
        mapped = data
    else:
        data = np.where(data==0, 1, data)
        mapped = np.exp( -( alpha/(data) ) )*255  
    return mapped.astype(np.uint8)

def expMapping8(data):

    data = np.where(data==0, 1, data)
    mapped = np.exp( -( 8/(data) ) )*255  
    return mapped.astype(np.uint8)


def expMapping25(data):
    data = np.where(data==0, 1, data)
    mapped = np.exp( -( 25/(data) ) )*1023  
    return mapped

def hardThreshold(data):
    th = np.where(data<th_lower, 0, 255)
    return th





def transofrmData(dataloader, class_names, mean, std, seed):

    train_features_batch, train_labels_batch = next(iter(dataloader))
    print("length of data: ", len(train_features_batch), 'length of labels: ', len(train_labels_batch))
    # display random datapoints
    fig = plt.figure(figsize=(9, 9))
    fig.suptitle(f'mean={mean}, std={std}')

    np.random.seed(seed)
    rows, cols  = 3, 3
    for pic in range(1, 1+rows*cols):
        rand_int = np.random.randint(0, batchsize)
        img = train_features_batch[rand_int]
        img_RGB = img.permute([1, 2, 0]).numpy()
        fig.add_subplot(rows, cols, pic)
        plt.imshow(img_RGB.squeeze())
        plt.axis(False)
        plt.title(class_names[train_labels_batch[rand_int]])


# random.seed(42)
# displayEachClass(pureDataPath, 'ammping apga applied in arduino')
# plt.show()
# random.seed(42)
# random.seed(42)
# displayEachClass(pureDataPath, 'alpha=8, hard', [expMapping8, hardThreshold])
# displayEachClass(pureDataPath, 'alpha=8, soft',[expMapping8, softThreshold])
# random.seed(42)
# displayEachClass(pureDataPath, 'alpha=8', [expMapping8])
# displayEachClass(mappedDataPath)

# dirs = os.listdir(pureDataPath)
# i = 5
# displayMultipleSameDir(pureDataPath/dirs[i], 
#                        dirs[i], 
#                        [expMapping8, softThreshold])


# dirs = os.listdir(pureDataPath) 
# for i in range(5):
#     display4with4mappings(pureDataPath/dirs[i])




if __name__=='__main__':
    batchsize=16
    seed = 20
    datapath = Path("extended_raw_test")
    mean=[0.5, 0.5, 0.5]
    std=[0.225, 0.225, 0.225]
    tranform = transforms.Compose([
        # expMapping(4),
        # softThreshold(30, 255),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 

    # dataloader, class_names, _ = create_dataset(path=datapath,
    #                                                 batchsize=batchsize,
    #                                                 preprocess=tranform, 
    #                                                 seed=42)

    # transofrmData(datapath, tranform, batchsize)


    mean=[0.5, 0.5, 0.5]
    std=[0.225, 0.225, 0.225]
    tranform = transforms.Compose([
        expMapping(4),
        softThreshold(30, 255),
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 

    # dataloader, class_names, _ = create_dataset(path=datapath,
    #                                                 batchsize=batchsize,
    #                                                 preprocess=tranform, 
    #                                                 seed=42)
    
    # transofrmData(datapath, tranform, batchsize)


    datapath = Path('extended_arduinoScaled_test')
    mean=[0.5, 0.5, 0.5]
    std = 0.12
    std=[std, std, std]
    tranform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 

    # dataloader, class_names, _ = create_dataset(path=datapath,
    #                                                 batchsize=batchsize,
    #                                                 preprocess=tranform, 
    #                                                 seed=42)

    # transofrmData(dataloader, class_names, mean, std)
    mean = 0.75
    mean=[mean, mean, mean]
    std = 0.001
    std=[std, std, std]
    tranform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 


    dataloader, class_names, _ = create_dataset(path=datapath,
                                                    batchsize=batchsize,
                                                    preprocess=tranform, 
                                                    seed=seed)

    transofrmData(dataloader, class_names, mean, std, seed=seed)

    mean = 0.5
    mean=[mean, mean, mean]
    std = 0.001
    std=[std, std, std]
    tranform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 

    dataloader, class_names, _ = create_dataset(path=datapath,
                                                    batchsize=batchsize,
                                                    preprocess=tranform, 
                                                    seed=seed)

    transofrmData(dataloader, class_names, mean, std, seed=seed)

    mean = 0.25
    mean=[mean, mean, mean]
    std = 0.001
    std=[std, std, std]
    tranform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]) 

    dataloader, class_names, _ = create_dataset(path=datapath,
                                                    batchsize=batchsize,
                                                    preprocess=tranform, 
                                                    seed=seed)

    transofrmData(dataloader, class_names, mean, std, seed=seed)

    plt.show()




















