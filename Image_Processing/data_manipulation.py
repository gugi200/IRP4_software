# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 21:22:44 2024

@author: micha
"""


#
#   Michael Gugala
#   02/12/2023
#   Image recognition
#   Master 4th year project
#   Univeristy of Bristol
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from PIL import Image
from PIL import UnidentifiedImageError
import requests
import random
import shutil
import zipfile
from pathlib import Path
import os



PATH_TRAIN = Path("extended_arduinoScaled_train")
PATH_TEST =  Path("extended_arduinoScaled_test")

SHORT_TRAIN = Path("short_train")
SHORT_TEST = Path("short_test")
TRAIN_RATIO = 0.75


data_folder_name = "dataset_pressure_sensor"
dest_folder_name = "extended_dataset_pressure_sensor"
extendedDataPath = Path(dest_folder_name)
##############################################
### ROTATING AND TRANSPOSE & ROTATING DATA ###
##############################################
def createExtendedDataset(alpha=None):
    customDirPath = Path(f"{data_folder_name}/dataCollection1_sensor")
    dirs = os.listdir(customDirPath)
    
    #  Create a dir with processed data
    extendedDataPath = Path(dest_folder_name)
    if extendedDataPath.is_dir():
        print('directory already exists')
    else:
        extendedDataPath.mkdir(parents=True, exist_ok=True)
        for dir in dirs:
            path = extendedDataPath / dir
            path.mkdir(parents=True, exist_ok=True)
    
    
    fails = 0
    index = 0
    for dir in dirs:
        files = os.listdir(customDirPath / dir)
    
        for file in files:
            try:
                img = Image.open(customDirPath / dir / file)
            except (NameError, UnidentifiedImageError):
                fails += 1
                pass
            img = np.asarray(img)
            
            if alpha:
                img_map = np.exp(-(alpha/(img+1)))*255
                img = np.floor(img_map)
                
            imgNp = np.asarray(img)
            imgNp_T = np.transpose(imgNp)
    
            im = Image.fromarray(imgNp)
            im.save(f"{extendedDataPath}/{dir}/{dir}_{index}.jpg")
    
            im = Image.fromarray(imgNp_T)
            im.save(f"{extendedDataPath}/{dir}/{dir}_{index+1}.jpg")
            for i in range(3):
                imgNp = np.rot90(imgNp)
                imgNp_T = np.rot90(imgNp_T)
    
                im = Image.fromarray(imgNp)
                im.save(f"{extendedDataPath}/{dir}/{dir}_{(index) + (2*(i+1))}.jpg")
    
                im = Image.fromarray(imgNp_T)
                im.save(f"{extendedDataPath}/{dir}/{dir}_{(index) + (2*(i+1)) + 1}.jpg")
    
            index += 8
    
    print('Data mutiplied succesfully')
    print('number of fails: ', fails)
    l = 0
    for dir in dirs:
        l += len(os.listdir(extendedDataPath/dir))
        print(dir, len(os.listdir(extendedDataPath/dir)), len(os.listdir(customDirPath/dir)), len(os.listdir(customDirPath/dir))*8)
    print(l)


##############################################
### SPLITTING THE DATA INTO TRAIN AND TEST ###
##############################################
def train_test_split(alpha=None):   
    dirs = os.listdir(extendedDataPath)
    
    
    #  Create a dir for train and test data
    extendedTrain = Path(PATH_TRAIN)
    extendedTest = Path(PATH_TEST)
    if extendedTrain.is_dir():
        print('directory already exists')
    else:
        extendedTrain.mkdir(parents=True, exist_ok=True)
        extendedTest.mkdir(parents=True, exist_ok=True)
        for dir in dirs:
            path = extendedTrain / dir
            path.mkdir(parents=True, exist_ok=True)
        for dir in dirs:
            path = extendedTest / dir
            path.mkdir(parents=True, exist_ok=True)
    
    for dir in dirs:
        files = os.listdir(extendedDataPath / dir)
        length = int(TRAIN_RATIO*len(files))
        random.shuffle(files)
    
        train_set = files[:length]
        test_set = files[length:]
    
        for data in train_set:
            if alpha:
                img = Image.open(extendedDataPath / dir / data)
                img = expMapping(alpha)(img)
                img_map = softThreshold(30, 255)(img)
                PIL_image = Image.fromarray(np.uint8(img_map))
                PIL_image.save(extendedTrain / dir / data)
            else:
                shutil.copy(extendedDataPath / dir / data, extendedTrain / dir / data)
    
        for data in test_set:
            if alpha:
                img = Image.open(extendedDataPath / dir / data)
                img = expMapping(alpha)(img)
                img_map = softThreshold(30, 255)(img)
                PIL_image = Image.fromarray(np.uint8(img_map))
                PIL_image.save(extendedTest / dir / data)
            else:
                shutil.copy(extendedDataPath / dir / data, extendedTest / dir / data)
    
    print('Data split succesfully')
    l = 0
    for dir in dirs:
        l += len(os.listdir(extendedTrain/dir))
        print(dir, len(os.listdir(extendedTrain/dir)))
    print(l)
    
    l = 0
    for dir in dirs:
        l += len(os.listdir(extendedTest/dir))
        print(dir, len(os.listdir(extendedTest/dir)))
    print(l)


################################################################
### SPLITTING THE DATA INTO TRAIN AND TEST  - TESTING SUBSET ###
################################################################
def train_test_split_subset(train_length, test_length):
    TRAIN_LENGTH_PER_CLASS = train_length
    TEST_LENGTH_PER_CLASS = test_length
    dirs = os.listdir(extendedDataPath)
    
    
    #  Create a dir for train and test data
    
    if SHORT_TRAIN.is_dir():
        print('directory already exists')
    else:
        SHORT_TRAIN.mkdir(parents=True, exist_ok=True)
        SHORT_TEST.mkdir(parents=True, exist_ok=True)
        for dir in dirs:
            path = SHORT_TRAIN / dir
            path.mkdir(parents=True, exist_ok=True)
        for dir in dirs:
            path = SHORT_TEST / dir
            path.mkdir(parents=True, exist_ok=True)
    
    for dir in dirs:
        files = os.listdir(extendedDataPath / dir)
        random.shuffle(files)
    
        train_set = files[:TRAIN_LENGTH_PER_CLASS]
        test_set = files[TRAIN_LENGTH_PER_CLASS:TRAIN_LENGTH_PER_CLASS+TEST_LENGTH_PER_CLASS]
    
        for data in train_set:
            shutil.copy(extendedDataPath / dir / data, SHORT_TRAIN / dir / data)
    
        for data in test_set:
            shutil.copy(extendedDataPath / dir / data, SHORT_TEST / dir / data)
    
    l = 0
    for dir in dirs:
        l += len(os.listdir(SHORT_TRAIN/dir))
        print(dir, len(os.listdir(SHORT_TRAIN/dir)))
    print(l)
    
    l = 0
    for dir in dirs:
        l += len(os.listdir(SHORT_TEST/dir))
        print(dir, len(os.listdir(SHORT_TEST/dir)))
    print(l)


def softThreshold(lower, upper):
    def func(data):
        data = np.asarray(data)
        th = np.where(data>upper, 255, data)
        th = np.where(th<lower, 0, th).astype(np.uint8)
        return Image.fromarray(th)
    return func

def expMapping(alpha):
    def func(data):
            data = np.asarray(data)
            data = np.where(data==0, 1, data)
            mapped = np.exp( -( alpha/(data) ) )*255
            mapped = mapped.astype(np.uint8)
            return Image.fromarray(mapped)
    return func





if __name__=='__main__':
    l = 0
    
    train_test_split()
    
    path = Path("dataset_pressure_sensor")/"dataCollection1_sensor"
    dirs = os.listdir(path)
    for dir in dirs:
        l += len(os.listdir(path/dir))
        print(dir, len(os.listdir(path/dir)))
    print(l)
    
    dirs = os.listdir(PATH_TRAIN)
    l = 0
    for dir in dirs:
        l += len(os.listdir(PATH_TRAIN/dir))
        print(dir, len(os.listdir(PATH_TRAIN/dir)))
    print(l)
    
    dirs = os.listdir(PATH_TEST)
    l = 0
    for dir in dirs:
        l += len(os.listdir(PATH_TEST/dir))
        print(dir, len(os.listdir(PATH_TEST/dir)))
    print(l)
