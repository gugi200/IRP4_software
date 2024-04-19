# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:47:44 2024

@author: micha
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os
import json


def getExp():
    with open("exp.json", "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def getInt():
    with open("int.json", "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict
    
def getSoft():
    with open("soft.json", "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def getGamma():
    with open("gamma.json", "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def getRaw():
    with open("raw.json", "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def getArduino():
    with open("arduino.json", "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict

def getNorm():
    with open("normalization_10epochs.json", "r") as json_file:
        loaded_dict = json.load(json_file)
    return loaded_dict


def plotJson(getData, title):
    
    data = getData()
    first_entry = next(iter(data.values()))
    print(first_entry)
    train_acc, train_loss, test_acc, test_loss = [], [], [], []
    for key, value in first_entry.items():
        if 'train_acc' in key:
            train_acc.append((key, value))
        if 'train_loss' in key:
            train_loss.append((key, value))
        if 'test_acc' in key:
            test_acc.append((key, value))
        if 'test_loss' in key:
            test_loss.append((key, value))

    # accuracy train
    plt.figure(figsize=(9, 9))
    plt.title(title+' accuracy - train dataset', fontsize=20)
    for i in range(len(train_acc)):
        plt.plot(np.arange(1, 11, 1, dtype=int), train_acc[i][1], 
                 label=train_acc[i][0])

    plt.grid()
    plt.legend(fontsize=12)
    plt.ylabel('accuracy', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=12)
    plt.yticks(fontsize=12)
    
    
    # loss train
    plt.figure(figsize=(9, 9))
    plt.title(title+' loss - train dataset', fontsize=20)
    for i in range(len(train_loss)):
        plt.plot(np.arange(1, 11, 1, dtype=int), train_loss[i][1], 
                 label=train_loss[i][0])

    plt.grid()
    plt.legend(fontsize=12)
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=12)
    plt.yticks(fontsize=12)
    
    
    # test accuracy
    plt.figure(figsize=(9, 9))
    plt.title(title+' accuracy - test dataset', fontsize=20)
    for i in range(len(train_acc)):
        plt.plot(np.arange(1, 11, 1, dtype=int), test_acc[i][1], 
                 label=test_acc[i][0])

    plt.grid()
    plt.legend(fontsize=12)
    plt.ylabel('accuracy', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=12)
    plt.yticks(fontsize=12)
    
    
    # test loss
    plt.figure(figsize=(9, 9))
    plt.title(title+' loss - test dataset', fontsize=20)
    for i in range(len(train_loss)):
        plt.plot(np.arange(1, 11, 1, dtype=int), test_loss[i][1], 
                 label=test_loss[i][0])

    plt.grid()
    plt.legend(fontsize=12)
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], fontsize=12)
    plt.yticks(fontsize=12)


        

# plotJson(getExp, title='Exponential mapping')    
# plotJson(getGamma, title='Gamma correction')
# plotJson(getSoft, title='Soft Threshold mapping')
# plotJson(getInt, title='Integer Scaling mapping')
# plotJson(getArduino, title='Scaled in arduino coeff=20')    
# plotJson(getRaw, title='Raw data')


def getTheBestOfEach():

    best_test = {}
    best_train = {}
    best_test_loss = {}
    best_train_loss = {}
    jsons = [getExp, getGamma, getSoft, getInt, getRaw]
    for file in jsons:
        dic = file()
        name, values = next(iter(dic.items()))
        best_test[name] = ['', [0]]
        for key, value in values.items():
            
            if 'test_acc' in key:
                if value[-1]>best_test[name][1][-1]:
                    best_test[name][1] = value
                    best_test[name][0] = key
                    
                    train = key.replace('test_acc', 'train_acc')
                    best_train[name] = [train, values[train]]
                    
                    train = key.replace('test_acc', 'test_loss')
                    best_test_loss[name] = [train, values[train]]
                    
                    train = key.replace('test_acc', 'train_loss')
                    best_train_loss[name] = [train, values[train]]
                    
    return best_test, best_train, best_test_loss, best_train_loss

def plotBestOfEach():
    test, train, te_loss, tr_loss = getTheBestOfEach()
    
    plt.figure(figsize=(9, 9))
    for key, value in test.items():
        plt.plot([1,2,3,4,5,6,7,8,9,10], value[1], label=key+' '+value[0])
    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.ylabel('accuracy', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10], fontsize=12)
    plt.title('Test dataset - accuracy', fontsize=20)
    
    plt.figure(figsize=(9, 9))
    for key, value in train.items():
        plt.plot([1,2,3,4,5,6,7,8,9,10], value[1], label=key+' '+value[0])
    plt.legend(loc='lower right', fontsize=12)
    plt.grid()
    plt.ylabel('accuracy', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],fontsize=12)
    plt.title('Train dataset - accuracy', fontsize=20)
    
    
    plt.figure(figsize=(9, 9))
    for key, value in te_loss.items():
        plt.plot([1,2,3,4,5,6,7,8,9,10], value[1], label=key+' '+value[0])
    plt.legend(loc='upper right', fontsize=12)
    plt.grid()
    plt.ylabel('loss', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10], fontsize=12)
    plt.title('Test dataset - loss', fontsize=20)
    
    plt.figure(figsize=(9, 9))
    for key, value in tr_loss.items():
        plt.plot([1,2,3,4,5,6,7,8,9,10], value[1], label=key+' '+value[0])
    plt.legend(loc='upper right', fontsize=12)
    plt.grid()
    plt.ylabel('loss', fontsize=15)
    plt.yticks(fontsize=12)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],fontsize=12)
    plt.title('Train dataset - loss', fontsize=20)
    
#plotBestOfEach()




def plotJson5Epoch(getData, title):
    
    data = getData()
    first_entry = next(iter(data.values()))
    print(first_entry)
    train_acc, train_loss, test_acc, test_loss = [], [], [], []
    for key, value in first_entry.items():
        if 'train_acc' in key:
            train_acc.append((key, value))
        if 'train_loss' in key:
            train_loss.append((key, value))
        if 'test_acc' in key:
            test_acc.append((key, value))
        if 'test_loss' in key:
            test_loss.append((key, value))

    # accuracy train
    plt.figure(figsize=(9, 9))
    plt.title(title+' accuracy - train dataset', fontsize=20)
    for i in range(len(train_acc)):
        plt.plot(np.arange(1, 11, 1, dtype=int), train_acc[i][1], 
                 label=train_acc[i][0])

    plt.grid()
    plt.legend(fontsize=12, loc='lower right')
    plt.ylabel('accuracy', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],fontsize=12)
    plt.yticks(fontsize=12)
    
    
    # loss train
    plt.figure(figsize=(9, 9))
    plt.title(title+' loss - train dataset', fontsize=20)
    for i in range(len(train_loss)):
        plt.plot(np.arange(1, 11, 1, dtype=int), train_loss[i][1], 
                 label=train_loss[i][0])

    plt.grid()
    plt.legend(fontsize=12, loc='upper right')
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],fontsize=12)
    plt.yticks(fontsize=12)
    
    
    # test accuracy
    plt.figure(figsize=(9, 9))
    plt.title(title+' accuracy - test dataset', fontsize=20)
    for i in range(len(train_acc)):
        plt.plot(np.arange(1, 11, 1, dtype=int), test_acc[i][1], 
                 label=test_acc[i][0])

    plt.grid()
    plt.legend(fontsize=12, loc='lower right')
    plt.ylabel('accuracy', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],fontsize=12)
    plt.yticks(fontsize=12)
    
    
    # test loss
    plt.figure(figsize=(9, 9))
    plt.title(title+' loss - test dataset', fontsize=20)
    for i in range(len(train_loss)):
        plt.plot(np.arange(1, 11, 1, dtype=int), test_loss[i][1], 
                 label=test_loss[i][0])

    plt.grid()
    plt.legend(fontsize=12, loc='upper right')
    plt.ylabel('loss', fontsize=15)
    plt.xlabel('epochs', fontsize=15)
    plt.xticks([1,2,3,4,5,6,7,8,9,10],fontsize=12)
    plt.yticks(fontsize=12)


        


plotJson5Epoch(getNorm, 'whatever')
plt.show()