# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:44:15 2024

@author: micha
"""
from trainLibTorch import *
import wandb
import torch
from torchvision import transforms
from torchvision import datasets
import numpy as np
import torchvision

from torchvision import datasets#

from timeit import default_timer as timer
from tqdm.auto import tqdm
from PIL import Image


torch.manual_seed(42)
torch.cuda.manual_seed(42)

#agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

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



sweep_config = {
    'method': 'grid'
    }
metric = {
    'name': 'loss',
    'goal': 'minimize'
    }
sweep_config['metric'] = metric

parameters_dict = {
    'optimizer': {
        'values': ['adam', 'sgd']
        },
    'fc_layer_size': {
        'values': [8]
        },
    }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'epochs': {
        'value': 5}
    })

# parameters_dict.update({
#     'learning_rate': {
#         # a flat distribution between 0 and 0.1
#         'distribution': 'uniform',
#         'min': 0,
#         'max': 0.1
#       },
#     'batch_size': {
#         # integers between 32 and 256
#         # with evenly-distributed logarithms
#         'distribution': 'q_log_uniform_values',
#         'q': 8,
#         'min': 8,
#         'max': 32,
#       }
#     })

parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'values': [0.01]
      },
    'batch_size': {
        # integers between 32 and 256
        # with evenly-distributed logarithms
        'values': [16, 32]
      }
    })



import pprint
pprint.pprint(sweep_config)






def train_model_densenet161(config=None):

    
    with wandb.init(config=config):

        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        tranform = transforms.Compose([
            expMapping(4),
            softThreshold(30, 255),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]) 

        config = wandb.config

        train_dataloader, class_names, _ = create_dataset(path=Path("train_raw"),
                                                          batchsize=config.batch_size,
                                                          preprocess=tranform)

        test_dataloader, _ , _ = create_dataset(path=Path("test_raw"),
                                                batchsize=config.batch_size,
                                                preprocess=tranform)

        weight = list(torchvision.models.get_model_weights('densenet161'))[-1]
        model = torch.hub.load('pytorch/vision', 'densenet161', weight).to(device)
        model.classifier = nn.Linear(2208 , 10, bias=True).to(device)

        train_test_loop(config=config,
                        model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        class_names=class_names,
                        device=device
                        )
    return model

'''DONE'''

# densenet161 pipeline
# sweep_id_densenet161 = wandb.sweep(sweep_config, project="densenet161-customData")
# wandb.agent(sweep_id_densenet161, train_model_densenet161)


def train_model_efficientnet_v2_l(config=None):
    with wandb.init(config=config):
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        tranform = transforms.Compose([
            expMapping(4),
            softThreshold(30, 255),
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]) 
    
        config = wandb.config
        train_dataloader, class_names, _ = create_dataset(path=Path("train_raw"),
                                                          batchsize=config.batch_size,
                                                          preprocess=tranform)
    
        test_dataloader, _ , _ = create_dataset(path=Path("test_raw"),
                                                batchsize=config.batch_size,
                                                preprocess=tranform)
    
        weight = list(torchvision.models.get_model_weights('efficientnet_v2_l'))[-1]
        model = torch.hub.load('pytorch/vision', 'efficientnet_v2_l', weight).to(device)
        model.classifier[1] = nn.Linear(1280 , 10, bias=True).to(device)
    
        train_test_loop(config=config,
                        model=model,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        class_names=class_names,
                        device=device
                        )
    return model


# efficientnet_v2_l pipeline

sweep_id_efficientnet_v2_l = wandb.sweep(sweep_config, project="efficientnet_v2_l-customData")
wandb.agent(sweep_id_efficientnet_v2_l, train_model_efficientnet_v2_l)
















