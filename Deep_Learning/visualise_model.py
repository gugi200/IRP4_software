# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:35:42 2024

@author: micha
"""
import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from trainLibTorch import *

# select the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


BATCH_SIZE = 16

mean=[0.5, 0.5, 0.5]
std=[0.001, 0.001, 0.001]
tranform = transforms.Compose([
    # expMapping(4),
    # softThreshold(30, 255),
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
]) 
train_dataloader, class_names, _ = create_dataset(path=Path("extended_arduinoScaled_train"),
                                                  batchsize=BATCH_SIZE,
                                                  preprocess=tranform)
print(class_names)
test_dataloader, _ , _ = create_dataset(path=Path("extended_arduinoScaled_test"),
                                        batchsize=BATCH_SIZE,
                                        preprocess=tranform)

# get model


# load model
model_path = "mobilenet_v3_large_test_6_classes.pth"
weight = list(torchvision.models.get_model_weights('mobilenet_v3_large'))[-1]
loadded_model = torch.hub.load('pytorch/vision', 'mobilenet_v3_large', weight).to(device)
loadded_model.classifier[3] = nn.Linear(1280 , len(class_names), bias=True).to(device)
loadded_model.load_state_dict(torch.load(f=model_path))


visualize_preds(model=loadded_model,
                dataloader=test_dataloader,
                class_names=class_names,
                batchsize=BATCH_SIZE)

preds, targets, accuracy = make_predictions_dataloader(loadded_model, 
                                              train_dataloader,
                                              device,
                                              class_names)
print("Train subset accurafe = ", accuracy)

plot_decision_matrix(class_names=class_names,
                      y_pred_tensor=torch.tensor(preds),
                      targets=torch.tensor(targets),
                      title='Decision matrix - Train Subset')


preds, targets, accuracy  = make_predictions_dataloader(loadded_model, 
                                              test_dataloader, 
                                              device,
                                              class_names)
print("Test subset accurafe = ", accuracy)
plot_decision_matrix(class_names=class_names,
                      y_pred_tensor=torch.tensor(preds),
                      targets=torch.tensor(targets),
                      title='Decision matrix - Test Subset')    

plt.show()