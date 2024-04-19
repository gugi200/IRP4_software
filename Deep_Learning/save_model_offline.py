import torch
import torchvision
from torch import nn


weights = list(torchvision.models.get_model_weights('mobilenet_v3_large'))[-1]
loaded_model = torch.hub.load('pytorch/vision', 'mobilenet_v3_large', weights)
loaded_model.classifier[3] = nn.Linear(1280, 6, bias=True)

loaded_model.load_state_dict(torch.load(f='mobilenet_v3_large_test_6_classes.pth'))



torch.save(loaded_model, 'mobilenet_6_classes.pth')

