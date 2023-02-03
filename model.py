import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

from config import *


class ClsfModel(nn.Module):
    def __init__(self):
        super(ClsfModel, self).__init__()
        pass
    def forward(self, x):
        return x


def init_model(config):
    assert config['modelname'] in config['MODEL_AVAILABLE'], f'"modelname" in config must in {config["MODEL_AVAILABLE"]}'
    model_name = config['modelname']
    if model_name == 'custom':
        model = ClsfModel()
    else:
        if model_name == 'mobilenetv3s':
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        elif model_name == 'regnetx800mf':
            backbone = regnet_x_800mf(weights=RegNet_X_800MF_Weights.DEFAULT)

        elif model_name == 'regnetx8gf':
            backbone = regnet_x_8gf(weights=RegNet_X_8GF_Weights.DEFAULT)

        elif model_name == 'resnet18':
            backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
           
        elif model_name == 'resnet50':
            backbone = resnet50(weights=ResNet50_Weights.DEFAULT)

        elif model_name == 'resnet152':
            backbone = resnet152(weights=ResNet152_Weights.DEFAULT)

        model = nn.Sequential(
            backbone,
            nn.Linear(1000, CLASS_INFO['num']),
            nn.Softmax(dim=1)
        )
    
    if config['load_checkpoint'] is not None:
        model.load_state_dict(torch.load(config['load_checkpoint']))
        try:
            config['logger'].info(f"Load checkpoint from {config['load_checkpoint']} successfully")
        except:
            print(f"Load checkpoint from {config['load_checkpoint']} successfully")

    return model
