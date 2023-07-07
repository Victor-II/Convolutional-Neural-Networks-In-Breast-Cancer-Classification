import torch.nn as nn
from torchvision.models import resnet50, densenet121, inception_v3

def resnet50_model(weights=None):
    model = resnet50(weights=weights)
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=2048, out_features=1),
        nn.Sigmoid()
    )
    name = 'resnet50' if weights == None else 'resnet50_pretrained'
    return model, name

def inceptionv3_model(weights=None):
    model = inception_v3(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(in_features=2048, out_features=1),
        nn.Sigmoid()
    )
    name = 'inceptionv3' if weights == None else 'inceptionv3_pretrained'
    return model, name

def densenet121_model(weights=None):
    model = densenet121(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features=1024, out_features=1),
        nn.Sigmoid()
    )
    name = 'densenet121' if weights == None else 'densenet121_pretrained'
    return model, name