# -*- coding: utf-8 -*-
from torch import nn
from torchvision import models

class CNNNet(nn.Module):
    def __init__(self, out_dim, **kwargs):
        super(CNNNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Sequential(
            # nn.Linear(2048, 2048),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(512, out_dim))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.model(x)
        if not self.training:
            out = self.sigmoid(out)
        return out