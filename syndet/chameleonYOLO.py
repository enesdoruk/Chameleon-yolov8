import cv2
import torch
import wandb
import numpy as np
import torch.nn as nn
from scipy.ndimage import zoom
from mmengine.visualization import Visualizer

import os
import sys
sys.path.insert(0, "/AI/syndet-yolo")

from syndet.modules import  Detect
from syndet.backbone import Backbone
from syndet.head import Head


class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        
        self.layers = []
        self.layers.append(Backbone())
        self.layers.append(Head())
        self.model = nn.Sequential(*self.layers)
        
        m = self.model[-1].detect 
        if isinstance(m, (Detect)):
            s = 256 
            m.inplace = True
            forward = lambda source:  self.forward(source)
            m.stride = torch.tensor([s / source.shape[-2] for source in forward(torch.zeros(1, 3, s, s))])  
            self.stride = m.stride
            m.bias_init() 

    def forward(self, source, target=None):      
        backb5, backb7, backb10 = self.model[0](source)
        
        if target is None:
            head, _ = self.model[1](backb5,
                                backb7, 
                                backb10)  
            return head
        
        else:            
            head, head_convs_s = self.model[1](backb5,
                                backb7, 
                                backb10)  
            
            
            return head