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
from syndet.discriminator import Discriminator


class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        
        self.layers = []
        self.layers.append(Discriminator(num_convs=4, in_channels=1024, grad_reverse_lambda=0.02))
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


    def forward(self, source, target=None, verbose=False, it=0, ep=0):      
        backb5, backb7, backb10 = self.model[1](source)
        
        if target is None:
            head = self.model[2](backb5,
                                backb7, 
                                backb10)  
            return head
        
        else:
            backb5_t, backb7_t, backb10_t = self.model[1](target)    
            
            head = self.model[2](backb5,
                                backb7, 
                                backb10)  
            

            grl_b10_s = self.model[0](backb10, 0, grl=True)            
            grl_b10_t = self.model[0](backb10_t, 1, grl=True)
            
            adv_loss = grl_b10_s + grl_b10_t
                        
            return head, adv_loss