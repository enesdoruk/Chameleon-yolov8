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
from syndet.multi_scale_alg import MultiScaleAlig


class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        
        self.layers = []
        self.layers.append(MultiScaleAlig())
        self.layers.append(Discriminator(num_convs=4, in_channels=1792, grad_reverse_lambda=0.02))
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
        backb5, backb7, backb10 = self.model[2](source)
        
        if target is None:
            head, _ = self.model[3](backb5,
                                backb7, 
                                backb10)  
            return head
        
        else:
            backb5_t, backb7_t, backb10_t = self.model[2](target)    
            
            head, head_convs_s = self.model[3](backb5,
                                backb7, 
                                backb10)  
            
            _, head_convs_t = self.model[3](backb5_t,
                                backb7_t, 
                                backb10_t)  
            
            source_disc_feat = self.model[0](head_convs_t[0], head_convs_t[1], head_convs_t[2])
            target_disc_feat = self.model[0](head_convs_s[0], head_convs_s[1], head_convs_s[2])
            
            grl_b10_s = self.model[1](source_disc_feat, 0, grl=True)            
            grl_b10_t = self.model[1](target_disc_feat, 1, grl=True)
            
            adv_loss = grl_b10_s + grl_b10_t
            
            return head, adv_loss