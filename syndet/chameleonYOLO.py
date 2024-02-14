from PIL import Image

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import sys
sys.path.insert(0, "/AI/syndet-yolo")

from syndet.gradientreversal import  DomainDiscriminator
from syndet.modules import  Detect
from syndet.backbone import Backbone
from syndet.head import Head

class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        
        self.layers = []
        self.layers.append(DomainDiscriminator())
        self.layers.append(Backbone())
        self.layers.append(Head())
        self.model = nn.Sequential(*self.layers)

        m = self.model[-1].detect 
        if isinstance(m, (Detect)):
            s = 256 
            m.inplace = True
            forward = lambda x:  self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])  
            self.stride = m.stride
            m.bias_init() 
        

    def forward(self, x, target=None, alpha=1.):      
        backb5, backb7, backb10 = self.model[1](x)
        
        head = self.model[2](backb5,
                             backb7, 
                             backb10)  

        if target is not None:
            backb5_t, backb7_t, backb10_t = self.model[1](target) 
    
            domain_output_s, log_soft_s = self.model[0](backb10)
            domain_output_t, _ = self.model[0](backb10_t)
            
            return head, domain_output_s, domain_output_t, log_soft_s
        else:
            return head
