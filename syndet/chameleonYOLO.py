import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/AI/syndet-yolo")

from syndet.modules import  Detect
from syndet.backbone import Backbone
from syndet.head import Head
from syndet.discriminator import Discriminator
from syndet.multi_scale_alg import MultiScaleAlig
from syndet.channel_attention import ChannelAttention


class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        
        
        self.layers = []
        self.layers.append(ChannelAttention(in_planes=1024+512+256, ratio=4))
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
        backb5, backb7, backb10 = self.model[3](source)
        
        if target is None:
            head, _ = self.model[4](backb5,
                                backb7, 
                                backb10)  
            return head
        
        else:
            backb5_t, backb7_t, backb10_t = self.model[3](target)    
            
            head, head_convs_s = self.model[4](backb5,
                                backb7, 
                                backb10)  
            
            _, head_convs_t = self.model[4](backb5_t,
                                backb7_t, 
                                backb10_t)  
            
            source_disc_feat = self.model[1](head_convs_t[0], head_convs_t[1], head_convs_t[2])
            target_disc_feat = self.model[1](head_convs_s[0], head_convs_s[1], head_convs_s[2])
            
            source_att = self.model[0](source_disc_feat)
            target_att = self.model[0](target_disc_feat)
            
            source_feat_att = torch.mul(source_disc_feat, source_att)
            target_feat_att = torch.mul(target_disc_feat, target_att)

            grl_b10_s = self.model[2](source_feat_att, 0, grl=True)            
            grl_b10_t = self.model[2](target_feat_att, 1, grl=True)
            
            adv_loss = grl_b10_s + grl_b10_t
            
            return head, adv_loss