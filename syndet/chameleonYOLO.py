import torch
import torch.nn as nn

import sys
sys.path.insert(0, "/AI/syndet-yolo")

from syndet.modules import  Detect
from syndet.backbone import Backbone
from syndet.head import Head
from syndet.discriminator import Discriminator
from syndet.multi_scale_alg import MultiScaleAlig
from syndet.channel_attention import ChannelAttention, domain_discrepancy


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
            
        self.tar_ca_last = [1.]
        self.src_ca_last = [1.]
        self.weight_d = 0.3
        self.ema_alpha = 0.999

    def forward(self, source, target=None, global_step=None):      
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
            
            
            ema_alpha = min(1 - 1 / (global_step+1), self.ema_alpha)
        
            source_att = self.model[0](source_disc_feat)
            target_att = self.model[0](target_disc_feat)
            
            source_feat_att = torch.mul(source_disc_feat, source_att)
            target_feat_att = torch.mul(target_disc_feat, target_att)
            
            mean_tar_ca = self.tar_ca_last[0] * ema_alpha + (1. - ema_alpha) * torch.mean(target_feat_att, 0)
            self.tar_ca_last[0] = mean_tar_ca.detach()
            
            mean_src_ca = self.src_ca_last[0] * ema_alpha + (1. - ema_alpha) * torch.mean(source_feat_att, 0)
            self.src_ca_last[0] = mean_src_ca.detach()
            
            d_const_loss = self.weight_d * domain_discrepancy(mean_src_ca, mean_tar_ca)
                            
            grl_b10_s = self.model[2](source_feat_att, 0, grl=True)            
            grl_b10_t = self.model[2](target_feat_att, 1, grl=True)
            
            adv_loss = grl_b10_s + grl_b10_t
            
            return head, adv_loss, d_const_loss