import torch
from torch import nn
import math

import sys
sys.path.insert(0, "/AI/syndet-yolo")

def get_L2norm_loss_self_driven(x, weight_L2norm=0.05):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 1.0
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return weight_L2norm * l


class DomainAttention(nn.Module):
    def __init__(self,  in_channels=1024, out_channel=32, drop_r=0.5) -> None:
        super(DomainAttention, self).__init__()
        
        self.dropout_rate = drop_r
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels= in_channels,
                                            out_channels= out_channel, 
                                            kernel_size=3, 
                                            padding= 1, 
                                            stride=1),
                                nn.LayerNorm((out_channel, 20, 20)),
                                nn.ReLU(),
                                 )
        
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Sequential(
            nn.Linear(12800, 1000),
            nn.BatchNorm1d(1000, affine=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout_rate)
            )

    def forward(self, feature) -> torch.Tensor:
        feat = self.conv(feature)
        feat = self.flatten(feat)
        out = self.fc1(feat)
        out.mul_(math.sqrt(1 - self.dropout_rate))    
               
        return out