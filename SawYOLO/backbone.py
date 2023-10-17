import os
import sys
sys.path.insert(0, os.path.expanduser('~') + "/syndet-yolo-dcan")

import torch.nn as nn
from SawYOLO.modules import (DFL, Concat, Upsample, Detect, Conv, Bottleneck, C2f, SPPF)


class Backbone(nn.Module):
    def __init__(self) -> None:
        self.layers = []
        super(Backbone, self).__init__()
        
        self.layers.append(Conv(3, 64, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(Conv(64, 128, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(128, 128, n=1, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(128, 256, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(256, 256, n=1, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(256, 512, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(512, 512, n=1, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(512, 1024, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(1024, 1024, n=1, shortcut=True, g=1, e=0.5))
        self.layers.append(SPPF(1024, 1024, k=5))
        
        self.model = nn.Sequential(*self.layers)
        
    def forward(self, x):
        b1 = self.model[0](x)

        b2 = self.model[1](b1)
        b3 = self.model[2](b2)

        b4 = self.model[3](b3)
        b5 = self.model[4](b4)

        b6 = self.model[5](b5)
        b7 = self.model[6](b6)

        b8 = self.model[7](b7)
        b9 = self.model[8](b8)
        b10 = self.model[9](b9)
        
        return b5, b7, b10