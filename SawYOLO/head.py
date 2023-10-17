import os
import sys
sys.path.insert(0, os.path.expanduser('~') + "/syndet-yolo-dcan")

import torch.nn as nn
from SawYOLO.modules import (DFL, Concat, Upsample, Detect, Conv, Bottleneck, C2f, SPPF)



class Head(nn.Module):
    def __init__(self, ) -> None:
        self.layers = []
        super(Head, self).__init__()
        
        self.layers.append(Upsample(1024, 2))
        self.layers.append(Concat())
        self.layers.append(C2f(1536, 512, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Upsample(512, 2))
        self.layers.append(Concat())
        self.layers.append(C2f(768, 256, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(256, 256, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(Concat())
        self.layers.append(C2f(768, 512, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(512, 512, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(Concat())
        self.layers.append(C2f(1536, 1024, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Detect(nc=4, ch=(1024, 512, 256)))

        self.model = nn.Sequential(*self.layers)
        
    def forward(self, b5, b7, b10):
        h11 = self.model[0](b10)
        h12 = self.model[1]((h11,b7))
        h13 = self.model[2](h12)

        h14 = self.model[3](h13)
        h15 = self.model[4]((h14,b5))
        h16 = self.model[5](h15)

        h17 = self.model[6](h16)
        h18 = self.model[7]((h17,h13))
        h19 = self.model[8](h18)

        h20 = self.model[9](h19)
        h21 = self.model[10]((h20,b10))
        h22 = self.model[11](h21)
        h23 = self.model[12]([h22, h19, h16])
        
        return h23