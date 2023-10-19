import os
import sys
sys.path.insert(0, os.path.expanduser('~') + "/syndet-yolo-dcan")

import torch.nn as nn
from SawYOLO.modules import  (Conv, C2f, SPPF)


class Backbone(nn.Module):
    def __init__(self) -> None:
        self.layers = []
        super(Backbone, self).__init__()
        
        self.conv1 = Conv(3, 64, k=3, s=2, p=1, g=1, d=1, act=True)
        self.conv2 = Conv(64, 128, k=3, s=2, p=1, g=1, d=1, act=True)
        self.c2f1 = C2f(128, 128, n=1, shortcut=True, g=1, e=0.5)

        self.conv3 = Conv(128, 256, k=3, s=2, p=1, g=1, d=1, act=True)
        self.c2f2 = C2f(256, 256, n=1, shortcut=True, g=1, e=0.5)

        self.conv4 = Conv(256, 512, k=3, s=2, p=1, g=1, d=1, act=True)
        self.c2f3 = C2f(512, 512, n=1, shortcut=True, g=1, e=0.5)

        self.conv5 = Conv(512, 1024, k=3, s=2, p=1, g=1, d=1, act=True)
        self.c2f4 = C2f(1024, 1024, n=1, shortcut=True, g=1, e=0.5)
        self.sppf = SPPF(1024, 1024, k=5)
        
                
    def forward(self, x):
        b1 = self.conv1(x)

        b2 = self.conv2(b1)
        b3 = self.c2f1(b2)

        b4 = self.conv3(b3)
        b5 = self.c2f2(b4)

        b6 = self.conv4(b5)
        b7 = self.c2f3(b6)

        b8 = self.conv5(b7)
        b9 = self.c2f4(b8)
        b10 = self.sppf(b9)
        
        return b5, b7, b10