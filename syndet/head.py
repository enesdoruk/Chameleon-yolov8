import os
import sys
sys.path.insert(0, os.path.expanduser('~') + "/syndet-yolo-grl")

import torch.nn as nn
from syndet.modules import (Concat, Upsample, Detect, Conv, C2f)



class Head(nn.Module):
    def __init__(self, ) -> None:
        self.layers = []
        super(Head, self).__init__()
        
        self.up1 = Upsample(1024, 2)
        self.cat1 = Concat()
        self.c2f1 = C2f(1536, 512, n=3, shortcut=True, g=1, e=0.5)

        self.up2 = Upsample(512, 2)
        self.cat2 = Concat()
        self.c2f2 = C2f(768, 256, n=3, shortcut=True, g=1, e=0.5)

        self.conv1 = Conv(256, 256, k=3, s=2, p=1, g=1, d=1, act=True)
        self.cat3 = Concat()
        self.c2f3 = C2f(768, 512, n=3, shortcut=True, g=1, e=0.5)

        self.conv2 = Conv(512, 512, k=3, s=2, p=1, g=1, d=1, act=True)
        self.cat4 = Concat()
        self.c2f4 = C2f(1536, 1024, n=3, shortcut=True, g=1, e=0.5)

        self.detect = Detect(nc=4, ch=(1024, 512, 256))

        
    def forward(self, b5, b7, b10):
        h11 = self.up1(b10)
        h12 = self.cat1((h11,b7))
        h13 = self.c2f1(h12)

        h14 = self.up2(h13)
        h15 = self.cat2((h14,b5))
        h16 = self.c2f2(h15)

        h17 = self.conv1(h16)
        h18 = self.cat3((h17,h13))
        h19 = self.c2f3(h18)

        h20 = self.conv2(h19)
        h21 = self.cat4((h20,b10))
        h22 = self.c2f4(h21)
        
        h23 = self.detect([h22, h19, h16])
        
        return h23