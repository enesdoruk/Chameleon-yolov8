import torch.nn as nn

import sys
sys.path.insert(0, "/AI/syndet-yolo")

from syndet.channel_attention import domain_discrepancy


class MultiLayerDisc(nn.Module):
    def __init__(self, b7_ch, b5_ch, b3_ch, reduction):
        super(MultiLayerDisc, self).__init__()
        
        self.fc1b3 = nn.Conv2d(b3_ch, b3_ch // reduction, kernel_size=1, padding=0)
        self.relub3 = nn.ReLU(inplace=True)
        self.fc2b3 = nn.Conv2d(b3_ch // reduction, b3_ch, kernel_size=1, padding=0)
        
        self.fc1b5 = nn.Conv2d(b5_ch, b5_ch // reduction, kernel_size=1, padding=0)
        self.relub5 = nn.ReLU(inplace=True)
        self.fc2b5 = nn.Conv2d(b5_ch // reduction, b5_ch, kernel_size=1, padding=0)
        
        self.fc1b7 = nn.Conv2d(b7_ch, b7_ch // reduction, kernel_size=1, padding=0)
        self.relub7 = nn.ReLU(inplace=True)
        self.fc2b7 = nn.Conv2d(b7_ch // reduction, b7_ch, kernel_size=1, padding=0)
        
        
    def forward(self, target, source):
        out_b7 = self.fc2b7(self.relub7(self.fc1b7(target[0]))) + target[0]
        out_b5 = self.fc2b5(self.relub5(self.fc1b5(target[1]))) + target[1]
        out_b3 = self.fc2b3(self.relub3(self.fc1b3(target[2]))) + target[2]

        coral_b7 = domain_discrepancy(out_b7, source[0])
        coral_b5 = domain_discrepancy(out_b5, source[1])
        coral_b3 = domain_discrepancy(out_b3, source[2])
        
        return coral_b3 + coral_b5 + coral_b7       
       
       