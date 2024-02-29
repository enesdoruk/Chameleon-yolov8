import torch
import torch.nn.functional as F
from torch import nn

import sys
sys.path.insert(0, "/AI/syndet-yolo")

from syndet.gradient_reversal import GradientReversal


class Discriminator(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0):
        super(Discriminator, self).__init__()

        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            dis_tower.append(nn.GroupNorm(32, in_channels))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )

        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()


    def forward(self, feature, target, grl=True):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        
        if grl:
            feature = self.grad_reverse(feature)
        x = self.dis_tower(feature)
        x_out = self.cls_logits(x)

        target = torch.full(x_out.shape, target, dtype=torch.float, device=x_out.device)
        loss = self.loss_fn(x_out, target)

        if grl: 
            return loss
        else:
            return x