import torch
import torch.nn as nn
from torch.autograd import grad
import torch.nn.functional as F
from torch.autograd import Function


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
    def grad_reverse(x, ctx):
        return ReverseLayerF.apply(x, ctx)
    

class DomainDiscriminator(nn.Module):
    def __init__(self) -> None:
        super(DomainDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(51200, 512)
        self.fc2 = nn.Linear(512, 2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x, ctx):
        x = ReverseLayerF.grad_reverse(x, ctx)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, 1)
        return x
