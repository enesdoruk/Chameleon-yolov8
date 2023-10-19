
import torch.nn as nn
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
    
    

class DomainDiscriminator(nn.Module):
    def __init__(self) -> None:
        super(DomainDiscriminator, self).__init__()
        
        self.layers = []

        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(1024*10*10, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.ReLU()
        self.lin2 = nn.Linear(1024, 128)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(128, 2)
        self.softmax = nn.LogSoftmax(dim=1)
        
        
    def forward(self, x):
        x = self.flat(x)
        x = self.relu1(self.bn1(self.lin1(x)))
        x = self.relu2(self.lin2(x))
        x = self.softmax(self.lin3(x))
        
        return x