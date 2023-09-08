import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    

class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=1, input_dim=1024*20*20, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.lin3 = nn.Linear(hidden_dim, num_classes)
        self.act3 = nn.Sigmoid()
        
    
    def forward(self, features):
        x = self.flat(features)
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))
        x = self.act3(self.lin3(x))  
        return x


class DomainAdaptationModel(nn.Module):
    def __init__(self, num_classes, input_dim, alpha):
        super(DomainAdaptationModel, self).__init__()
        self.discriminator = DomainDiscriminator(num_classes=num_classes, input_dim=input_dim)
        self.reversed_features = GradientReversal(alpha)

    def forward(self, features):
        reversed_features = self.reversed_features(features)
        disc_output = self.discriminator(reversed_features)
        return disc_output
    
    
