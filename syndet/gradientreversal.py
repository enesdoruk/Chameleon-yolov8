import torch.nn as nn
import torch.nn.functional as F


class DomainDiscriminator(nn.Module):
    def __init__(self) -> None:
        super(DomainDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(51200, 512)
        self.fc2 = nn.Linear(512, 2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
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
        soft = x
        log_soft = x
        log_soft = F.log_softmax(log_soft, 1)
        soft = F.softmax(soft,1)
        return soft, log_soft
