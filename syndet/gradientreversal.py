import torch
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
        coral_x = x 
        coral_x = self.leaky_relu(self.conv1(coral_x))
        coral_x = self.leaky_relu(self.conv2(coral_x))
        coral_x = self.leaky_relu(self.conv3(coral_x))
        coral_x = self.leaky_relu(self.fc1(coral_x.view(coral_x.size(0), -1)))
        coral_x = F.log_softmax(self.fc2(coral_x), 1)
        return coral_x
