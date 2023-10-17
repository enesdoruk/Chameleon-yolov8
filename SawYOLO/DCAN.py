import torch
import torch.nn as nn


class DCCAModule(nn.Module):
    def __init__(self, channels, reduction):
        super(DCCAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        src = self.fc0(x[:int(x.size(0) / 2), ])
        trg = self.fc1(x[int(x.size(0) / 2):, ])
        x = torch.cat((src, trg), 0)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
    
    
    
class DCCABottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(DCCABottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = DCCAModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out
    
    
class FeatureCorrection(nn.Module):
    def __init__(self, channels, reduction):
        super(FeatureCorrection, self).__init__()
        self.fc0 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        module_input = x[int(x.size(0) / 2):, ]
        trg = x[int(x.size(0) / 2):, ]

        trg = self.fc0(trg)
        trg = self.relu1(trg)
        trg = self.fc1(trg)
        trg = self.relu2(trg)
        
        x = torch.cat((x[:int(x.size(0) / 2), ], module_input + trg), 0)

        return  x  



if __name__ == '__main__':
    import numpy as np
    input_img_2 = np.random.rand(20,20,1024).transpose(2,1,0)
    input_img_tensor_2 = torch.from_numpy(input_img_2).unsqueeze(0).float().to('cuda')
    
    model = DCCABottleneck(groups=1, reduction=16, inplanes=1024, planes=256).to('cuda')
    output = model(input_img_tensor_2)