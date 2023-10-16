import torch
import torch.nn as nn
from collections import OrderedDict


class DCCAModule(nn.Module):
    """
    Domain Conditioned Channel Attention module.
    Capture domain-specific information in low-level stage and model the independencies
    between the convolutional channels for source and target respectively.
    """

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
        if self.training:
            src = self.fc0(x[:int(x.size(0) / 2), ])
            # src = self.fc1(x[:int(x.size(0) / 2), ])
            trg = self.fc1(x[int(x.size(0) / 2):, ])
            x = torch.cat((src, trg), 0)
        else:
            x = self.fc1(x)

        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """

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



class DCCAResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Domain Conditioned Channel Attention module.
    It follows Caffe implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """

    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(DCCAResNetBottleneck, self).__init__()
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



class DCCANet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1):
        super(DCCANet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64, track_running_stats=True)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64, track_running_stats=True)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes, track_running_stats=True)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(1, inplanes, kernel_size=20, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes, track_running_stats=True)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True` is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None

        self.feature = nn.Sequential(self.layer0, self.layer1, self.layer2, self.layer3, self.layer4)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion, track_running_stats=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        # x = self.features(x)
        x = self.feature(x)
        #x = self.logits(x)
        return x



if __name__ == "__main__":
    model = DCCANet(DCCAResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
                    dropout_p=None, inplanes=1024, input_3x3=False,
                    downsample_kernel_size=1, downsample_padding=0).to('cuda')
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    
    input_img_1 = np.random.rand(20,20,1024).transpose(2,1,0)
    input_img_tensor_1 = torch.from_numpy(input_img_1).unsqueeze(1).float().to('cuda')
    output_1 = model(input_img_tensor_1)
    print("1: ", output_1.shape)
    
    # input_img_2 = np.random.rand(40,40,512).transpose(2,1,0)
    # input_img_tensor_2 = torch.from_numpy(input_img_2).unsqueeze(0).float().to('cuda')
    # output_2 = model(input_img_tensor_2)
    # print("2: ", output_2.shape)
    
    # input_img_3 = np.random.rand(80,80,256).transpose(2,1,0)
    # input_img_tensor_3 = torch.from_numpy(input_img_3).unsqueeze(0).float().to('cuda')
    # output_3 = model(input_img_tensor_3)
    # print("3: ", output_3.shape)
    
    
    input_img_2 = np.random.rand(40,40,512).transpose(2,1,0)
    input_img_tensor_2 = torch.from_numpy(input_img_2).unsqueeze(0).float().to('cuda')
    model_dca = DCCAModule(channels=512, reduction=16).to('cuda')
    output = model_dca(input_img_tensor_2)
    print(output.shape)
    
    
    model_bott = DCCAResNetBottleneck(groups=1, reduction=16, inplanes=512, planes=128).to('cuda')
    print(list(model_bott.children()))
    output2 = model_bott(input_img_tensor_2)
    output2_sq = output2.squeeze(0).detach().cpu()
    
    activation = {}
    def get_activation(name):
        def hook(module, input, output):
            activation[name] = output
        return hook
    
    print(model_bott.conv1.weight[0])
    print(model_bott.get_layer("conv1"))
    # target_layer = model_bott.conv3
    # target_layer.register_forward_hook(get_activation("target_layer"))
    # output2 = model_bott(input_img_tensor_2)
    # activation_map = activation["target_layer"].squeeze(0)
    # channel = 0
    # plt.imshow(activation_map[channel].cpu().detach().numpy(), cmap='viridis')
    # plt.axis('off')
    # plt.show()
    
    # gray_scale = torch.sum(output2_sq,0)
    # gray_scale = gray_scale / output2_sq.shape[0]
    # plt.imshow(gray_scale)
    # plt.show()

    # print(output2.shape)
    # print(gray_scale.shape)
    
    
    """ from torchview import draw_graph
    model_graph = draw_graph(model, input_size=(1,3,416,416), device='meta', save_graph=True, filename='arch', roll=True, hide_inner_tensors=False,
    hide_module_functions=False)#, expand_nested=True)
    model_graph.visual_graph """