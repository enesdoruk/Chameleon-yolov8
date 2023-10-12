from PIL import Image

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary
import torchvision.transforms as transforms

import os
import sys
sys.path.insert(0, os.path.expanduser('~') + "/syndet-yolo-grl")

from syndet.gradientreversal import ReverseLayerF
from syndet.modules import (DFL, Concat, Upsample, Detect, Conv, Bottleneck, C2f, SPPF)


class DetectionModel(nn.Module):
    def __init__(self):
        self.layers = []
        self.layers_grl = []
        super(DetectionModel, self).__init__()

        self.layers.append(Conv(3, 64, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(Conv(64, 128, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(128, 128, n=1, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(128, 256, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(256, 256, n=1, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(256, 512, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(512, 512, n=1, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(512, 1024, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(C2f(1024, 1024, n=1, shortcut=True, g=1, e=0.5))
        self.layers.append(SPPF(1024, 1024, k=5))

        self.layers.append(Upsample(1024, 2))
        self.layers.append(Concat())
        self.layers.append(C2f(1536, 512, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Upsample(512, 2))
        self.layers.append(Concat())
        self.layers.append(C2f(768, 256, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(256, 256, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(Concat())
        self.layers.append(C2f(768, 512, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Conv(512, 512, k=3, s=2, p=1, g=1, d=1, act=True))
        self.layers.append(Concat())
        self.layers.append(C2f(1536, 1024, n=3, shortcut=True, g=1, e=0.5))

        self.layers.append(Detect(nc=4, ch=(1024, 512, 256)))

        self.model = nn.Sequential(*self.layers)

        m = self.model[-1]  
        if isinstance(m, (Detect)):
            s = 256 
            m.inplace = True
            forward = lambda x:  self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])  
            self.stride = m.stride
            m.bias_init() 
        
        self.layers_grl.append(nn.Flatten())
        self.layers_grl.append(nn.Linear(1024*10*10, 1024))
        self.layers_grl.append(nn.BatchNorm1d(1024))
        self.layers_grl.append(nn.ReLU())
        self.layers_grl.append(nn.Linear(1024, 128))
        self.layers_grl.append(nn.ReLU())
        self.layers_grl.append(nn.Linear(128, 2))
        self.layers_grl.append(nn.LogSoftmax(dim=1))

        self.model_dom = nn.Sequential(*self.layers_grl)

    def forward(self, x, target=None, alpha=1.):      
        b1 = self.model[0](x)
  
        b2 = self.model[1](b1)
        b3 = self.model[2](b2)

        b4 = self.model[3](b3)
        b5 = self.model[4](b4)

        b6 = self.model[5](b5)
        b7 = self.model[6](b6)

        b8 = self.model[7](b7)
        b9 = self.model[8](b8)
        b10 = self.model[9](b9)

        h11 = self.model[10](b10)
        h12 = self.model[11]((h11,b7))
        h13 = self.model[12](h12)

        h14 = self.model[13](h13)
        h15 = self.model[14]((h14,b5))
        h16 = self.model[15](h15)

        h17 = self.model[16](h16)
        h18 = self.model[17]((h17,h13))
        h19 = self.model[18](h18)

        h20 = self.model[19](h19)
        h21 = self.model[20]((h20,b10))
        h22 = self.model[21](h21)
        h23 = self.model[22]([h22, h19, h16])      

        if target is not None:
            b1_t = self.model[0](target)
    
            b2_t = self.model[1](b1_t)
            b3_t = self.model[2](b2_t)

            b4_t = self.model[3](b3_t)
            b5_t = self.model[4](b4_t)

            b6_t = self.model[5](b5_t)
            b7_t = self.model[6](b6_t)

            b8_t = self.model[7](b7_t)
            b9_t = self.model[8](b8_t)
            b10_t = self.model[9](b9_t)
            
            b10_view = b10.view(-1, 1024*10*10)
            b10_t_view = b10_t.view(-1, 1024*10*10)
            reverse_feat = ReverseLayerF.apply(b10_view, alpha)
            reverse_feat_tar = ReverseLayerF.apply(b10_t_view, alpha)  
            
            reverse = torch.cat([reverse_feat, reverse_feat_tar])       
            
            domain_cls_0 = self.model_dom[0](reverse)
            domain_cls_1 = self.model_dom[3](self.model_dom[2](self.model_dom[1](domain_cls_0)))
            domain_cls_2 = self.model_dom[5](self.model_dom[4](domain_cls_1))
            domain_output = self.model_dom[7](self.model_dom[6](domain_cls_2))
            
            return h23, domain_output      
        else:
            return h23



if __name__ == "__main__":
    img_np = Image.open('test.png')

    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()])
    
    img = transform(img_np)
    img = img.unsqueeze(0).to("cuda")

    model = DetectionModel().to("cuda")
    model.eval()
    
    model_disc = DomainAdaptationModel(num_classes=1, input_channels=1024).to("cuda")
    model_disc.eval()

    #summary(model, (3,640,640))
    #print(model))

    with torch.no_grad():
        out, source_feat, target_feat = model(img, img)
        print(source_feat.shape)
        
        out_disc_source = model_disc(source_feat, 2) 
        out_disc_target = model_disc(target_feat, 2) 
        print(out_disc_source)
        print(out_disc_target)     

    """ import mmcv
    from mmengine.visualization import Visualizer
    img_mm = mmcv.imread('test.png', channel_order='rgb')
    visualizer = Visualizer()
    drawn_img = visualizer.draw_featmap(out[0], img_mm, channel_reduction='select_max')
    visualizer.show(drawn_img) """


    """ from torchview import draw_graph
    model_graph = draw_graph(model, input_size=(1,3,416,416), device='meta', save_graph=True, filename='arch', roll=True, hide_inner_tensors=False,
    hide_module_functions=False)#, expand_nested=True)
    model_graph.visual_graph """