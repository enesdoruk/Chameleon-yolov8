from PIL import Image

import cv2
import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary
import torchvision.transforms as transforms

import os
import sys
sys.path.insert(0, "/AI/syndet-yolo-grl")

from syndet.gradientreversal import ReverseLayerF, DomainDiscriminator
from syndet.modules import (DFL, Concat, Upsample, Detect, Conv, Bottleneck, C2f, SPPF)
from syndet.backbone import Backbone
from syndet.head import Head

class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()
        
        self.layers = []
        self.layers.append(Backbone())
        self.layers.append(Head())
        self.model = nn.Sequential(*self.layers)

        m = self.model[-1].detect 
        if isinstance(m, (Detect)):
            s = 256 
            m.inplace = True
            forward = lambda x:  self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])  
            self.stride = m.stride
            m.bias_init() 
        
        self.layers_grl = []
        self.layers_grl.append(DomainDiscriminator())
        self.model_dom = nn.Sequential(*self.layers_grl)


    def forward(self, x, target=None, alpha=1.):      
        backb5, backb7, backb10 = self.model[0](x)
        
        head = self.model[1](backb5,
                             backb7, 
                             backb10)  

        if target is not None:
            backb5_t, backb7_t, backb10_t = self.model[0](target) 
    
            domain_output_s = self.model_dom[0](backb10, alpha)
            domain_output_t = self.model_dom[0](backb10_t, alpha)
            
            return head, domain_output_s, domain_output_t
        else:
            return head



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