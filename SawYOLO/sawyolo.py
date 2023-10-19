import os
import sys
sys.path.insert(0, os.path.expanduser('~') + "/syndet-yolo-dcan")

import cv2
import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage import zoom
from mmengine.visualization import Visualizer

from SawYOLO.head import Head
from SawYOLO.modules import Detect
from SawYOLO.backbone import Backbone
from SawYOLO.DCAN import DCCABottleneck, FeatureCorrection



class DetectionModel(nn.Module):
    def __init__(self):
        super(DetectionModel, self).__init__()

        self.layers = []
        self.layers.append(Backbone())
        
        self.layers.append(DCCABottleneck(groups=1, reduction=16, inplanes=256, planes=64))
        self.layers.append(FeatureCorrection(channels=256, reduction=4))
        
        self.layers.append(DCCABottleneck(groups=1, reduction=16, inplanes=512, planes=128))
        self.layers.append(FeatureCorrection(channels=512, reduction=4))
        
        self.layers.append(DCCABottleneck(groups=1, reduction=16, inplanes=1024, planes=256))
        self.layers.append(FeatureCorrection(channels=1024, reduction=4))
        
        self.layers.append(Head())
        
        self.model = nn.Sequential(*self.layers)
        
        m = self.model[-1].detect
        if isinstance(m, (Detect)):
            s = 256 
            m.inplace = True
            forward = lambda x:  self.forward(x, target=None, verbose=False)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])  
            self.stride = m.stride
            m.bias_init()  


    def forward(self, x, target=None, verbose=False, save_path=None):
        input = x
        if target is not None:
            x = torch.cat((x, target), 0)

        backb5, backb7, backb10 = self.model[0](x)
            
        dcanet_b5 = self.model[1](backb5)
        feat_corr_b5 = self.model[2](dcanet_b5)
        
        dcanet_b7 = self.model[3](backb7)
        feat_corr_b7 = self.model[4](dcanet_b7)
        
        dcanet_b10 = self.model[5](backb10)
        feat_corr_b10 = self.model[6](dcanet_b10)
        
        # head = self.model[7](feat_corr_b5[:int(feat_corr_b5.size(0) / 2), ], 
        #                      feat_corr_b7[:int(feat_corr_b7.size(0) / 2), ], 
        #                      feat_corr_b10[:int(feat_corr_b10.size(0) / 2), ])
        
        head = self.model[7](feat_corr_b5, 
                             feat_corr_b7, 
                             feat_corr_b10)
        
        if target is not None and verbose is True:
            visualizer = Visualizer(vis_backends=[dict(type='LocalVisBackend')], save_dir=os.getcwd())
            img = np.array(input[0].cpu(), dtype=np.uint8).transpose(2,1,0)
            img = cv2.resize(img, (320, 320))
            
            feat_corr_b5 = zoom(feat_corr_b5[0].to(torch.float32).cpu().detach().numpy(), (1, 4, 4), order=1)
            feat_corr_b5 = torch.tensor(feat_corr_b5).to('cuda')
            drawn_img_b5 = visualizer.draw_featmap(feat_corr_b5, img, channel_reduction='select_max')
            
            feat_corr_b7 = zoom(feat_corr_b7[0].to(torch.float32).cpu().detach().numpy(), (1, 8, 8), order=1)
            feat_corr_b7 = torch.tensor(feat_corr_b7).to('cuda')
            drawn_img_b7 = visualizer.draw_featmap(feat_corr_b7, img, channel_reduction='select_max')
            
            feat_corr_b10 = zoom(feat_corr_b10[0].to(torch.float32).cpu().detach().numpy(), (1, 16, 16), order=1)
            feat_corr_b10 = torch.tensor(feat_corr_b10).to('cuda')
            drawn_img_b10 = visualizer.draw_featmap(feat_corr_b10, img, channel_reduction='select_max')
            
            act_img = cv2.hconcat([drawn_img_b5, drawn_img_b7, drawn_img_b10]) 
            cv2.imwrite(save_path, act_img)
            
        return head


if __name__ == "__main__":
    input_img = cv2.imread("/home/adastec/syndet-yolo-dcan/SawYOLO/test.png")
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (640,640)).transpose(2,1,0)
    input_img_tensor = torch.from_numpy(input_img).unsqueeze(0).float().to('cuda')

    model = DetectionModel().to("cuda")
    model.train()
    out = model(input_img_tensor, input_img_tensor, verbose=True)
    
    
    # inp = cv2.imread("/home/adastec/syndet-yolo-dcan/SawYOLO/test.png")
    # inp = cv2.resize(inp, (80,80))
    # visualizer = Visualizer()
    # drawn_img = visualizer.draw_featmap(out[2][0], inp, channel_reduction='select_max')
    # visualizer.show(drawn_img)

    # from torchview import draw_graph
    # model_graph = draw_graph(model, input_size=(1,3,416,416), device='meta', save_graph=True, filename='arch', roll=True, hide_inner_tensors=False,
    # hide_module_functions=False)#, expand_nested=True)
    # model_graph.visual_graph