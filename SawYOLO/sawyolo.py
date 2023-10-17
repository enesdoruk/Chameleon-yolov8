import os
import sys
sys.path.insert(0, os.path.expanduser('~') + "/syndet-yolo-dcan")

import torch
import torch.nn as nn

from SawYOLO.head import Head
from SawYOLO.modules import Detect
from SawYOLO.backbone import Backbone
from SawYOLO.DCAN import DCCABottleneck, FeatureCorrection



class DetectionModel(nn.Module):
    def __init__(self):
        self.layers = []
        self.output = []
        super(DetectionModel, self).__init__()
        
        self.backbone = Backbone()
        self.head = Head()
        self.dcanet_backb5 = DCCABottleneck(groups=1, reduction=16, inplanes=256, planes=64)
        self.dcanet_backb7 = DCCABottleneck(groups=1, reduction=16, inplanes=512, planes=128)
        self.dcanet_backb10 = DCCABottleneck(groups=1, reduction=16, inplanes=1024, planes=256)
        
        self.featcorr_back5 = FeatureCorrection(channels=256, reduction=4)
        self.featcorr_back7 = FeatureCorrection(channels=512, reduction=4)
        self.featcorr_back10 = FeatureCorrection(channels=1024, reduction=4)
        
        m = self.head.model[-1]  
        if isinstance(m, (Detect)):
            s = 256 
            m.inplace = True
            forward = lambda x:  self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, 3, s, s))])  
            self.stride = m.stride
            m.bias_init()  


    def forward(self, x, target=None):
        if target is not None:
            x = torch.cat((x, target), 0)
            
        backb5, backb7, backb10 = self.backbone(x)
        
        dcanet_b5 = self.dcanet_backb5(backb5)
        feat_corr_b5 = self.featcorr_back5(dcanet_b5)
        
        dcanet_b7 = self.dcanet_backb7(backb7)
        feat_corr_b7 = self.featcorr_back7(dcanet_b7)
        
        dcanet_b10 = self.dcanet_backb10(backb10)
        feat_corr_b10 = self.featcorr_back10(dcanet_b10)
        
        head = self.head(feat_corr_b5, feat_corr_b7, feat_corr_b10)

        return head


if __name__ == "__main__":
    import numpy as np
    input_img = np.random.rand(640,640,3).transpose(2,1,0)
    input_img_tensor = torch.from_numpy(input_img).unsqueeze(0).float().to('cuda')

    model = DetectionModel().to("cuda")
    out = model(input_img_tensor)
        

    """ import mmcv
    from mmengine.visualization import Visualizer
    img_mm = mmcv.imread('test.png', channel_order='rgb')
    visualizer = Visualizer()
    drawn_img = visualizer.draw_featmap(out[0], img_mm, channel_reduction='select_max')
    visualizer.show(drawn_img) """


    # from torchview import draw_graph
    # model_graph = draw_graph(model, input_size=(1,3,416,416), device='meta', save_graph=True, filename='arch', roll=True, hide_inner_tensors=False,
    # hide_module_functions=False)#, expand_nested=True)
    # model_graph.visual_graph