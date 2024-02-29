import torch 
import numpy as np

def multi_scale_align(backbone) -> torch.Tensor:
    hcf = np.gcd(np.gcd(backbone[0].shape[2], backbone[1].shape[2]), backbone[2].shape[2])
    
    bacb_list = []
    
    for bacb in backbone: 
        for i in range(0, bacb.shape[2] // hcf):
            bacb_list.append(bacb[:, :, hcf*i : hcf*(i+1), hcf*i : hcf*(i+1)])
    
    backbones_feat = torch.cat(bacb_list, dim=1)
    
    return backbones_feat