import torch 
import torch.nn as nn

    
class MultiScaleAlig(nn.Module):
    def __init__(self) -> None:
        super(MultiScaleAlig, self).__init__()
        
        self.br2_1 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=2, dilation=2)
        self.prelu2_1 = nn.PReLU()
        
        self.br3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=4, padding=2, dilation=2)
        self.prelu3_1 = nn.PReLU()
        
        self.conv = nn.Conv2d(1024+512+256, 1024+512+256, kernel_size=1)
        self.prelu = nn.PReLU()
            
    def forward(self, in_br1, in_br2, in_br3) -> torch.Tensor: 
        import pdb; pdb.set_trace()
        out_br2_1 = self.prelu2_1(self.br2_1(in_br2))
        
        out_br3_1 = self.prelu3_1(self.br3_1(in_br3))
        
        out_cat = torch.cat([in_br1, out_br2_1, out_br3_1], dim=1)
        
        out_conv = self.prelu(self.conv(out_cat))
   
        return out_conv