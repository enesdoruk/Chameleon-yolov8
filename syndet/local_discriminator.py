import torch.nn as nn
import torch


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super(LocalDiscriminator, self).__init__()

        self.fc1 = nn.Linear(1024, 1)
        
        self.sigmoid = nn.Sigmoid()
        self.logit_loss = nn.BCEWithLogitsLoss()

        
    def forward(self, x, mode):
        loss_sum = 0
        for i in range(x.shape[2]):
            for j in range(x.shape[2]):
                out = self.sigmoid(self.fc1(x[:,:,i,j]))
                                
                if mode == "source":
                    source_labels = torch.zeros(x.shape[0]).type(torch.float16).to('cuda').unsqueeze(1)
                    loss_sum  += self.logit_loss(out, source_labels)
                elif mode == "target":
                    target_labels = torch.ones(x.shape[0]).type(torch.float16).to('cuda').unsqueeze(1)
                    loss_sum  += self.logit_loss(out, target_labels)
                else:
                    print('wrong Mode....')
                    
        return loss_sum / (x.shape[2] * x.shape[2])            
                    
                    
