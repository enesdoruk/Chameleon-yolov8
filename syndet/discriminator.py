import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_shape=32*18*18, cls_num=4):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(1024, 32, kernel_size=3)
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(input_shape, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, cls_num)
        
        self.relu1 = nn.ReLU()

        self.ln1 = nn.LayerNorm(1024)
        self.ln2 = nn.LayerNorm(128)

    def forward(self, feature):
        feat = self.relu1(self.conv1(feature))
        feat = self.flatten(feat)
                
        out = self.ln1(self.fc1(feat))
        out = self.ln2(self.fc2(out))
        out = self.fc3(out)
        
        return feat, out
      

