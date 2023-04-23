import torch.nn as nn
import torch
import channelAttention as cA
import spatialAttention as sA

class HSAttention(nn.Module):
    def __init__(self,dim,hidden_dim):
        super(HSAttention,self).__init__()
        self.ca = cA.CSAttention(dim=dim,hidden_dim=hidden_dim)
        self.sa = sA.SSAttention(channels=dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,
                              kernel_size=3,stride=1,padding=1)
    def forward(self,x):
        s = self.sa(x)
        c = self.ca(x)
        x = torch.cat([s,c],dim=1)
        x = self.relu(x)
        x = self.conv(x)
        return x