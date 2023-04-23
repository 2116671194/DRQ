import torch.nn as nn
import torch

class SSAttention(nn.Module):
    def __init__(self,channels,bete=0.3):
        super(SSAttention,self).__init__()
        self.bete = bete
        self.d_conv = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv = nn.Conv2d(in_channels=channels,out_channels=channels,
                              kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        B,C,H,W = x.shape
        q = self.d_conv(x).reshape(B,H*W,C)
        k = self.d_conv(x).reshape(B,C,H*W)
        v = self.conv(x).reshape(B,H*W,C)
        qk = torch.bmm(q,k)
        mask = torch.where(torch.abs(qk)>self.bete,1.0,0)
        qk = qk * mask
        qk = self.softmax(qk)
        att = torch.bmm(qk,v)
        att = att.transpose(1,2).reshape(B,C,H,W)
        x = x + att
        return x



