import torch.nn as nn
import torch
import math
class MHAttention(nn.Module):
    def __init__(self,dim_in,dim_k,dim_v):
        super(MHAttention,self).__init__()
        # 输入->[32,16,512]
        self.num_head = 8
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.scale = 1 / math.sqrt(self.dim_k // self.num_head)
        self.q = nn.Linear(in_features=dim_in,out_features=dim_k)
        self.k = nn.Linear(in_features=dim_in,out_features=dim_k)
        self.v = nn.Linear(in_features=dim_in,out_features=dim_v)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self,x):
        # x的输入是[32,16,512]
        b, s, e = x.shape
        # [32,16,1024]->[32,16,8,128]->[32,8,16,128]
        q = self.q(x).reshape(b,s,self.num_head,
                              self.dim_k // self.num_head).permute(0,2,1,3)
        k = self.k(x).reshape(b,s,self.num_head,
                              self.dim_k // self.num_head).permute(0,2,1,3)
        # [32,8,16,64]
        v = self.v(x).reshape(b,s,self.num_head,
                              self.dim_v // self.num_head).permute(0,2,1,3)
        # [32,8,16,16]
        dist = torch.matmul(q,k.transpose(2,3)) * self.scale
        dist = self.softmax(dist)
        # [32,8,16,64]->[32,16,8,64]->[32,16,512]
        output = torch.matmul(dist,v).transpose(1,2).reshape(b,s,self.dim_v)
        return output
