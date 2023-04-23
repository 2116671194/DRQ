import torch.nn as nn
import multiHeadAttention as MHA

class TransformBlock(nn.Module):
    def __init__(self,dim_in,dim_mh_k,dim_mh_v,hidden_dim,dim_out):
        super(TransformBlock,self).__init__()
        self.dim_in = dim_in
        self.dim_mh_k = dim_mh_k
        self.dim_mh_v = dim_mh_v
        self.hidden_dim = hidden_dim
        self.dim_out = dim_out
        self.LN = nn.LayerNorm(self.dim_in)
        self.mha = MHA.MHAttention(dim_in=dim_in,dim_k=dim_mh_k,dim_v=dim_mh_v)
        self.LN_1 = nn.LayerNorm(self.dim_mh_v)
        self.linear_1 = nn.Linear(in_features=dim_mh_v,out_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(in_features=hidden_dim,out_features=dim_out)
    def forward(self,x):
        # x的输入是[32,512,4,4]->[32,16,512]
        b, c, h, w = x.shape
        # [32,16,512]
        x = x.reshape(b, c, h * w).transpose(1, 2)
        x = x + self.mha(self.LN(x))
        x = x + self.linear_2(self.relu(self.linear_1(self.LN_1(x))))
        x = x.transpose(1,2).reshape(b,c,h,w)
        return x
