import torch.nn as nn

class CSAttention(nn.Module):
    def __init__(self,dim,hidden_dim):
        super(CSAttention,self).__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim,
                              kernel_size=1)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.linear_1 = nn.Linear(in_features=dim, out_features=hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        b,c,h,w = x.shape
        y = self.GAP(x)
        y = y.reshape(b,c)
        y = self.linear_1(y)
        y = self.relu(y)
        y = self.linear_2(y)
        y = self.sigmoid(y)
        y = y.reshape(b,c,1,1)
        z = self.conv(x)
        z = z * y
        x = x + z
        return x



