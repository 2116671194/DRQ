import torch
import torch.nn as nn
import doubleNet as dn
import transform
import hybridAttention as hA

# U-net网络
class U_NET(nn.Module):
    def __init__(self,channel,features=[64,128,256,512]):
        super(U_NET, self).__init__()
        # 下采样
        self.down_sample = nn.ModuleList()
        # 上采样
        self.up_sample = nn.ModuleList()
        # 最大池化层
        self.max_pooling = nn.MaxPool2d(kernel_size=2,padding=0,stride=2)
        for feature in features:
            self.down_sample.append(dn.DoubleConv(channel,feature))
            channel = feature
        for feature in reversed(features):
            # 转置卷积计算公式：(i-1)*s -2*p + k(可用于上采用)
            self.up_sample.append(nn.ConvTranspose2d(
                in_channels=2*feature,out_channels=feature,
                kernel_size=2,stride=2
            ))
            self.up_sample.append(dn.DoubleConv(feature*2,feature))
        self.tf_1 = transform.TransformBlock(dim_in=features[-1],dim_mh_k=features[-1]*2,
                                             dim_mh_v=features[-1],hidden_dim=features[-1]*2,
                                             dim_out=features[-1])
        self.tf_2 = transform.TransformBlock(dim_in=features[-2],dim_mh_k=features[-2]*2,
                                             dim_mh_v=features[-2],hidden_dim=features[-2]*2,
                                             dim_out=features[-2])
        self.sAttention = hA.HSAttention(dim=features[-1],hidden_dim=features[-1] * 2)
        self.final_conv = nn.Conv2d(in_channels=features[0],out_channels=1,
                                    kernel_size=1)
    def forward(self,x):
        down_double_output = []
        for down in self.down_sample:
            x = down(x)
            down_double_output.append(x)
            x = self.max_pooling(x)
        x = self.sAttention(x)
        down_double_output_T = down_double_output[::-1]
        for idx in range(0,len(self.up_sample),2):
            x = self.up_sample[idx](x)
            d = down_double_output_T[idx//2]
            if idx // 2 == 0:
                d = self.tf_1(d)
            if idx // 2 == 1:
                d = self.tf_2(d)
            x = torch.cat((d,x),dim=1)
            x =self.up_sample[idx+1](x)
        x = self.final_conv(x)
        return x
data = torch.ones(32,1,64,64)
model = U_NET(1)
print(model(data).shape)
