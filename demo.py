import numpy as np
import torch
import torch.nn as nn
data = np.arange(1,13).reshape(3,2,2)
print(data)
data = data.reshape(3,2*2).transpose(1,0)
print(data)
data = np.arange(1,10).reshape(3,3,1,1)
print(data)
print(data.reshape(3,3,1,1))
data = torch.randn(32,512,4,4)
gap = nn.AdaptiveAvgPool2d(1)
print(gap(data).shape)
data = torch.tensor([1.5,-0.8,-0.1,0.2,2.8])
print(torch.abs(data))