import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(DoubleConv, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channel,out_channels=out_channel,
                    kernel_size=3,padding=1,stride=1)
        self.BN_1 = nn.BatchNorm2d(num_features=out_channel)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                    kernel_size=3, padding=1, stride=1)
        self.BN_2 = nn.BatchNorm2d(num_features=out_channel)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self,x):
        x = self.relu_1(self.BN_1(self.conv_1(x)))
        x = x + self.relu_2(self.BN_2(self.conv_2(x)))
        return x

