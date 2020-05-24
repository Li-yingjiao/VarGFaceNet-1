import torch
from torch import nn

'''
求Input的二范数，为其输入除以其模长
角度蒸馏Loss需要用到
'''
def l2_norm(input, axis=1):
    norm  = torch.norm(input, axis, keepdim=True) # 默认p=2
    output = torch.div(input, norm)
    return output


'''
变组卷积，S表示每个通道的channel数量
'''
def VarGConv(in_channels, out_channels, kernel_size, stride, S):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, groups=in_channels//S, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.PReLU()
    )

'''
pointwise卷积，这里的kernelsize都是1，不过这里也要分组吗？？
'''
def PointConv(in_channels, out_channels, stride, S, isPReLU):
    return nn.Sequential(

    )

'''
SE block
'''
class SqueezeAndExcite(nn.Module):
    def __init__(self, in_channels, out_channels, divide=4):
        super(SqueezeAndExcite, self).__init__()
        mid_channels = in_channels // divide
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.SEblock = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=mid_channels),
            nn.ReLU6(inplace=True),
            nn.Linear(in_features=mid_channels, out_features=out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        out = self.pool(x)
        out = out.view(b, -1)
        out = self.SEblock(out)
        out = out.view(b, c, 1, 1)
        return out * x

'''
normal block
'''
class NormalBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, S=8):
        super(NormalBlock, self).__init__()
        out_channels = 2 * in_channels
        self.vargconv1 = VarGConv(in_channels, out_channels, kernel_size, stride, S)
        self.pointconv1 = PointConv(out_channels, in_channels, stride, S, isPReLU=True)

        self.vargconv2 = VarGConv(in_channels, out_channels, kernel_size, stride, S)
        self.pointconv2 = PointConv(out_channels, in_channels, stride, S, isPReLU=False)

        self.se = SqueezeAndExcite(in_channels, in_channels)