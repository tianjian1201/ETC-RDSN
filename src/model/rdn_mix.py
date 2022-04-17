# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797
'''
    1. 结合UNet网络模型思想，加入CA
    2. 输入图像去均值处理
    3. 改变尾部上采样模式
    4 mix field 每个RDB模块中加入CA
'''
from model import common
from .unet import UNet

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3, reduction=16):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

        self.calayer = nn.Sequential(*[
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(G, G // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(G // reduction, G, 1, padding=0, bias=True),
            nn.Sigmoid()
        ])

    def forward(self, x):
        out = self.conv(x)
        weight = self.calayer(out)

        return torch.cat((x, out * weight), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x


class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.SFENet = nn.Sequential(*[
            nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0=G0, growRate=G, nConvLayers=C)
            )

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
        ])

        # Channel Attention
        self.unet = UNet(args.n_colors, args.G0,subpixel=True)

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet1 = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(r)
            ])
            self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(args.n_colors, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(r)
            ])
        elif r == 4:
            self.UPNet1 = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2)
            ])
            self.UPNet2 = nn.Sequential(*[
                nn.Conv2d(args.n_colors, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
                nn.PixelShuffle(2)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

        self.final_conv = nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

    def forward(self, x):
        x = self.sub_mean(x)
        weight = self.unet(x)
        f__1 = self.UPNet2(x)
        x = self.SFENet(x)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            RDBs_out.append(x)

        x = self.GFF(torch.cat(RDBs_out, 1))
        x *= weight
        x = self.UPNet1(x)
        x += f__1
        x = self.final_conv(x)
        return self.add_mean(x)