#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 16:31
# @Author  : Chenhao
# @File    : adsr.py
# @Software: PyCharm
# @Done    : 
# 1.以RDN网络模型提取低频信息；
# 2. 通过HR与SR相减，获取残余信息；
# 3. CA提取高频信息；
# 4.两个信息相加，通过卷积得到最终图像

from model import common

import torch
import torch.nn as nn

def make_model(args, parent=False):
    return RDN(args)


class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


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

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64)
        }[args.RDNconfig]

        self.sub_mean = common.MeanShift(args.rgb_range)

        # Shallow feature extraction net
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

        self.hr_conv = nn.Conv2d(args.n_colors, G, kSize, padding=(kSize - 1) // 2, stride=1)

        self.sub_feature = nn.Sequential(*[
            nn.Conv2d(G, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.Conv2d(G, G, kSize, padding=(kSize-1)//2, stride=1),
            CALayer(G),
            nn.Conv2d(G, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.Conv2d(G, G, kSize, padding=(kSize - 1) // 2, stride=1),
        ])

        self.final_conv = nn.Conv2d(G, args.n_colors, kSize, padding=(kSize - 1) // 2, stride=1)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

    # x[lr,hr]
    def forward(self, x):
        if isinstance(x,list):
            temp = self.sub_mean(x[0])
        else:
            temp = self.sub_mean(x)

        f__1 = self.UPNet2(temp)
        f__2 = self.SFENet(temp)

        RDBs_out = []
        for i in range(self.D):
            f__2 = self.RDBs[i](f__2)
            RDBs_out.append(f__2)

        f__2 = self.GFF(torch.cat(RDBs_out, 1))
        f__2 = self.UPNet1(f__2)
        f__2 += f__1

        if isinstance(x, list):
            x[1] = self.hr_conv(x[1])
            Sub_SR = x[1] - f__2
        else:
            Sub_SR = f__2

        Sub_SR = self.sub_feature(Sub_SR)
        SR = f__2 + Sub_SR
        SR = self.final_conv(SR)
        return self.add_mean(SR)

