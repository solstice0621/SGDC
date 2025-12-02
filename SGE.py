import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BottleneckRefiner(nn.Module):
    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        reduced = channels // reduction_ratio
        self.refine = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, reduced, 3, padding=1, bias=False),
            nn.BatchNorm2d(reduced),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, 1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.refine(x))


class SGE(nn.Module):
    """
    Structural Guidance Extractor (SGE)
    Extracts explicit high-frequency structural priors to guide SGDC.
    """
    def __init__(self, s4_in=2048, s1_in=256, mid=256, guide_ch=16,
                 bottleneck=256, reduction_ratio=4):
        super().__init__()

        self.s4_proj = ConvBNReLU(s4_in, bottleneck, ks=1, padding=0)
        self.s1_proj = ConvBNReLU(s1_in, bottleneck, ks=1, padding=0)

        self.sobel_x, self.sobel_y = self._create_sobel_layers(bottleneck)
        self.refine4 = BottleneckRefiner(bottleneck, reduction_ratio)
        self.refine1 = BottleneckRefiner(bottleneck, reduction_ratio)

        self.fuse = ConvBNReLU(bottleneck * 2, mid)
        self.out_edge_map = nn.Conv2d(mid, 1, 1, bias=False)
        self.out_guidance = nn.Conv2d(mid, guide_ch, 1, bias=False)

    def _create_sobel_layers(self, ch):
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32)
        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32)

        sobel_x = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        sobel_y = nn.Conv2d(ch, ch, 3, padding=1, bias=False)

        weight_x = torch.zeros(ch, ch, 3, 3)
        weight_y = torch.zeros(ch, ch, 3, 3)
        for i in range(ch):
            for j in range(ch):
                weight_x[i, j] = kx
                weight_y[i, j] = ky

        sobel_x.weight = nn.Parameter(weight_x, requires_grad=False)
        sobel_y.weight = nn.Parameter(weight_y, requires_grad=False)
        return sobel_x, sobel_y

    def _apply_sobel(self, x, refiner):
        gx = self.sobel_x(x)
        gy = self.sobel_y(x)
        mag = torch.sqrt(gx ** 2 + gy ** 2)
        mod = torch.sigmoid(mag) * x
        return refiner(mod)

    def forward(self, x4, x1):
        s4 = self.s4_proj(x4)
        s1 = self.s1_proj(x1)

        s4 = self._apply_sobel(s4, self.refine4)
        s1 = self._apply_sobel(s1, self.refine1)

        s4 = F.interpolate(s4, size=s1.shape[2:], mode='bilinear', align_corners=False)
        fused = self.fuse(torch.cat([s4, s1], dim=1))

        edge_map = self.out_edge_map(fused)
        guidance = self.out_guidance(fused)
        return edge_map, guidance
