import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, einsum
from timm.models.layers import DropPath


class LayerNorm2d(nn.LayerNorm):
    def __init__(self, dim):
        super().__init__(dim, eps=1e-6)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        return rearrange(x, 'b h w c -> b c h w')


class SGDC(nn.Module):
    """
    SGDC (Structure-Guided Dynamic Convolution)
    A pooling-free dynamic kernel modulation guided by structural priors.
    """
    def __init__(self, dim, guide_ch=16, num_heads=4, kernel_size=7,
                 mlp_ratio=4., drop_path=0., ls_init=1e-6):
        super().__init__()

        self.dim = dim
        self.h = num_heads
        self.k = kernel_size
        self.k_area = kernel_size * kernel_size

        self.proj_guidance = nn.Conv2d(guide_ch, dim, 1)
        self.attn = nn.Conv2d(dim, dim * 2 + self.h * self.k_area, 1)

        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )

        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()
        self.ls1 = nn.Parameter(ls_init * torch.ones(dim))
        self.ls2 = nn.Parameter(ls_init * torch.ones(dim))

    def forward(self, x, g):
        identity = x

        g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g = self.proj_guidance(g)
        w = self.attn(x + g)

        attn1, attn2, kernel = torch.split(w, [self.dim, self.dim, self.h * self.k_area], dim=1)

        x1 = self.dwconv(x * torch.sigmoid(attn1))

        patches = F.unfold(self.norm1(x), kernel_size=self.k, padding=self.k // 2)
        patches = rearrange(patches, 'b (h c k) l -> b h c k l',
                            h=self.h, c=self.dim // self.h, k=self.k_area)

        kernel = rearrange(kernel, 'b (h k) hgt wdt -> b h k (hgt*wdt)', h=self.h)
        kernel = F.softmax(kernel, dim=2)

        out = einsum(patches, kernel, 'b h c k l, b h k l -> b h c l')
        out = rearrange(out, 'b h c l -> b (h c) l')
        out = F.fold(out, output_size=x.shape[-2:], kernel_size=1)

        x2 = out * torch.sigmoid(attn2)
        y = x1 + x2

        x = identity + self.drop_path(self.ls1.view(1, -1, 1, 1) * y)

        y2 = self.mlp(self.norm2(x))
        return x + self.drop_path(self.ls2.view(1, -1, 1, 1) * y2)
