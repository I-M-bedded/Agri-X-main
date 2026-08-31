# -*- coding: utf-8 -*-
"""Building blocks for the CCRDNet reproduction.

Only the two mechanisms the paper describes are implemented: the depthwise
separable convolution block and the lightweight ASPP.  No attention, SE, CBAM
or transformer modules (spec section 8).
"""

import torch
from torch import nn


class DSCBlock(nn.Module):
    """Paper-described DSC: 3x3 grouped conv -> GELU -> 1x1 pointwise -> BN -> GELU.

    The paper does not state the group count; depthwise (groups = in_channels)
    is the standard depthwise-separable reading and matches the reported
    parameter scale.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1,
            groups=in_channels, bias=False,
        )
        self.act1 = nn.GELU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act2 = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.depthwise(x))
        return self.act2(self.norm(self.pointwise(x)))


class PlainConvBlock(nn.Module):
    """3x3 full convolution stand-in for the DSC-off ablation (B0/B2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class LiteASPP(nn.Module):
    """Paper-described ASPP: parallel dilated 3x3 convs (rates 2/4/6/8),
    concatenated and fused by a 1x1 projection.

    The paper states ASPP uses grouped convolutions to reduce parameters, so
    the dilated branches are depthwise and the projection is grouped.  Output
    channels equal input channels so the module drops into a skip connection
    unchanged.
    """

    def __init__(self, channels: int, rates=(2, 4, 6, 8), projection_groups: int = 4):
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Conv2d(
                channels, channels, kernel_size=3, padding=rate, dilation=rate,
                groups=channels, bias=False,
            )
            for rate in rates
        )
        merged = channels * len(rates)
        valid = merged % projection_groups == 0 and channels % projection_groups == 0
        groups = projection_groups if valid else 1
        self.project = nn.Conv2d(merged, channels, kernel_size=1, groups=groups, bias=False)
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        merged = torch.cat([branch(x) for branch in self.branches], dim=1)
        return self.act(self.norm(self.project(merged)))
