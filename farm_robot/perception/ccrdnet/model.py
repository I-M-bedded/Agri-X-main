# -*- coding: utf-8 -*-
"""CCRDNet reproduction (AgriCCRDNet-v0).

This is a *reproduction*, not an exact reimplementation: the paper does not
publish channel counts or the down/upsample operators, so those are inferred
and fixed in :class:`CCRDNetConfig` (spec sections 8-9).  Output logits are
``B x num_classes x H x W`` at the input resolution.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from perception.ccrdnet.blocks import DSCBlock, LiteASPP, PlainConvBlock
from perception.ccrdnet.config import CCRDNetConfig


class CCRDNet(nn.Module):
    def __init__(self, config: CCRDNetConfig = CCRDNetConfig()):
        super().__init__()
        if config.replicate_grayscale_input and config.in_channels != 1:
            raise ValueError("replicate_grayscale_input requires in_channels=1")
        self.config = config
        block = DSCBlock if config.use_dsc else PlainConvBlock
        channels = config.encoder_channels
        if len(channels) < 3:
            raise ValueError("encoder_channels needs at least 3 stages")

        # Encoder: stage 0 at full resolution, each later stage after a 2x pool.
        self.encoder = nn.ModuleList()
        previous = 3 if config.replicate_grayscale_input else config.in_channels
        for width in channels:
            self.encoder.append(block(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)

        # ASPP on the deepest `aspp_skip_count` skip connections.  Skips come
        # from every encoder stage except the deepest (bottleneck) stage.
        skip_channels = list(channels[:-1])
        self.aspp = nn.ModuleDict()
        if config.aspp_skip_count > 0 and config.aspp_rates:
            for index in range(
                max(0, len(skip_channels) - config.aspp_skip_count), len(skip_channels)
            ):
                self.aspp[str(index)] = LiteASPP(
                    skip_channels[index],
                    rates=config.aspp_rates,
                    projection_groups=config.aspp_projection_groups,
                )

        # Decoder mirrors the encoder ladder: bilinear 2x up, concat skip, block.
        self.decoder = nn.ModuleList()
        previous = channels[-1]
        for width in reversed(skip_channels):
            self.decoder.append(block(previous + width, width))
            previous = width
        self.head = nn.Conv2d(previous, config.num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: List[torch.Tensor] = []
        if self.config.replicate_grayscale_input:
            x_model = x.repeat(1, 3, 1, 1)
        else:
            x_model = x
        out = self.pool(x_model) if self.config.stem_downsample else x_model
        for index, stage in enumerate(self.encoder):
            if index > 0:
                out = self.pool(out)
            out = stage(out)
            if index < len(self.encoder) - 1:
                skips.append(out)

        for index, stage in enumerate(self.decoder):
            skip_index = len(skips) - 1 - index
            skip = skips[skip_index]
            key = str(skip_index)
            if key in self.aspp:
                skip = self.aspp[key](skip)
            out = F.interpolate(out, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            out = stage(torch.cat([out, skip], dim=1))
        logits = self.head(out)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits, size=x.shape[-2:], mode="bilinear", align_corners=False
            )
        return logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_size: Tuple[int, int] = (256, 256)) -> int:
    """Multiply-accumulate count for the convolutions (thop-style 'FLOPs').

    The paper reports 38.226M FLOPs for 33.6K parameters, which matches the
    common tool convention of counting one MAC as one FLOP; multiply by two for
    the strict flop count.
    """

    totals: Dict[str, int] = {"macs": 0}
    hooks = []

    def conv_hook(module: nn.Conv2d, _inputs, output):
        out_elems = output.numel()
        per_element = (module.in_channels // module.groups) * (
            module.kernel_size[0] * module.kernel_size[1]
        )
        totals["macs"] += out_elems * per_element

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, model.config.in_channels, *input_size))
    for hook in hooks:
        hook.remove()
    if was_training:
        model.train()
    return totals["macs"]
