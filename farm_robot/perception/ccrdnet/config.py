# -*- coding: utf-8 -*-
"""Configuration for the AgriCCRDNet-v0 reproduction.

The CCRDNet paper (Zheng & Wang 2026, doi:10.3389/fpls.2026.1744637) does not
publish per-stage channel counts, block repetition counts, or the down/upsample
operators.  Everything the paper *does* state is fixed here; everything inferred
is kept in this single dataclass so an experiment is fully described by one
config instance (spec section 9).

Stated by the paper:
  - U-shaped encoder-decoder with skip connections
  - DSC block: 3x3 grouped conv -> GELU -> 1x1 pointwise -> BatchNorm -> GELU
  - ASPP with dilation rates 2/4/6/8, branch concat -> 1x1 projection,
    applied to the last three skip connections, grouped convs inside
  - input 256x256 RGB, 3 output classes
  - scale: ~33.6K parameters, ~38.2M FLOPs at 256x256

Inferred here (documented assumptions):
  - grouped 3x3 conv == depthwise (groups = in_channels)
  - one DSC block per scale, MaxPool 2x2 downsample, bilinear upsample
  - ASPP dilated branches are depthwise; the 1x1 projection is grouped
    (``aspp_projection_groups``) because the paper says ASPP uses grouped
    convolutions to reduce parameters
  - channel ladder chosen by a small search so parameters/FLOPs land near the
    paper's reported scale (see reports/ccrdnet_scale_search.md)
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class CCRDNetConfig:
    in_channels: int = 3
    num_classes: int = 3          # OTHER / STRUCTURE / NAV_BAND
    input_size: Tuple[int, int] = (256, 256)   # (H, W)

    # encoder_channels[0] is the full-resolution stem; each later entry is one
    # 2x downsample.  The decoder mirrors this ladder back up.
    # (12, 20, 36, 56, 80) -> 33,694 params / 37.61M MACs at 256x256, matching
    # the paper's 33,621 params / 38.226M FLOPs within 2%.
    encoder_channels: Tuple[int, ...] = (12, 20, 36, 56, 80)

    # ASPP (paper-fixed rates), applied to the deepest `aspp_skip_count` skip
    # connections; 0 disables ASPP entirely (ablation B0/B1).
    aspp_rates: Tuple[int, ...] = (2, 4, 6, 8)
    aspp_skip_count: int = 3
    aspp_projection_groups: int = 4

    # Ablation switch: False replaces every DSC block with a plain 3x3
    # convolution of the same in/out channels (ablation B0/B2).
    use_dsc: bool = True

    # If True the stem pools the input 2x before the first block and the head
    # logits are bilinearly upsampled back to the input size.  Inferred: the
    # paper's 33.6K-parameter/38M-FLOP ratio is unreachable with full-resolution
    # first/last stages, so the ladder must start below input resolution.
    stem_downsample: bool = True

    # Class ids in the produced masks (kept with the model so training,
    # post-processing and target building cannot disagree).
    class_other: int = 0
    class_structure: int = 1
    class_nav_band: int = 2


PAPER_BASELINE = CCRDNetConfig()
