# AgriCCRDNet-v0 — architecture reproduction notes

Task 3 deliverable (spec sections 8–9, 38). This documents every inference made
where the CCRDNet paper (Zheng & Wang 2026, doi:10.3389/fpls.2026.1744637) is
silent, so the implementation is honestly a **reproduction, not an exact
reimplementation**.

## Stated by the paper (implemented as written)

| Item | Paper | Implementation |
|---|---|---|
| Topology | U-shaped encoder–decoder with skips | `perception/ccrdnet/model.py` |
| DSC block | 3×3 grouped conv → GELU → 1×1 pointwise → BN → GELU | `blocks.DSCBlock` |
| ASPP | dilated 3×3 at rates 2/4/6/8, concat → 1×1 projection, grouped convs, on the last three skip connections | `blocks.LiteASPP`, wired to the 3 deepest skips |
| Input / output | 256×256 RGB → 3-class map | default config |
| Scale | 33.621 K params / 38.226 M FLOPs | **33,694 params / 37.61 M MACs** (−1.6 %) |

## Inferred (not stated in the paper)

1. **Grouped 3×3 = depthwise** (`groups = in_channels`) — the standard
   depthwise-separable reading; matches the parameter scale.
2. **One DSC block per scale**, MaxPool 2×2 downsample, bilinear 2× upsample.
3. **Stem downsample**: the ladder starts after one 2× pool and the head logits
   are bilinearly upsampled back to input size. Without this, any channel
   ladder reaching ≥33 K parameters costs ≥80 M MACs because the two
   full-resolution stages dominate compute; the paper's 33.6 K / 38.2 M ratio
   is only reachable with the ladder starting at 128×128. (Search evidence:
   full-res variants gave 14–27 K params at 72–171 M MACs.)
4. **Channel ladder (12, 20, 36, 56, 80)** chosen by grid search to match the
   paper's reported params/FLOPs within 2 %.
5. **ASPP projection grouped with 4 groups** — the paper says ASPP "uses
   grouped convolutions to reduce parameters" without giving the group count.
6. **FLOPs convention**: the paper's 38.226 M "FLOPs" for this size matches the
   thop-style MAC count; we report MACs and note ×2 for strict FLOPs.

## Ablation switches (experiment matrix B, spec §27)

- `use_dsc=False` → plain 3×3 conv blocks (B0/B2)
- `aspp_skip_count=0` → no ASPP (B0/B1)

## Verification

`farm_robot/tests/test_ccrdnet_shapes.py` (shape/scale/gradients, ±10 % scale
gate) and `test_ccrdnet_postprocess.py`, `test_ccrdnet_geometry.py` — 24 tests
passing on 2026-08-30.
