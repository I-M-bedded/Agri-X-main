# CCRDNet evaluation summary

- checkpoint: `runs/crdld_furrow_gray_v1/best.pt` (epoch 31)
- split: test, 430 frames (430 with a GT line)
- detection rate: 1.0000

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 10.961 | 8.700 | 28.314 |
| lateral near (px) | 38.46 | 27.16 | 108.99 |
| line IoU | 0.0839 | 0.0320 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.2163

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).
