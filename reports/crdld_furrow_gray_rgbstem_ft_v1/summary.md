# CCRDNet evaluation summary

- checkpoint: `runs/crdld_furrow_gray_rgbstem_ft_v1/best.pt` (epoch 27)
- split: test, 430 frames (430 with a GT line)
- detection rate: 1.0000

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 10.539 | 8.001 | 29.336 |
| lateral near (px) | 34.04 | 25.71 | 100.08 |
| line IoU | 0.0972 | 0.0382 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.2186

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).
