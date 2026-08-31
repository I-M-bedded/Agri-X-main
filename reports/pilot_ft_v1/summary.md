# CCRDNet evaluation summary

- checkpoint: `../runs/pilot_ft_seed42/best.pt` (epoch 39)
- split: val, 6 frames (5 with a GT line)
- detection rate: 0.2000

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 16.087 | 16.087 | 16.087 |
| lateral near (px) | 12.28 | 12.28 | 12.28 |
| line IoU | 0.0492 | 0.0492 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.0000

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).
