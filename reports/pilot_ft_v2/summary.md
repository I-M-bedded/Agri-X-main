# CCRDNet evaluation summary

- checkpoint: `../runs/pilot_ft_v2/best.pt` (epoch 310)
- split: test, 13 frames (13 with a GT line)
- detection rate: 1.0000

| metric | mean | median | P95 |
|---|---:|---:|---:|
| angle error (deg) | 10.718 | 4.921 | 32.845 |
| lateral near (px) | 13.06 | 2.89 | 40.57 |
| line IoU | 0.1948 | 0.1547 | - |

line accuracy (AE<=5.0deg AND lateral_near<=10.0px): 0.3846

Overlays in `overlays/`, worst cases in `failure_cases/`.
Colours: green = GT band/line, red = predicted band/line, yellow = overlap, orange = STRUCTURE (GT).
