# Agri-X CCRDNet-Centered Vision Plan
## Critical Review + AI Coder Execution Spec

**Status:** experimental implementation guide  
**Target:** Raspberry Pi 4B + front Camera V2  
**Current decision:** AI-Hub route is discarded. The cotton-row dataset tested so far is the preferred data source. CCRDNet is the primary architecture/task-formulation to reproduce and adapt.

---

# 1. Executive conclusion

The current best hypothesis is:

```text
cotton-row data
    +
CCRDNet-style direct navigation-target supervision
    +
auxiliary structural supervision
    +
tiny DSC U-Net + ASPP
    +
simple line fitting
    +
ToF / odometry fallback
```

But CCRDNet must **not** be copied literally.

The original CCRDNet predicts the **central crop row** and treats that row as the navigation line.

Agri-X drives **inside the furrow between raised rows/ridges**. Therefore the direct target must become the **central traversable furrow / navigation band**, not the crop-row center.

```text
Original CCRDNet
crop row center
      ↓
navigation line

Agri-X
ridge/row      traversable furrow      ridge/row
████████                              ████████
████████            │                 ████████
                    ↓
             navigation centerline
```

The main CCRDNet idea worth preserving is:

> Predict only the structure directly needed for control instead of segmenting every visible row and solving a later association problem.

---

# 2. Facts re-checked from the CCRDNet paper

Paper: Zheng & Wang, **“Extracting the central crop row with CCRDNet for universal in-row navigation in agriculture”**, Frontiers in Plant Science, 2026.

Sources:
- https://doi.org/10.3389/fpls.2026.1744637
- https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2026.1744637/full
- dataset: https://zenodo.org/records/15194034

Paper baseline:

```text
Input                256×256 RGB
Classes              background / vegetation / central crop row
Loss                 cross entropy
Optimizer            Adam
Learning rate        2e-4
Batch size           4
Training             500 epochs
Parameters           ~33.621 K
FLOPs                ~38.226 M
ASPP dilation        2, 4, 6, 8
Postprocess          largest connected component + least squares
```

Reported navigation results:

```text
Line IoU             78.37%
Average angle error   1.13°
Line accuracy        95.57%
```

Architecture ablation reported in the paper:

| Variant | DSC | ASPP | FLOPs | Params | AE | Line IoU | Line Accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| CCRDNet-1 | no | no | 100.844M | 77.186K | 1.32° | 75.73% | 91.71% |
| CCRDNet-2 | no | yes | 141.967M | 235.530K | 1.09° | 78.97% | 95.85% |
| CCRDNet-3 | yes | no | 31.791M | 12.077K | 1.43° | 74.30% | 90.06% |
| CCRDNet | yes | yes | 38.226M | 33.621K | 1.13° | 78.37% | 95.57% |

Interpretation:

- **DSC** is primarily the efficiency mechanism.
- **ASPP** provides a meaningful accuracy gain.
- Do not remove ASPP before reproducing its benefit on our data.

The paper also shows that 3-class supervision beats 2-class supervision for CCRDNet:

| Classes | Line IoU | AE | Line Accuracy |
|---|---:|---:|---:|
| background + target | 74.92% | 1.38° | 91.46% |
| background + vegetation + target | 78.37% | 1.13° | 95.57% |

This auxiliary-class result is directly relevant to Agri-X.

---

# 3. Agri-X label definition

Do not initially use ambiguous names such as `ridge` unless the cotton dataset really labels the physical ridge.

Use:

```text
0 = OTHER
1 = STRUCTURE
2 = NAV_BAND
```

Definitions:

### OTHER
Everything not useful as explicit row context or navigation target.

### STRUCTURE
The surrounding crop-row / ridge-like visual structure that helps determine where the traversable corridor lies.

Initially this is **auxiliary supervision only**.

Do not use it as a hard collision mask until it has been separately validated.

### NAV_BAND
A fixed-width band centered on the desired path through the current furrow.

This is the control-critical output.

---

# 4. First task: audit the cotton dataset before any training

The AI coder must first create:

```text
reports/cotton_dataset_audit.md
reports/cotton_dataset_audit/samples_overlay.jpg
```

The audit must explicitly state:

```text
dataset source
license
image count
image resolutions
annotation format
class meanings
camera pose/view diversity
field/lighting diversity
existing train/val/test split
whether labels represent:
    vegetation
    crop row
    physical ridge
    furrow
    centerline
    drivable area
```

The overlay montage must contain at least 30 randomly sampled annotated frames.

**Hard gate:** no model training until the coder can explain exactly how the dataset labels are converted into the desired Agri-X navigation target.

Do not infer label semantics from filenames.

---

# 5. Target generation

Preferred supervision is a **fixed-width navigation band**, not a variable-width furrow-area mask.

Reason:

The control objective is the corridor center. The visible corridor width varies with perspective, row geometry and field appearance, but the desired navigation reference remains a line-like structure.

```text
STRUCTURE               STRUCTURE
████████                ████████
████████                ████████
         | NAV_BAND |
              │
              ↓
        desired path
```

This follows the strongest CCRDNet principle: supervise invariant navigation geometry rather than transient appearance.

---

# 6. Do not blindly copy the paper's “15 px” line width

CCRDNet annotated a fixed 15-pixel line after resizing source images to 640×480, while model training used 256×256 inputs.

Therefore `15 px` is not a universal value at 256×256.

Do not hardcode:

```python
LINE_WIDTH = 15
```

for our model.

At 256 scale, test a small set such as:

```text
4 px
6 px
8 px
```

or use a normalized width ratio and transform it consistently.

Choose width based on navigation performance, not mask IoU.

---

# 7. Preprocessing experiments

Start with a paper-style baseline:

```text
640×480 RGB
→ resize 256×256
→ CCRDNet
```

This is only the reproduction baseline.

Because direct 4:3 → 1:1 resizing distorts geometry, compare later:

```text
P0: stretch 256×256
P1: letterbox 256×256
P2: preserve aspect 256×192
P3: preserve aspect 320×240
```

All predicted coordinates must map correctly back to original camera coordinates.

Do not add CLAHE, Retinex, bird's-eye transforms or complex preprocessing before establishing the baseline.

---

# 8. CCRDNet reproduction requirements

Implement modular blocks:

```python
DSCBlock
LiteASPP
CCRDNet
```

Paper-described DSC concept:

```text
3×3 depthwise/grouped conv
→ GELU
→ 1×1 pointwise conv
→ BatchNorm
→ GELU
```

Paper-described ASPP:

```text
dilation = 2,4,6,8
branches concatenated
→ 1×1 projection
```

ASPP is applied only to the final three skip connections.

Do not add:

```text
attention
SE
CBAM
transformers
extra residual towers
custom decoder tricks
```

to the first baseline.

---

# 9. Architecture uncertainty must be explicit

If exact channel counts cannot be recovered from the paper/figure:

1. keep all inferred channels in one configuration,
2. document the inference,
3. report params and FLOPs,
4. tune only enough to stay close to the paper's scale.

Target scale:

```text
~33.6 K parameters
~38.2 M FLOPs at 256×256
```

If the implementation becomes 500K parameters or 1 GFLOP, do not call it a CCRDNet reproduction.

---

# 10. Critical deployment warning: 38M FLOPs does not guarantee Pi speed

This is the most important runtime criticism.

The paper reports roughly:

```text
RTX3060 GPU       86.76 FPS
laptop CPU        40.94 FPS
Orin NX GPU       48.37 FPS
Orin NX CPU        2.58 FPS
```

The Orin NX CPU result shows that tiny theoretical complexity can still perform poorly on an ARM CPU.

Potential reasons:

```text
depthwise/grouped-conv kernel efficiency
GELU kernel efficiency
ASPP branching
memory-bound execution
small-tensor dispatch overhead
poor backend fusion
```

Therefore Pi 4 acceptance must use **measured latency**, not FLOPs.

---

# 11. Runtime/backend plan

After a model is visually validated:

```text
PyTorch reference
    ↓
ONNX
    ↓
ONNX Runtime ARM
```

Optionally test:

```text
NCNN
TFLite
```

Only then consider deployment-specific ablations:

```text
GELU → ReLU
GELU → ReLU6
GELU → HardSwish
ASPP branch simplification
BatchNorm folding
```

Any operator change must be retrained/validated.

Do not graph-edit activations and assume equal accuracy.

---

# 12. Why central-target prediction is attractive

The original paper compared detecting all rows with detecting only the navigation-critical center row.

Central-only supervision produced better navigation metrics.

That supports avoiding this pipeline:

```text
detect every row
→ cluster every row
→ associate rows
→ choose target
→ fit path
```

Preferred Agri-X philosophy:

```text
predict THE corridor currently being followed
```

This reduces association ambiguity and post-processing failure points.

---

# 13. Safety limitation of direct NAV_BAND prediction

A centerline tells the controller:

```text
where to go
```

but not necessarily:

```text
where never to go
```

Agri-X must not climb the raised row.

Therefore retain STRUCTURE as an auxiliary output when possible.

But the paper notes that non-central semantic segmentation can be imperfect while the navigation line is still correct.

So initially:

```text
NAV_BAND  = control-critical
STRUCTURE = auxiliary/context
ToF       = independent physical safety cue
```

Do not treat STRUCTURE as a certified forbidden mask without separate recall/clearance tests.

---

# 14. Baseline model definition

Working name:

```text
AgriCCRDNet-v0
```

Baseline:

```text
Input   RGB 256×256
Output  OTHER / STRUCTURE / NAV_BAND
Loss    CrossEntropyLoss
```

Use simple CE first.

Only after baseline, test:

```text
weighted CE
CE + Dice(NAV_BAND)
```

Keep a new loss only if it improves navigation metrics.

---

# 15. Primary metrics

Do not rank models by mIoU alone.

Primary metrics:

### Angle Error
```text
AE = |theta_pred - theta_gt|
```

Report:

```text
mean
median
P95
```

### Lateral Error

At configurable lookahead rows:

```text
|x_pred(y) - x_gt(y)|
```

Report in:

```text
pixels
normalized image width
millimeters if calibration supports it
```

### Target success rate

Define a project-specific criterion combining angle + lateral tolerance.

Do not blindly inherit the paper's Line-IoU threshold as the final physical acceptance criterion.

---

# 16. Additional critical metrics

Track:

```text
no target predicted
wrong adjacent corridor selected
frame-to-frame target switching
false line on headland/non-furrow scenes
high-confidence wrong prediction
```

The worst failure is:

```text
wrong line + high confidence
```

not merely an empty prediction.

---

# 17. Post-processing baseline and criticism

Paper baseline:

```text
NAV_BAND mask
→ largest connected component
→ least-squares line
```

Implement this first.

But largest-component selection can fail when:

```text
the true row is fragmented
a false blob is larger
large missing-row gaps occur
```

The paper itself reports difficulty with dense rows and long crop gaps.

Mandatory postprocess ablations:

```text
C0 largest component
C1 bottom-center connected/preferred component
C2 scored component:
       area
       center proximity
       bottom proximity
       vertical span
C3 optional temporal consistency
```

---

# 18. Line fitting experiments

Compare:

```text
F0 ordinary least squares
F1 PCA / Total Least Squares
F2 RANSAC
```

PCA/TLS is attractive because it does not require privileging x or y as the independent variable.

But reproduce the paper's LS baseline first.

---

# 19. Control geometry output

Convert fitted geometry into the existing model-neutral interface:

```python
FurrowEstimate(
    normalized_error,
    heading_error,
    confidence,
    coverage,
)
```

Recommended geometry:

```text
near lookahead: y ≈ 0.80H
far lookahead:  y ≈ 0.45H
```

Use near-point displacement for lateral error and near/far relation for heading.

Keep fractions configurable.

---

# 20. Confidence definition

Do not use only max softmax probability.

Preferred confidence combines:

```text
NAV_BAND probability
component vertical span
fit residual
usable coverage
optional temporal consistency
```

For example:

```text
confidence =
model_score
× span_score
× fit_score
```

Keep it interpretable.

---

# 21. Temporal extension

Do not put temporal modeling inside the CNN initially.

After single-frame performance is known, use minimal state estimation:

```text
current fitted line
+
previous fitted line
→ EMA
```

or:

```text
small Kalman filter
```

The CCRDNet paper itself suggests temporal continuity for long missing-row gaps.

A recurrent/video network is not justified at this stage.

---

# 22. FSM interaction

CCRDNet should mainly be trusted in:

```text
ACQUIRE_FURROW
FOLLOW_OUTBOUND
FOLLOW_RETURN
```

Do not expect it to solve:

```text
TURN_INTO_FURROW
TURN_AT_END
EXIT_FURROW
```

Those remain handled by FSM / ArUco / odometry / ToF.

This keeps the vision problem constrained.

---

# 23. Data splitting

Do not random-split adjacent video frames into train and validation.

Split by coherent domain units where possible:

```text
field
row
recording session
lighting condition
camera setup
date
```

At least one validation subset must represent a genuinely unseen condition.

---

# 24. Required condition tags

Evaluation subsets should include:

```text
clean
shadow
overexposed
dark
dry_soil
wet_soil
weed
sparse_structure
missing_structure
dense_structure
off_center
large_heading_error
entrance
near_end
```

Report condition-wise metrics, not only aggregate results.

---

# 25. Augmentation

Recommended initial augmentations:

```text
brightness/contrast
gamma
mild color shift
shadow
blur
sensor noise
rotation
lateral shift/crop
small perspective perturbation
```

All geometric transforms must transform NAV_BAND identically.

If horizontal flip is used, line geometry and steering sign semantics must also be flipped correctly.

Add visual tests for augmentation correctness.

---

# 26. Real Camera V2 evaluation is mandatory

Public cotton data is only the starting point.

Create a fixed real-data set:

```text
data/agrix_eval/
```

Initial target:

```text
50–150 diverse Camera V2 frames
```

Include the real fixed camera height, pitch, FOV and vibration conditions.

The true question is not:

```text
Does CCRDNet work on cotton?
```

It is:

```text
Does the learned geometric prior transfer to the Agri-X camera domain?
```

---

# 27. Core experiment matrix

## A. Label/task ablation

| ID | Labels |
|---|---|
| A0 | OTHER + NAV_BAND |
| A1 | OTHER + STRUCTURE + NAV_BAND |
| A2 | all-row style control experiment |

Expected hypothesis:

```text
A1 should be more robust than A0 and A2.
```

If not, inspect label construction before changing architecture.

## B. Architecture ablation

| ID | DSC | ASPP |
|---|---:|---:|
| B0 | no | no |
| B1 | yes | no |
| B2 | no | yes |
| B3 | yes | yes |

This should reproduce the paper's qualitative claim:

```text
DSC → efficiency
ASPP → navigation accuracy
```

## C. Postprocessing

```text
C0 largest CC + LS
C1 bottom-center component + LS
C2 bottom-center component + PCA/TLS
C3 temporal component scoring + PCA/TLS
```

## D. Resolution/runtime

```text
256×256 stretch
256×256 letterbox
256×192
320×240
```

---

# 28. Pi 4 benchmark acceptance

Initial engineering target:

```text
>= 5 Hz preferred
>= 2 Hz minimum for continued consideration
```

But bounded latency matters more than average FPS.

Required runtime behavior:

```text
latest-frame-only worker
no backlog
no frame queue
bounded perception age
20 Hz control loop remains independent
```

Benchmark with:

```text
camera active
control loop active
inference active
```

Report:

```text
mean latency
P50
P95
FPS
CPU
RAM
temperature
perception age
control-loop jitter
```

---

# 29. Proposed code organization

Experimental module:

```text
farm_robot/
├── perception/
│   └── ccrdnet/
│       ├── blocks.py
│       ├── model.py
│       ├── postprocess.py
│       ├── runtime.py
│       └── config.py
├── tools/
│   ├── build_navigation_targets.py
│   ├── train_ccrdnet.py
│   ├── eval_ccrdnet.py
│   ├── export_ccrdnet_onnx.py
│   └── benchmark_ccrdnet.py
└── tests/
    ├── test_ccrdnet_shapes.py
    ├── test_ccrdnet_geometry.py
    └── test_ccrdnet_postprocess.py
```

Do not modify the mission FSM during model experiments.

The model must adapt to the existing perception contract.

---

# 30. Reproducibility requirements

Every training run must save:

```text
experiment ID
config
random seed
git commit
dataset version
train/val manifests
parameter count
FLOP estimate
metrics.csv
best checkpoint
last checkpoint
```

Suggested run name:

```text
ccrd_a1_b3_256_ce_seed42
```

Every evaluation run should create:

```text
reports/<experiment>/
├── metrics.json
├── metrics_by_condition.csv
├── overlays/
├── failure_cases/
└── summary.md
```

`summary.md` must end with:

```text
KEEP
REJECT
or
INCONCLUSIVE
```

and explain why.

---

# 31. Required visual overlays

For each evaluation image show:

```text
ground-truth NAV_BAND
predicted NAV_BAND
ground-truth fitted line
predicted fitted line
near lookahead point
far lookahead point
confidence
angle error
lateral error
```

This is mandatory for reviewing failures.

---

# 32. Unit tests

Implement tests for:

### Model shape
```text
B×3×H×W → B×C×H×W
```

### Coordinate transforms
After stretch/crop/letterbox, mapped coordinates must match synthetic known points in 640×480 coordinates.

### Empty mask
Must return a valid no-guidance result with confidence 0.

### Synthetic line masks
Test:

```text
centered vertical
shifted vertical
+15°
-15°
broken line
two competing lines
large false blob
short near-field line
curved line
```

### Determinism
Given the same mask/config, post-processing selects the same component and line.

---

# 33. Failure classification

Tag every failure as one of:

```text
MODEL_MISS
WRONG_CORRIDOR
FRAGMENTATION
WRONG_COMPONENT
FIT_FAILURE
DOMAIN_SHIFT
MOTION_BLUR
SHADOW
OVEREXPOSURE
MISSING_STRUCTURE
END_OF_ROW
```

Count categories.

Do not change architecture based on anecdotal examples. Address the dominant failure class.

---

# 34. Stop criteria for CCRDNet

Substantially redesign or drop the approach if these persist:

```text
repeated adjacent-corridor switching
high-confidence wrong line on Camera V2
systematic failure on missing structure
Pi 4 runtime remains impractical after backend optimization
3-class structural supervision gives no benefit
direct NAV_BAND prediction is clearly worse than boundary-derived centerline
```

Do not optimize indefinitely because the paper reports good results.

---

# 35. Fallback if direct NAV_BAND prediction fails

Do not jump to a huge foundation model.

Fallback:

```text
left/right structural boundary segmentation
          ↓
midpoint geometry
          ↓
navigation line
```

This keeps the pipeline lightweight and physically interpretable.

Do not return to generic YOLOE prompts such as:

```text
untraversable ground
farm
soil
```

for core navigation.

---

# 36. AI coder Task 1 — dataset audit

Give the coder this exact first task:

> Inspect the selected cotton dataset before training anything. Produce `reports/cotton_dataset_audit.md` and a montage of at least 30 random labeled samples. Determine exactly what every annotation class represents and whether it corresponds to crop vegetation, crop row, physical ridge, furrow, centerline, or drivable area. Document dataset license, image count, resolution, annotation format, camera/view diversity and train/validation leakage risks. Then propose a deterministic conversion to `OTHER`, `STRUCTURE`, and a fixed-width `NAV_BAND` centered on the desired traversable furrow. Do not train a model. Flag any semantic ambiguity that makes this conversion unreliable.

**Acceptance gate:** labels and target generation are semantically defensible.

---

# 37. AI coder Task 2 — target builder

> Implement `build_navigation_targets.py`. Convert the reviewed cotton annotations into deterministic 3-class masks: `OTHER`, `STRUCTURE`, `NAV_BAND`. Make navigation-band width configurable. Preserve geometry through resize/crop/letterbox transforms. Generate train/validation manifests and an overlay-review mode. Add unit tests for coordinate transforms and target width.

**Acceptance gate:** inspect at least 100 converted samples with no systematic label error.

---

# 38. AI coder Task 3 — CCRDNet reproduction

> Implement an isolated CCRDNet-style baseline using a U-shaped encoder-decoder, depthwise separable convolution blocks, and lightweight ASPP with dilation rates 2/4/6/8 on the last three skip connections. Put inferred channel counts in a configuration file. Report parameter count and FLOPs and compare with the paper's ~33.6K parameters / ~38.2M FLOPs at 256×256. Do not add attention, transformers, custom losses or temporal modules. If exact architectural details are unavailable, explicitly call the model a reproduction rather than an exact implementation.

**Acceptance gate:** shape tests pass; scale is close to CCRDNet; architecture assumptions documented.

---

# 39. AI coder Task 4 — paper-style baseline training

> Train the 3-class baseline with 256×256 RGB input, cross-entropy loss and Adam at 2e-4. Use validation curves rather than blindly forcing 500 epochs. Save reproducible configs, split manifests, metrics and checkpoints. Implement paper-style largest-connected-component plus least-squares navigation-line extraction. Report angle error, lateral error, Line IoU and condition-wise failure rates. Generate overlays for failures. Do not select the model by mIoU alone.

**Acceptance gate:** model clearly learns the correct corridor on held-out cotton data.

---

# 40. AI coder Task 5 — mandatory ablations

Run:

```text
2-class vs 3-class
DSC off/on
ASPP off/on
largest CC vs bottom-center selection
LS vs PCA/TLS
```

Each experiment gets its own report.

**Acceptance gate:** determine whether the key CCRDNet claims actually hold on this domain.

---

# 41. AI coder Task 6 — real Agri-X evaluation

> Run the strongest variants on the fixed Camera V2 evaluation set. Review wrong-corridor selection, target switching, missing-structure behavior and confidence calibration. Do not integrate into the mission runtime yet.

**Acceptance gate:** no systematic dangerous failure under the real camera domain.

---

# 42. AI coder Task 7 — export and Pi 4 benchmark

> Export the accepted model to ONNX and verify numerical agreement with PyTorch before benchmarking. Test ONNX Runtime ARM first, then NCNN/TFLite if useful. Benchmark while camera and the 20 Hz control loop are active. Record mean/P95 latency, FPS, CPU, RAM, temperature, perception age and control-loop jitter. Do not infer speed from FLOPs.

**Acceptance gate:** practical sustained latency with no stale-frame queue.

---

# 43. AI coder Task 8 — runtime adapter

Only after all prior gates:

> Implement the model backend behind the existing model-neutral perception interface and convert its fitted geometry into `FurrowEstimate(normalized_error, heading_error, confidence, coverage)`. Do not introduce CCRDNet-specific tensors or classes into the mission FSM.

---

# 44. Final review

The strongest idea in CCRDNet is **not** “use depthwise convolution”.

It is:

```text
task-aligned target supervision
+
structural auxiliary supervision
+
very small segmentation network
+
minimal geometric extraction
```

For Agri-X, the correct adaptation is:

```text
central crop row
        ↓ change semantics
central traversable furrow / NAV_BAND
```

The two biggest risks are:

1. **semantic mismatch** — training the crop-row center when the robot must drive between rows;
2. **deployment assumption** — treating 38M FLOPs as proof of fast Raspberry Pi inference.

The required order of work is therefore:

```text
dataset semantics
→ target construction
→ faithful baseline
→ navigation metrics
→ architecture/postprocess ablations
→ real Camera V2 evaluation
→ ARM benchmark
→ runtime integration
```

Do not reverse that order.

