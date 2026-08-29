# Lightweight Agri-X mission pipeline

This branch keeps the runtime intentionally non-ROS. A Raspberry Pi 4 owns one
camera, two side ToF sensors, encoders, two motors and an optional pump; a single
20 Hz FSM is easier to profile and recover than a multi-node runtime.

## Field marker convention

The new pipeline uses a deliberately simple installation contract.

- `0`: HOME marker. It must be visible while the robot returns along the headland.
- `1..N`: furrow entrance markers in visiting order.
- `249`: dedicated END marker placed on the headland after the final furrow.

An entrance marker is a **turn waypoint**, not a full pose reference. The robot
centres the marker loosely, stops at a fixed stand-off distance, makes a coarse
+/-90 degree encoder turn, and then hands residual lateral/heading error to the
furrow follower.

This differs from the legacy `mission_state_machine.py` END-marker convention;
`pipeline_main.py` and `agri_pipeline_fsm.py` follow the convention above.

## Runtime graph

```text
RPi Camera V2 (640x480 BGR)
        |
        +---- ArUco, ~10 Hz only in marker states
        |
        +---- latest-frame AI worker, 2 Hz initial target on Pi 4
                  |
                  +---- furrow / farm-furrow / soil-trench masks
                  |        -> bottom-connected component
                  |        -> left/right mask boundaries
                  |        -> geometric centre line
                  |        -> lateral + heading error
                  |
                  +---- obstacle / non-traversable prompts
                           -> near-field trapezoid intersection
                           -> global SAFE_HALT

Side VL53L1X pair ---------------------- 20 Hz
        |                                  |
        +---- near-wall emergency ---------+
        +---- furrow centring assist ------+--> motor command, 20 Hz
        +---- both-walls-lost = furrow end |

Encoders --------------------------------+
        +---- 90/180 degree turns
        +---- headland heading hold
        +---- minimum travelled distance
```

The AI worker never queues frames. If inference is slower than the requested
rate, old frames are dropped and only the newest image is processed. The motor
and ToF loop therefore keeps running at 20 Hz. The control loop checks the latest
AI safety snapshot on every tick; "always on" means the watchdog is continuously
scheduled and enforced, not that a new CNN result is produced at 20 Hz.

## FSM

```text
INIT
  -> SEARCH_MARKER
  -> APPROACH_MARKER
  -> TURN_INTO_FURROW          (coarse +/-90 deg)
  -> ACQUIRE_FURROW            (AI centre line and/or side ToF)
  -> FOLLOW_OUTBOUND
  -> TURN_AT_END               (180 deg)
  -> FOLLOW_RETURN
  -> EXIT_FURROW
  -> TURN_TO_HEADLAND          (same +/-90 deg sign as entry)
  -> SEARCH_NEXT_MARKER
       -> APPROACH_MARKER ...  (next numbered marker)
       -> RETURN_HOME_TURN      (END marker 249)
  -> RETURN_HOME               (encoder heading hold + marker scan)
  -> HOME_APPROACH             (marker 0)
  -> MISSION_COMPLETE

Any stale camera/AI, drive-corridor obstacle, extreme side-ToF clearance,
missing expected marker, lost guidance, or timeout -> SAFE_HALT.
```

## Zero-shot model

The current reference model is **YOLOE-26n-seg**. Text prompts are baked into
an exported model once, so the Pi does not run/download the text encoder.
Prompts are kept small because open-vocabulary class matching also costs CPU.

Furrow prompts:

```text
furrow, farm furrow, soil trench
```

Safety prompts:

```text
person, animal, vehicle, tractor, rock, log, box,
farm equipment, hole, water puddle, obstacle,
untraversable ground
```

Important: open-vocabulary segmentation is a first-stage watchdog, not a proof
of general traversability. Object-like prompts are expected to be more reliable
than broad terrain concepts such as `untraversable ground`. An unknown obstacle
that does not match these concepts can be missed. After field videos are
collected, a small local `traversable / non-traversable / furrow` model should
replace or distil this watchdog if arbitrary terrain safety becomes a requirement.

Inside a furrow, the selected furrow mask itself also defines the desired drive
corridor: the controller stays near its geometric centre and side ToF prevents
contact with the ridges. The zero-shot safety branch is an additional front-area
watchdog rather than the sole navigation signal.

## Raspberry Pi 4 deployment

Do the text-prompt setup and export on a development machine:

```bash
cd farm_robot
python3 tools/export_yoloe_rpi.py --imgsz 320
```

Copy/keep the resulting directory at:

```text
farm_robot/models/agri_yoloe26n_ncnn_model
```

Then on the Pi:

```bash
python3 main.py \
  --model models/agri_yoloe26n_ncnn_model \
  --imgsz 320 \
  --inference-hz 2
```

Start at `320x320`, 2 Hz AI, `0.28` maximum furrow speed. These are conservative
**initial targets, not measured Pi 4 performance claims**. Read
`snapshot.inference_sec`/logs on the actual Pi before increasing inference rate
or speed. ArUco and the 20 Hz controller remain separate from the AI worker.

The previous mission implementation is preserved as:

```bash
python3 main.py --legacy
```

## Cheap verification before ROS

A deterministic dependency-injected smoke test checks the intended one-furrow
mission sequence without GPIO or physics:

```bash
python3 tools/pipeline_smoke_test.py
```

It is only a state-machine regression test. It does not validate vehicle dynamics,
camera geometry, ToF physics, or zero-shot perception accuracy.

## Why ROS is not in the runtime yet

ROS2/Gazebo is useful here, but mainly for **geometry and state-machine
verification**, not for proving RGB perception accuracy.

Good Gazebo targets:

- skid-steer / track turn error and 90/180 degree manoeuvres;
- marker placement/FOV and headland approach geometry;
- side ToF acquisition and loss at furrow entrances/exits;
- FSM transition/timeout/recovery cases;
- HOME return and missed-marker cases.

Poor Gazebo target:

- deciding whether zero-shot segmentation will work on real soil, shadows,
  water, dust and exposure. Use recorded real RPi Camera V2 video for that.

The new FSM uses dependency injection (`camera`, `tof`, `odom`, `motors`,
`perception`), so ROS adapters can later implement those interfaces without
rewriting mission logic. Keep ROS as a simulation/adapter layer unless the
sensor stack grows enough to justify a distributed runtime.
