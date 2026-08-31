# -*- coding: utf-8 -*-
"""Convert the cotton_pilot YOLO-seg labels into Agri-X 3-class masks.

Source: data/cotton_pilot/trajectory{01,02,03}/{images,labels}/train, 640x480
RealSense frames taken while driving *inside* the furrow.  Only two label
classes are actually used:

    0 hilera  crop-row / ridge strip polygons  -> 1 STRUCTURE
    2 maleza  weeds                            -> 0 OTHER (not row structure)

There is no furrow or central-row class, so NAV_BAND is derived per frame as
the Agri-X navigation target (spec section 3): walking rows bottom-up, the
furrow centre at row y is the midpoint between the right edge of the nearest
hilera region left of the tracked centre and the left edge of the nearest
region right of it.  The centre track starts at the image centre column at the
bottom.  A fixed-width band (default 17 px, the cotton dataset convention at
640x480) is drawn along the midpoints; rows with no structure on both sides
(sky, headland) get no band.

    python tools/build_pilot_targets.py --source ../data/cotton_pilot \
        --out ../data/cotton_pilot/prepared
"""

import argparse
import glob
import os
import shutil
import sys

import cv2
import numpy as np

_FARM_ROBOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from perception.ccrdnet.structure_midline import derive_nav_band  # noqa: E402

HILERA, MALEZA = 0, 2


def rasterize(label_path: str, shape):
    h, w = shape
    structure = np.zeros((h, w), np.uint8)
    for line in open(label_path, "r", encoding="utf-8"):
        parts = line.split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        if cls != HILERA:
            continue
        pts = np.array([float(v) for v in parts[1:]], np.float64).reshape(-1, 2)
        pts = (pts * [w, h]).astype(np.int32)
        cv2.fillPoly(structure, [pts], 1)
    return structure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--band-width", type=int, default=17)
    parser.add_argument("--val-per-traj", type=int, default=3,
                        help="frames from the end of each train trajectory used as val")
    args = parser.parse_args()

    for sub in ("images", "masks", "manifests"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    per_traj = {}
    for img_path in sorted(glob.glob(
            os.path.join(args.source, "trajectory*", "images", "train", "*.jpg"))):
        traj = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path))))
        label_path = img_path.replace(
            os.sep + "images" + os.sep, os.sep + "labels" + os.sep
        ).replace(".jpg", ".txt")
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        structure = rasterize(label_path, (h, w))
        nav = derive_nav_band(structure, args.band_width)
        mask = np.zeros((h, w), np.uint8)
        mask[structure > 0] = 1
        mask[nav > 0] = 2          # band wins over structure where they touch
        name = f"{traj}_{os.path.splitext(os.path.basename(img_path))[0]}"
        cv2.imwrite(os.path.join(args.out, "masks", name + ".png"), mask)
        shutil.copyfile(img_path, os.path.join(args.out, "images", name + ".jpg"))
        per_traj.setdefault(traj, []).append((name, int(nav.sum())))

    # manifests: all 47 as zero-shot eval; traj02 held out for fine-tune test
    manifests = {"test_all": [], "train": [], "val": [], "test": []}
    for traj, items in sorted(per_traj.items()):
        names = [n for n, _ in items]
        manifests["test_all"] += names
        if traj == "trajectory02":
            manifests["test"] += names
        else:
            manifests["train"] += names[: -args.val_per_traj]
            manifests["val"] += names[-args.val_per_traj:]
        empty = [n for n, s in items if s == 0]
        print(f"{traj}: {len(names)} frames, no-band frames: {len(empty)} {empty[:3]}")

    for split, names in manifests.items():
        with open(os.path.join(args.out, "manifests", split + ".txt"), "w",
                  encoding="utf-8") as f:
            f.write("\n".join(names) + "\n")
        print(f"{split}: {len(names)}")


if __name__ == "__main__":
    main()
