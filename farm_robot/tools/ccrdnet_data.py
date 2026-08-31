# -*- coding: utf-8 -*-
"""Dataset plumbing shared by train/eval scripts.

Expects a prepared dataset root:

    <root>/
      images/<name>.jpg|png        RGB frames
      masks/<name>.png             uint8 class ids {0 OTHER, 1 STRUCTURE, 2 NAV_BAND}
      manifests/train.txt          one <name> per line
      manifests/val.txt
      manifests/test.txt

Masks are stored at source resolution; resizing to model input happens here
(stretch mode baseline, nearest for masks) via perception.ccrdnet.preprocess.
"""

import os
import random
import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

_FARM_ROBOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _FARM_ROBOT not in sys.path:
    sys.path.insert(0, _FARM_ROBOT)

from perception.ccrdnet.preprocess import preprocess_image, preprocess_mask  # noqa: E402


def read_manifest(root: str, split: str) -> List[str]:
    path = os.path.join(root, "manifests", f"{split}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_image(root: str, name: str) -> str:
    for ext in (".jpg", ".png", ".jpeg", ".JPG", ".PNG"):
        p = os.path.join(root, "images", name + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"no image for {name} under {root}/images")


class CCRDDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        input_size: Tuple[int, int] = (256, 256),  # (H, W)
        resize_mode: str = "stretch",
        augment: bool = False,
        geometric_augment: bool = False,
        seed: int = 42,
    ):
        self.root = root
        self.names = read_manifest(root, split)
        self.input_size = input_size
        self.resize_mode = resize_mode
        self.augment = augment
        self.geometric_augment = geometric_augment
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.names)

    def load_pair(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(find_image(self.root, name), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"failed to read image for {name}")
        mask = cv2.imread(
            os.path.join(self.root, "masks", name + ".png"), cv2.IMREAD_GRAYSCALE
        )
        if mask is None:
            raise IOError(f"failed to read mask for {name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, mask

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        r = self.rng
        if r.random() < 0.5:  # horizontal flip: geometry flips identically
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
        if self.geometric_augment and r.random() < 0.7:
            # small rotation + shift, identical for image and mask (spec section 25)
            h, w = mask.shape
            angle = r.uniform(-7.0, 7.0)
            tx = r.uniform(-0.10, 0.10) * w
            ty = r.uniform(-0.06, 0.06) * h
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            matrix[0, 2] += tx
            matrix[1, 2] += ty
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT_101)
            mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if r.random() < 0.8:  # brightness / contrast
            alpha = 1.0 + r.uniform(-0.25, 0.25)
            beta = r.uniform(-25, 25)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        if r.random() < 0.3:  # gamma
            gamma = r.uniform(0.7, 1.4)
            lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            image = lut[image]
        if r.random() < 0.2:  # mild blur
            image = cv2.GaussianBlur(image, (3, 3), 0)
        return image, mask

    def __getitem__(self, index: int):
        name = self.names[index]
        image, mask = self.load_pair(name)
        if self.augment:
            image, mask = self._augment(image, mask)
        th, tw = self.input_size
        image_r, _ = preprocess_image(image, (tw, th), self.resize_mode)
        mask_r, _ = preprocess_mask(mask, (tw, th), self.resize_mode)
        tensor = torch.from_numpy(image_r.astype(np.float32) / 255.0).permute(2, 0, 1)
        target = torch.from_numpy(mask_r.astype(np.int64))
        return tensor, target, name


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
