"""
dataset.py — WoundDataset
--------------------------
Supports two label modes:
  1. YOLO polygon .txt files  → converted to binary masks on-the-fly
  2. Pre-made binary mask PNGs (masks/ folder) matched by filename stem

Priority: If a mask PNG exists in masks/ dir for an image, it is used.
Otherwise, the paired YOLO .txt label is converted to a mask.

Label format  (YOLO segmentation):
  class_id  x1 y1 x2 y2 ... xN yN     (all coords normalised [0, 1])
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: YOLO polygon → binary mask
# ─────────────────────────────────────────────────────────────────────────────

def yolo_polygon_to_mask(
    label_path: str,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Read a YOLO segmentation label and rasterise polygons into a binary uint8 mask.

    Returns:
        mask: np.ndarray of shape (H, W), values in {0, 1}
    """
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) < 7:          # class_id + at least 3 xy pairs = 7
            continue

        coords = list(map(float, parts[1:]))

        if len(coords) % 2 != 0:
            coords = coords[:-1]    # drop trailing odd value if any

        points = []
        for i in range(0, len(coords), 2):
            px = int(round(coords[i]     * img_w))
            py = int(round(coords[i + 1] * img_h))
            px = np.clip(px, 0, img_w - 1)
            py = np.clip(py, 0, img_h - 1)
            points.append([px, py])

        if len(points) >= 3:
            polygon = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [polygon], 1)

    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Augmentation pipelines
# ─────────────────────────────────────────────────────────────────────────────

def get_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),

        # Spatial
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=30,
            border_mode=cv2.BORDER_CONSTANT, p=0.6
        ),
        A.ElasticTransform(
            alpha=60, sigma=12, alpha_affine=12,
            border_mode=cv2.BORDER_CONSTANT, p=0.3
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),

        # Colour / texture
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20),
        ], p=0.7),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=3.0, p=0.3),         # Helps with wound texture

        # Dropout
        A.CoarseDropout(
            max_holes=8, max_height=32, max_width=32,
            min_holes=1, fill_value=0, p=0.2
        ),

        # Normalise & tensor
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class WoundDataset(Dataset):
    """
    Dataset for wound binary segmentation.

    Args:
        images_dir  : path to images (train or val)
        labels_dir  : path to YOLO .txt labels (train or val)
        masks_dir   : path to optional pre-made binary PNG masks
        transform   : albumentations Compose pipeline
        image_size  : target spatial size (used when constructing mask from label)
        label_smoothing : float in [0, 0.1]  — gently smooth ground-truth masks
    """

    def __init__(
        self,
        images_dir: str,
        labels_dir: str,
        masks_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: int = 512,
        label_smoothing: float = 0.0,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.masks_dir  = Path(masks_dir) if masks_dir else None
        self.transform  = transform
        self.image_size = image_size
        self.label_smoothing = label_smoothing

        # Collect image paths
        self.image_paths: List[Path] = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ])

        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {images_dir}. "
                f"Supported extensions: {SUPPORTED_IMAGE_EXTS}"
            )

        print(f"[Dataset] Found {len(self.image_paths)} images in {images_dir}")

    # ──────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.image_paths)

    # ──────────────────────────────────────────
    def _load_mask(self, image_path: Path) -> np.ndarray:
        """
        Load mask with priority:
          1. masks_dir / <stem>.png
          2. labels_dir / <stem>.txt  → polygon rasterisation
        """
        stem = image_path.stem

        # Priority 1: pre-made mask PNG
        if self.masks_dir is not None:
            for ext in (".png", ".PNG", ".jpg", ".jpeg"):
                mask_file = self.masks_dir / f"{stem}{ext}"
                if mask_file.exists():
                    mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = (mask > 127).astype(np.uint8)
                        return mask

        # Priority 2: YOLO polygon label
        label_file = self.labels_dir / f"{stem}.txt"
        img = cv2.imread(str(image_path))
        if img is None:
            raise IOError(f"Cannot read image: {image_path}")
        h, w = img.shape[:2]
        mask = yolo_polygon_to_mask(str(label_file), h, w)
        return mask

    # ──────────────────────────────────────────
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_paths[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(str(image_path))
        if image is None:
            raise IOError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load / generate mask
        mask = self._load_mask(image_path)         # uint8  H×W  {0,1}

        # Albumentations transform
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]             # (C, H, W) float32 tensor
            mask  = augmented["mask"]              # (H, W) tensor
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask  = torch.from_numpy(mask)

        # Ensure float32 mask in [0, 1] with channel dim  → (1, H, W)
        mask = mask.float()
        if self.label_smoothing > 0:
            mask = mask * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        mask = mask.unsqueeze(0)                   # (1, H, W)

        return image, mask


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check (run this file directly)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ds = WoundDataset(
        images_dir="images/train",
        labels_dir="labels/train",
        masks_dir="masks",
        transform=get_train_transforms(512),
        image_size=512,
        label_smoothing=0.05,
    )

    img, msk = ds[0]
    print(f"Image shape : {img.shape}  dtype: {img.dtype}")
    print(f"Mask  shape : {msk.shape}  dtype: {msk.dtype}")
    print(f"Mask  min/max: {msk.min():.3f} / {msk.max():.3f}")
    print("Dataset OK ✓")
