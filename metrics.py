"""
metrics.py — Segmentation metrics for wound detection
------------------------------------------------------
All metrics work on binary masks (post-threshold sigmoid).

Metrics:
  • Dice / F1 Score
  • IoU  (Jaccard Index)
  • Precision
  • Recall  (Sensitivity)
  • Specificity
  • Hausdorff Distance  (boundary quality)
"""

import torch
import numpy as np
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
#  Core pixel-level metrics
# ─────────────────────────────────────────────────────────────────────────────

class SegmentationMetrics:
    """
    Accumulates TP / FP / FN / TN over batches, computes metrics at the end.

    Usage:
        metrics = SegmentationMetrics(threshold=0.5)
        for batch in loader:
            probs = torch.sigmoid(model(images))
            metrics.update(probs, masks)
        results = metrics.compute()
        metrics.reset()
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6):
        self.threshold = threshold
        self.smooth    = smooth
        self.reset()

    def reset(self) -> None:
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0
        self.n_samples = 0

    @torch.no_grad()
    def update(
        self,
        probs:   torch.Tensor,    # (B, 1, H, W) float  [0, 1]
        targets: torch.Tensor,    # (B, 1, H, W) float  {0, 1}
    ) -> None:
        preds = (probs >= self.threshold).float()
        tgt   = (targets >= 0.5).float()         # binarise smoothed labels

        self.tp += (preds * tgt).sum().item()
        self.fp += (preds * (1 - tgt)).sum().item()
        self.fn += ((1 - preds) * tgt).sum().item()
        self.tn += ((1 - preds) * (1 - tgt)).sum().item()
        self.n_samples += probs.size(0)

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        s = self.smooth

        dice        = (2 * tp + s) / (2 * tp + fp + fn + s)
        iou         = (tp + s)     / (tp + fp + fn + s)
        precision   = (tp + s)     / (tp + fp + s)
        recall      = (tp + s)     / (tp + fn + s)
        specificity = (tn + s)     / (tn + fp + s)
        f1          = dice

        return {
            "dice":        round(dice, 4),
            "iou":         round(iou,  4),
            "precision":   round(precision, 4),
            "recall":      round(recall, 4),
            "specificity": round(specificity, 4),
            "f1":          round(f1, 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Per-sample Dice  (for logging individual hard samples)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def dice_per_sample(
    probs:     torch.Tensor,
    targets:   torch.Tensor,
    threshold: float = 0.5,
    smooth:    float = 1e-6,
) -> torch.Tensor:
    """Returns per-image Dice scores as a 1-D tensor."""
    preds = (probs >= threshold).float()
    tgt   = (targets >= 0.5).float()

    b = preds.size(0)
    preds = preds.view(b, -1)
    tgt   = tgt.view(b, -1)

    intersection = (preds * tgt).sum(dim=1)
    union        = preds.sum(dim=1) + tgt.sum(dim=1)
    return (2 * intersection + smooth) / (union + smooth)


# ─────────────────────────────────────────────────────────────────────────────
#  Hausdorff Distance  (optional, numpy-based, call on CPU tensors)
# ─────────────────────────────────────────────────────────────────────────────

def hausdorff_distance(
    pred_mask:   np.ndarray,
    target_mask: np.ndarray,
    percentile:  float = 95.0,
) -> float:
    """
    95th-percentile Hausdorff Distance between boundary pixels.
    Lower is better. Returns 0.0 if either mask is empty.

    Args:
        pred_mask   : H×W binary numpy array (uint8 or bool)
        target_mask : H×W binary numpy array
        percentile  : 95.0 is standard in medical image segmentation
    """
    try:
        from scipy.ndimage import distance_transform_edt
    except ImportError:
        return float("nan")

    p = pred_mask.astype(bool)
    g = target_mask.astype(bool)

    if not p.any() or not g.any():
        return 0.0

    # Distance from every pred boundary pixel to nearest GT pixel
    dt_g = distance_transform_edt(~g)
    dt_p = distance_transform_edt(~p)

    hd_p_to_g = dt_g[p]
    hd_g_to_p = dt_p[g]

    return float(max(
        np.percentile(hd_p_to_g, percentile),
        np.percentile(hd_g_to_p, percentile),
    ))
