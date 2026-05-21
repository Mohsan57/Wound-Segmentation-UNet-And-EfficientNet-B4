"""
loss.py — Hybrid Loss = weighted Dice Loss + Focal Loss
---------------------------------------------------------
Both losses are designed for:
  - Binary segmentation (single-channel sigmoid output)
  - Class imbalance  (wound region << background)
  - Hard-example mining (focal term)

Usage:
    criterion = HybridLoss(dice_weight=0.5, focal_weight=0.5)
    loss = criterion(logits, targets)          # logits = raw output (pre-sigmoid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Dice Loss
# ─────────────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Works on sigmoid probabilities (NOT logits).
    Smooth term prevents division by zero on empty masks.

    dice_loss = 1 - (2 · |P ∩ G| + ε) / (|P| + |G| + ε)
    """

    def __init__(self, smooth: float = 1e-6, reduction: str = "mean"):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs   : (B, 1, H, W)  sigmoid probabilities  [0, 1]
            targets : (B, 1, H, W)  ground-truth masks     {0, 1} or smoothed
        """
        assert probs.shape == targets.shape, (
            f"Shape mismatch: probs {probs.shape} vs targets {targets.shape}"
        )

        # Flatten spatial dims → (B, N)
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        union        = probs.sum(dim=1) + targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss  = 1.0 - dice_score

        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        return dice_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Focal Loss
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for binary segmentation (Lin et al., 2017).

    FL(p_t) = -alpha_t · (1 - p_t)^gamma · log(p_t)

    Accepts raw logits and applies sigmoid internally.

    Args:
        gamma : focusing parameter — higher = more focus on hard examples
        alpha : foreground class weight (0-1);  background weight = 1 - alpha
        reduction : "mean" | "sum" | "none"
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, 1, H, W)  raw model output  (pre-sigmoid)
            targets : (B, 1, H, W)  ground-truth masks {0, 1} or smoothed
        """
        # Numerically stable BCE
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # p_t = probability of correct class
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)

        # Alpha weighting: alpha for positives, (1-alpha) for negatives
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal modulating factor
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        focal_loss = focal_weight * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ─────────────────────────────────────────────────────────────────────────────
#  Hybrid Loss  (Dice + Focal)
# ─────────────────────────────────────────────────────────────────────────────

class HybridLoss(nn.Module):
    """
    Hybrid Loss = w_dice · DiceLoss  +  w_focal · FocalLoss

    Why this combination?
    ─────────────────────
    • DiceLoss   → directly optimises overlap metric (Dice / F1),
                   robust to class imbalance, smooth gradients on masks.
    • FocalLoss  → down-weights easy background pixels, forces the model
                   to focus on hard wound boundary pixels.

    Together they handle:
      ✓ Large background / small wound imbalance
      ✓ Smooth well-segmented regions  (Dice)
      ✓ Sharp boundary learning        (Focal)

    Args:
        dice_weight  : weight for Dice component     (default 0.5)
        focal_weight : weight for Focal component    (default 0.5)
        focal_gamma  : focusing parameter γ          (default 2.0)
        focal_alpha  : foreground weight α           (default 0.25)
        dice_smooth  : Dice smoothing constant ε     (default 1e-6)
    """

    def __init__(
        self,
        dice_weight:  float = 0.5,
        focal_weight: float = 0.5,
        focal_gamma:  float = 2.0,
        focal_alpha:  float = 0.25,
        dice_smooth:  float = 1e-6,
    ):
        super().__init__()
        assert abs(dice_weight + focal_weight - 1.0) < 1e-5, (
            "dice_weight + focal_weight should equal 1.0"
        )
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight

        self.dice_loss  = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            logits  : (B, 1, H, W)  raw model output (pre-sigmoid)
            targets : (B, 1, H, W)  ground-truth binary masks

        Returns:
            total_loss : scalar tensor
            loss_dict  : {"total": ..., "dice": ..., "focal": ...}
        """
        probs = torch.sigmoid(logits)

        l_dice  = self.dice_loss(probs, targets)
        l_focal = self.focal_loss(logits, targets)

        total = self.dice_weight * l_dice + self.focal_weight * l_focal

        return total, {
            "total": total.item(),
            "dice":  l_dice.item(),
            "focal": l_focal.item(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Optional: Tversky Loss  (α controls FP, β controls FN penalty)
#  Useful if you want to penalise false negatives more heavily
# ─────────────────────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalisation of Dice.
    Set alpha=0.5, beta=0.5 → Dice Loss.
    Set alpha=0.3, beta=0.7 → penalise FN more (recall-focused).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        TP = (probs * targets).sum(dim=1)
        FP = (probs * (1 - targets)).sum(dim=1)
        FN = ((1 - probs) * targets).sum(dim=1)

        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1.0 - tversky).mean()


# ─────────────────────────────────────────────────────────────────────────────
#  Quick unit test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W = 4, 512, 512
    logits  = torch.randn(B, 1, H, W)
    targets = torch.randint(0, 2, (B, 1, H, W)).float()

    criterion = HybridLoss(dice_weight=0.5, focal_weight=0.5)
    loss, loss_dict = criterion(logits, targets)

    print(f"Total loss : {loss_dict['total']:.4f}")
    print(f"Dice  loss : {loss_dict['dice']:.4f}")
    print(f"Focal loss : {loss_dict['focal']:.4f}")
    print("Loss OK ✓")
