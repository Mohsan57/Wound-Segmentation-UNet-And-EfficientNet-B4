"""
evaluate.py — Full Evaluation on Validation Set
-------------------------------------------------
Produces:
  • Per-image metrics CSV
  • Aggregate metrics (Dice, IoU, Precision, Recall, Specificity)
  • Hausdorff Distance (boundary accuracy)
  • Confusion matrix
  • Visual grid of worst / best predictions

Usage:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from config  import Config
from dataset import WoundDataset, get_val_transforms
from metrics import SegmentationMetrics, dice_per_sample, hausdorff_distance
from model   import build_model, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """CHW float tensor → HWC uint8 RGB."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Binary HxW mask → BGR image."""
    m = (mask * 255).astype(np.uint8)
    return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)


def overlay_mask(image_rgb: np.ndarray, mask: np.ndarray,
                 color=(0, 255, 0), alpha=0.45) -> np.ndarray:
    bgr     = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    colored = np.zeros_like(bgr)
    colored[mask == 1] = color
    result  = cv2.addWeighted(colored, alpha, bgr, 1 - alpha, 0)
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, color, 2)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Per-image evaluation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_dataset(
    model:      torch.nn.Module,
    dataset:    WoundDataset,
    device:     torch.device,
    threshold:  float = 0.5,
    batch_size: int = 4,
) -> List[Dict]:
    """
    Run model on every image in dataset, collect per-image metrics.

    Returns:
        List of dicts with keys:
          image_path, dice, iou, precision, recall, hausdorff
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    model.eval()

    results = []
    img_idx = 0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        probs  = torch.sigmoid(logits)

        # Per-sample Dice
        dices  = dice_per_sample(probs, masks, threshold=threshold)

        preds_np  = (probs  >= threshold).float().cpu().numpy()
        masks_np  =  masks.cpu().numpy()

        for b in range(images.size(0)):
            pred_b = preds_np[b, 0]         # H×W  float
            mask_b = (masks_np[b, 0] >= 0.5).astype(np.uint8)

            tp = float((pred_b * mask_b).sum())
            fp = float((pred_b * (1 - mask_b)).sum())
            fn = float(((1 - pred_b) * mask_b).sum())
            tn = float(((1 - pred_b) * (1 - mask_b)).sum())
            s  = 1e-6

            iou         = (tp + s) / (tp + fp + fn + s)
            precision   = (tp + s) / (tp + fp + s)
            recall      = (tp + s) / (tp + fn + s)
            specificity = (tn + s) / (tn + fp + s)

            hd = hausdorff_distance(pred_b.astype(bool), mask_b.astype(bool))

            if img_idx < len(dataset.image_paths):
                img_name = dataset.image_paths[img_idx].name
            else:
                img_name = f"image_{img_idx}"

            results.append({
                "image":       img_name,
                "dice":        round(float(dices[b]), 4),
                "iou":         round(iou, 4),
                "precision":   round(precision, 4),
                "recall":      round(recall, 4),
                "specificity": round(specificity, 4),
                "hausdorff":   round(hd, 2),
            })
            img_idx += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Visual grid — best & worst predictions
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_visual_grid(
    model:      torch.nn.Module,
    dataset:    WoundDataset,
    device:     torch.device,
    results:    List[Dict],
    output_dir: str,
    n:          int = 5,
    threshold:  float = 0.5,
) -> None:
    """
    Save a grid of N best and N worst predictions side-by-side:
      [Image | GT Mask | Predicted | Overlay]
    """
    out = Path(output_dir) / "visualisations"
    out.mkdir(parents=True, exist_ok=True)

    # Sort by Dice
    sorted_res  = sorted(results, key=lambda x: x["dice"])
    worst_names = {r["image"] for r in sorted_res[:n]}
    best_names  = {r["image"] for r in sorted_res[-n:]}

    target_names = worst_names | best_names
    collected    = {}

    for idx, img_path in enumerate(dataset.image_paths):
        if img_path.name not in target_names:
            continue

        image_t, mask_t = dataset[idx]

        image_t = image_t.unsqueeze(0).to(device)
        mask_t  = mask_t.unsqueeze(0).to(device)

        logit = model(image_t)
        prob  = torch.sigmoid(logit)
        pred  = (prob >= threshold).squeeze().cpu().numpy().astype(np.uint8)
        gt    = (mask_t.squeeze().cpu().numpy() >= 0.5).astype(np.uint8)
        rgb   = denormalise(image_t.squeeze(0).cpu())

        # Build 4-panel row: image | GT | pred | overlay
        h, w = 256, 256
        img_r   = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (w, h))
        gt_r    = cv2.resize(mask_to_bgr(gt),   (w, h))
        pred_r  = cv2.resize(mask_to_bgr(pred), (w, h))
        ov_r    = cv2.resize(overlay_mask(rgb, pred), (w, h))

        # Add label headers
        def add_header(img_bgr, text):
            out_img = img_bgr.copy()
            cv2.putText(out_img, text, (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            return out_img

        row = np.hstack([
            add_header(img_r,  "Input"),
            add_header(gt_r,   "GT Mask"),
            add_header(pred_r, "Predicted"),
            add_header(ov_r,   "Overlay"),
        ])

        dice_val = next((r["dice"] for r in results if r["image"] == img_path.name), 0.0)
        tag      = "BEST" if img_path.name in best_names else "WORST"
        fname    = f"{tag}_dice{dice_val:.3f}_{img_path.stem}.png"
        cv2.imwrite(str(out / fname), row)
        collected[img_path.name] = row

    print(f"[Visualise] Saved {len(collected)} panels -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  Save confusion matrix image
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(metrics_agg: Dict, output_dir: str) -> None:
    """Render a simple text-based confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping confusion matrix plot.")
        return

    prec = metrics_agg["precision"]
    rec  = metrics_agg["recall"]
    spec = metrics_agg["specificity"]
    dice = metrics_agg["dice"]
    iou  = metrics_agg["iou"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Wound Segmentation — Evaluation Results", fontsize=14, fontweight="bold")

    # Bar chart
    metric_names  = ["Dice", "IoU", "Precision", "Recall", "Specificity"]
    metric_values = [dice,   iou,   prec,        rec,      spec]
    colors = ["#4CAF50" if v >= 0.85 else "#FF9800" if v >= 0.70 else "#F44336"
              for v in metric_values]
    bars = axes[0].bar(metric_names, metric_values, color=colors, edgecolor="black")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Aggregate Metrics")
    axes[0].axhline(y=0.85, color="green", linestyle="--", alpha=0.5, label="Production target")
    axes[0].legend()
    for bar, val in zip(bars, metric_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01, f"{val:.3f}",
                     ha="center", va="bottom", fontsize=10)

    # Table of values
    table_data = [[n, f"{v:.4f}"] for n, v in zip(metric_names, metric_values)]
    axes[1].axis("off")
    tbl = axes[1].table(
        cellText=table_data,
        colLabels=["Metric", "Score"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.5, 2.0)
    axes[1].set_title("Summary Table")

    plt.tight_layout()
    out_path = Path(output_dir) / "evaluation_summary.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Summary chart -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main(cfg: Config, checkpoint_path: str, output_dir: str) -> None:
    device = torch.device(cfg.device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(
        architecture    = cfg.architecture,
        encoder_name    = cfg.encoder_name,
        encoder_weights = None,
        in_channels     = cfg.image_channels,
        num_classes     = cfg.num_classes,
        activation      = None,
    ).to(device)
    load_checkpoint(model, checkpoint_path, device=str(device))

    # ── Dataset ────────────────────────────────────────────────────────────
    data_root = Path(cfg.data_root)
    val_ds = WoundDataset(
        images_dir = str(data_root / "images" / "val"),
        labels_dir = str(data_root / "labels" / "val"),
        masks_dir  = str(data_root / "masks"),
        transform  = get_val_transforms(cfg.image_size),
        image_size = cfg.image_size,
    )
    print(f"[Evaluate] Val images: {len(val_ds)}")

    # ── Per-image metrics ──────────────────────────────────────────────────
    print("[Evaluate] Running inference …")
    results = evaluate_dataset(model, val_ds, device, cfg.threshold, cfg.batch_size)

    # ── Save CSV ───────────────────────────────────────────────────────────
    csv_path = Path(output_dir) / "per_image_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[Evaluate] Per-image CSV -> {csv_path}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    keys = ["dice", "iou", "precision", "recall", "specificity", "hausdorff"]
    agg  = {k: round(float(np.mean([r[k] for r in results])), 4) for k in keys}
    agg_std = {k: round(float(np.std([r[k] for r in results])), 4) for k in keys}

    print("\n" + "=" * 55)
    print("  AGGREGATE METRICS  (mean ± std)")
    print("=" * 55)
    for k in keys:
        bar = "█" * int(agg[k] * 20)
        print(f"  {k:<14} {agg[k]:.4f} ± {agg_std[k]:.4f}  |{bar}")
    print("=" * 55)

    # Production-grade thresholds check
    print("\n  Production targets:")
    print(f"  Dice  ≥ 0.88 :  {'✅ PASS' if agg['dice'] >= 0.88 else '❌ FAIL'} ({agg['dice']:.4f})")
    print(f"  IoU   ≥ 0.82 :  {'✅ PASS' if agg['iou']  >= 0.82 else '❌ FAIL'} ({agg['iou']:.4f})")
    print(f"  Recall≥ 0.85 :  {'✅ PASS' if agg['recall']>= 0.85 else '❌ FAIL'} ({agg['recall']:.4f})")

    # ── Visualisations ─────────────────────────────────────────────────────
    print("\n[Evaluate] Generating visual panels …")
    save_visual_grid(model, val_ds, device, results, output_dir, n=5, threshold=cfg.threshold)

    # ── Summary chart ──────────────────────────────────────────────────────
    save_confusion_matrix(agg, output_dir)

    # ── Worst performers report ────────────────────────────────────────────
    worst = sorted(results, key=lambda x: x["dice"])[:10]
    print("\n  10 Hardest images (lowest Dice):")
    for r in worst:
        print(f"    {r['image']:<50}  dice={r['dice']:.4f}  hd={r['hausdorff']:.1f}")


if __name__ == "__main__":
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--output_dir",  type=str, default="eval_results")
    parser.add_argument("--device", type=str, default=None, help="Device to run evaluation on (e.g., cuda or cpu)")
    args = parser.parse_args()

    cfg = Config()
    if args.device:
        cfg.device = args.device
    main(cfg, args.checkpoint, args.output_dir)
