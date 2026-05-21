"""
train.py — Complete training pipeline
--------------------------------------
Features:
  ✓ AMP (fp16 mixed precision)
  ✓ Gradient accumulation
  ✓ Two-phase training: frozen encoder → full fine-tune
  ✓ Cosine Annealing LR schedule with linear warmup
  ✓ Early stopping
  ✓ Best & last checkpoint saving
  ✓ TensorBoard logging
  ✓ Per-epoch console summary

Run:
    python train.py
"""

import os
import sys
import time
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast

from config  import Config
from dataset import WoundDataset, get_train_transforms, get_val_transforms
from loss    import HybridLoss
from metrics import SegmentationMetrics
from model   import build_model, freeze_encoder, unfreeze_encoder, save_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
#  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
#  Logger
# ─────────────────────────────────────────────────────────────────────────────

def get_logger(log_dir: str, name: str = "train") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ─────────────────────────────────────────────────────────────────────────────
#  LR Scheduler with linear warmup
# ─────────────────────────────────────────────────────────────────────────────

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup for warmup_epochs, then Cosine Annealing."""

    def __init__(self, optimizer, warmup_epochs: int, T_max: int, eta_min: float = 1e-6):
        self.warmup_epochs = warmup_epochs
        self.T_max         = T_max
        self.eta_min       = eta_min
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / max(self.warmup_epochs, 1)
            return [base_lr * factor for base_lr in self.base_lrs]

        cos_epoch = epoch - self.warmup_epochs
        cos_total = self.T_max - self.warmup_epochs
        factor = 0.5 * (1 + np.cos(np.pi * cos_epoch / max(cos_total, 1)))
        return [
            self.eta_min + (base_lr - self.eta_min) * factor
            for base_lr in self.base_lrs
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Early Stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# ─────────────────────────────────────────────────────────────────────────────
#  One epoch — train
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    grad_accumulation_steps: int,
    grad_clip_norm: float,
    epoch: int,
    logger,
) -> dict:
    model.train()
    total_loss   = 0.0
    total_dice   = 0.0
    total_focal  = 0.0
    n_batches    = len(loader)
    start        = time.time()

    optimizer.zero_grad()

    for step, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast(enabled=scaler.is_enabled()):
            logits = model(images)
            loss, loss_dict = criterion(logits, masks)
            loss = loss / grad_accumulation_steps   # scale for accumulation

        scaler.scale(loss).backward()

        if (step + 1) % grad_accumulation_steps == 0 or (step + 1) == n_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss  += loss_dict["total"]
        total_dice  += loss_dict["dice"]
        total_focal += loss_dict["focal"]

        # Print progress every 50 steps
        if (step + 1) % 50 == 0:
            elapsed = time.time() - start
            logger.info(
                f"  Epoch {epoch:03d}  step {step+1:04d}/{n_batches}  "
                f"loss={loss_dict['total']:.4f}  "
                f"dice={loss_dict['dice']:.4f}  "
                f"focal={loss_dict['focal']:.4f}  "
                f"time={elapsed:.1f}s"
            )

    n = max(n_batches, 1)
    return {
        "loss":  total_loss  / n,
        "dice":  total_dice  / n,
        "focal": total_focal / n,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  One epoch — validate
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    device,
    threshold: float = 0.5,
) -> dict:
    model.eval()
    total_loss  = 0.0
    seg_metrics = SegmentationMetrics(threshold=threshold)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        logits = model(images)
        _, loss_dict = criterion(logits, masks)
        total_loss  += loss_dict["total"]

        probs = torch.sigmoid(logits)
        seg_metrics.update(probs, masks)

    results = seg_metrics.compute()
    results["loss"] = total_loss / max(len(loader), 1)
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config) -> None:
    set_seed(cfg.seed)
    logger = get_logger(cfg.log_dir)
    writer = SummaryWriter(log_dir=cfg.log_dir)
    device = torch.device(cfg.device)
    
    
    logger.info("=" * 60)
    logger.info("  Wound Segmentation Training — UNet + EfficientNet-B4")
    logger.info("=" * 60)
    logger.info(f"  Device          : {device}")
    logger.info(f"  Image size      : {cfg.image_size}")
    logger.info(f"  Batch size      : {cfg.batch_size}")
    logger.info(f"  Epochs          : {cfg.num_epochs}")
    logger.info(f"  LR              : {cfg.learning_rate}")
    logger.info(f"  Dice weight     : {cfg.dice_weight}")
    logger.info(f"  Focal weight    : {cfg.focal_weight}")
    logger.info("=" * 60)

    # ── Datasets ──────────────────────────────────────────────────────────
    data_root = Path(cfg.data_root)

    train_ds = WoundDataset(
        images_dir     = str(data_root / "images" / "train"),
        labels_dir     = str(data_root / "labels" / "train"),
        masks_dir      = str(data_root / "masks"),
        transform      = get_train_transforms(cfg.image_size) if cfg.use_augmentation else get_val_transforms(cfg.image_size),
        image_size     = cfg.image_size,
        label_smoothing= cfg.label_smoothing,
    )

    val_ds = WoundDataset(
        images_dir = str(data_root / "images" / "val"),
        labels_dir = str(data_root / "labels" / "val"),
        masks_dir  = str(data_root / "masks"),
        transform  = get_val_transforms(cfg.image_size),
        image_size = cfg.image_size,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    logger.info(f"  Train batches : {len(train_loader)}")
    logger.info(f"  Val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(
        architecture    = cfg.architecture,
        encoder_name    = cfg.encoder_name,
        encoder_weights = cfg.encoder_weights,
        in_channels     = cfg.image_channels,
        num_classes     = cfg.num_classes,
        activation      = None,               # raw logits → loss handles sigmoid
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = HybridLoss(
        dice_weight  = cfg.dice_weight,
        focal_weight = cfg.focal_weight,
        focal_gamma  = cfg.focal_gamma,
        focal_alpha  = cfg.focal_alpha,
        dice_smooth  = cfg.dice_smooth,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs = cfg.warmup_epochs,
        T_max         = cfg.scheduler_t_max,
        eta_min       = cfg.scheduler_eta_min,
    )

    # ── AMP scaler ────────────────────────────────────────────────────────
    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")

    # ── Early stopping ────────────────────────────────────────────────────
    early_stop = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")

    # ── Two-phase training ────────────────────────────────────────────────
    # Phase 1 (epochs 1–10): freeze encoder, train decoder only
    # Phase 2 (epoch 11+):   unfreeze everything
    freeze_encoder(model)
    UNFREEZE_EPOCH = 10
    encoder_unfrozen = False

    best_dice = 0.0
    history   = []

    # ─────────────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.num_epochs + 1):

        # Unfreeze at epoch UNFREEZE_EPOCH
        if epoch == UNFREEZE_EPOCH + 1 and not encoder_unfrozen:
            unfreeze_encoder(model)
            # Re-create optimizer with all params and lower LR for encoder
            optimizer = optim.AdamW([
                {"params": model.encoder.parameters(), "lr": cfg.learning_rate * 0.1},
                {"params": model.decoder.parameters(), "lr": cfg.learning_rate},
                {"params": model.segmentation_head.parameters(), "lr": cfg.learning_rate},
            ], weight_decay=cfg.weight_decay)
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs = 0,
                T_max         = cfg.scheduler_t_max,
                eta_min       = cfg.scheduler_eta_min,
            )
            scaler  = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
            encoder_unfrozen = True
            logger.info(f"[Epoch {epoch}] Encoder unfrozen — full fine-tuning begins")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, cfg.grad_accumulation_steps, cfg.grad_clip_norm,
            epoch, logger,
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, cfg.threshold)

        # Scheduler step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Log to TensorBoard
        writer.add_scalar("Loss/train",      train_metrics["loss"], epoch)
        writer.add_scalar("Loss/val",        val_metrics["loss"],   epoch)
        writer.add_scalar("Dice/val",        val_metrics["dice"],   epoch)
        writer.add_scalar("IoU/val",         val_metrics["iou"],    epoch)
        writer.add_scalar("Precision/val",   val_metrics["precision"], epoch)
        writer.add_scalar("Recall/val",      val_metrics["recall"],    epoch)
        writer.add_scalar("LR",              current_lr,            epoch)

        # Console summary
        logger.info(
            f"Epoch {epoch:03d}/{cfg.num_epochs}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_dice={val_metrics['dice']:.4f}  "
            f"val_iou={val_metrics['iou']:.4f}  "
            f"prec={val_metrics['precision']:.4f}  "
            f"rec={val_metrics['recall']:.4f}  "
            f"lr={current_lr:.2e}"
        )

        history.append({"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

        # Save best checkpoint
        val_dice = val_metrics["dice"]
        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                path=str(Path(cfg.checkpoint_dir) / "best_model.pth"),
            )
            logger.info(f"  ★ New best Dice: {best_dice:.4f}")

        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, epoch, val_metrics,
            path=str(Path(cfg.checkpoint_dir) / "last_model.pth"),
        )

        # Early stopping
        if early_stop.step(val_dice):
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    writer.close()
    logger.info(f"\nTraining complete.  Best val Dice = {best_dice:.4f}")
    logger.info(f"Best model saved → {cfg.checkpoint_dir}/best_model.pth")
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = Config()
    train(cfg)
