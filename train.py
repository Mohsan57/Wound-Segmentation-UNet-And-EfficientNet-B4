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
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
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

def get_logger(log_dir: str, name: str = "train", rank: int = 0) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if rank == 0:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        logger.addHandler(logging.NullHandler())

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
    is_ddp: bool = False,
    rank: int = 0,
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

        # NaN/Inf Loss Protection
        if not torch.isfinite(loss):
            if rank == 0:
                logger.error(f"Non-finite loss detected at step {step+1}")
            raise RuntimeError(f"NaN/Inf loss detected at step {step+1}: loss = {loss.item()}")

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
        if rank == 0 and (step + 1) % 50 == 0:
            elapsed = time.time() - start
            mem_info = ""
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / 1024**3
                mem_info = f"  mem={mem:.2f}GB"
            logger.info(
                f"  Epoch {epoch:03d}  step {step+1:04d}/{n_batches}  "
                f"loss={loss_dict['total']:.4f}  "
                f"dice={loss_dict['dice']:.4f}  "
                f"focal={loss_dict['focal']:.4f}  "
                f"time={elapsed:.1f}s"
                f"{mem_info}"
            )

    total_loss_val = total_loss
    total_dice_val = total_dice
    total_focal_val = total_focal

    if is_ddp:
        # Sum local values across all ranks
        metrics_tensor = torch.tensor([total_loss_val, total_dice_val, total_focal_val, float(n_batches)], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        total_loss_val, total_dice_val, total_focal_val, total_batches = metrics_tensor.tolist()
        n = max(total_batches, 1)
    else:
        n = max(n_batches, 1)

    return {
        "loss":  total_loss_val  / n,
        "dice":  total_dice_val  / n,
        "focal": total_focal_val / n,
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
    is_ddp: bool = False,
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

    if is_ddp:
        # Aggregate validation loss across all ranks
        loss_tensor = torch.tensor([total_loss, float(len(loader))], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1)

        # Aggregate TP, FP, FN, TN and sample count across all ranks
        metrics_tensor = torch.tensor([
            seg_metrics.tp, seg_metrics.fp, seg_metrics.fn, seg_metrics.tn, float(seg_metrics.n_samples)
        ], device=device)
        dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        seg_metrics.tp = metrics_tensor[0].item()
        seg_metrics.fp = metrics_tensor[1].item()
        seg_metrics.fn = metrics_tensor[2].item()
        seg_metrics.tn = metrics_tensor[3].item()
        seg_metrics.n_samples = int(metrics_tensor[4].item())
    else:
        avg_loss = total_loss / max(len(loader), 1)

    results = seg_metrics.compute()
    results["loss"] = avg_loss
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: Config, resume_path: Optional[str] = None) -> Optional[list]:
    # Check for DDP environment variables
    is_ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = torch.device(cfg.device)

    set_seed(cfg.seed)
    
    # Configure logger (rank-aware)
    logger = get_logger(cfg.log_dir, rank=rank)
    
    # Configure SummaryWriter (only on rank 0)
    writer = SummaryWriter(log_dir=cfg.log_dir) if rank == 0 else None
    
    if rank == 0:
        logger.info("=" * 60)
        logger.info("  Wound Segmentation Training - UNet + EfficientNet-B4")
        logger.info("=" * 60)
        logger.info(f"  Device          : {device} (DDP={is_ddp}, World Size={world_size})")
        logger.info(f"  Image size      : {cfg.image_size}")
        logger.info(f"  Batch size      : {cfg.batch_size} (per GPU, effective={cfg.batch_size * world_size})")
        logger.info(f"  Epochs          : {cfg.num_epochs}")
        logger.info(f"  LR              : {cfg.learning_rate}")
        logger.info(f"  Dice weight     : {cfg.dice_weight}")
        logger.info(f"  Focal weight    : {cfg.focal_weight}")
        if resume_path:
            logger.info(f"  Resume Path     : {resume_path}")
        logger.info("=" * 60)

    try:
        # Datasets
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

        if is_ddp:
            train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
            val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        else:
            train_sampler = None
            val_sampler = None

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None),
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
            sampler=train_sampler
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
            sampler=val_sampler
        )
        
        if rank == 0:
            logger.info(f"  Train batches : {len(train_loader)} (total={len(train_loader) * world_size})")
            logger.info(f"  Val   batches : {len(val_loader)} (total={len(val_loader) * world_size})")

        # Model
        model = build_model(
            architecture    = cfg.architecture,
            encoder_name    = cfg.encoder_name,
            encoder_weights = cfg.encoder_weights,
            in_channels     = cfg.image_channels,
            num_classes     = cfg.num_classes,
            activation      = None,               # raw logits -> loss handles sigmoid
        ).to(device)

        # Loss
        criterion = HybridLoss(
            dice_weight      = cfg.dice_weight,
            focal_weight     = cfg.focal_weight,
            focal_gamma      = cfg.focal_gamma,
            focal_alpha      = cfg.focal_alpha,
            dice_smooth      = cfg.dice_smooth,
            use_tversky_loss = cfg.use_tversky_loss,
            tversky_alpha    = cfg.tversky_alpha,
            tversky_beta     = cfg.tversky_beta,
        )

        # Variables to track training progress / state
        start_epoch = 1
        best_dice = 0.0
        encoder_unfrozen = False
        UNFREEZE_EPOCH = 10

        # Initialization (scratch vs resume)
        raw_model = model

        if resume_path and os.path.exists(resume_path):
            if rank == 0:
                logger.info(f"[Resume] Loading checkpoint from {resume_path}")
            try:
                checkpoint = torch.load(resume_path, map_location=device)
            except Exception as e:
                checkpoint = torch.load(resume_path, map_location=device, weights_only=False)

            if isinstance(checkpoint, dict) and "model" in checkpoint:
                # Full training checkpoint
                raw_model.load_state_dict(checkpoint["model"])
                start_epoch = checkpoint["epoch"] + 1
                best_dice = checkpoint.get("best_dice", 0.0)
                if rank == 0:
                    logger.info(f"[Resume] Loaded full training checkpoint. Resuming from epoch {start_epoch} (best validation Dice so far: {best_dice:.4f})")
                
                # Explicitly load encoder unfreeze state with fallback
                if "encoder_unfrozen" in checkpoint:
                    encoder_unfrozen = checkpoint["encoder_unfrozen"]
                    if rank == 0:
                        logger.info(f"[Resume] Loaded explicit encoder_unfrozen state: {encoder_unfrozen}")
                else:
                    encoder_unfrozen = start_epoch > UNFREEZE_EPOCH + 1
                    if rank == 0:
                        logger.info(f"[Resume] Inferred encoder_unfrozen state: {encoder_unfrozen}")

                if encoder_unfrozen:
                    # Re-create optimizer with Phase 2 param groups
                    unfreeze_encoder(raw_model)
                    head_params = list(raw_model.segmentation_head.parameters()) if hasattr(raw_model, "segmentation_head") else []
                    optimizer = optim.AdamW([
                        {"params": raw_model.encoder.parameters(), "lr": cfg.learning_rate * 0.1},
                        {"params": raw_model.decoder.parameters(), "lr": cfg.learning_rate},
                        {"params": head_params, "lr": cfg.learning_rate},
                    ], weight_decay=cfg.weight_decay)
                    scheduler = WarmupCosineScheduler(
                        optimizer,
                        warmup_epochs = 0,
                        T_max         = cfg.scheduler_t_max,
                        eta_min       = cfg.scheduler_eta_min,
                    )
                    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
                    if rank == 0:
                        logger.info("[Resume] Encoder is unfrozen (Phase 2)")
                else:
                    # Re-create optimizer with Phase 1 param groups
                    freeze_encoder(raw_model)
                    optimizer = optim.AdamW(
                        filter(lambda p: p.requires_grad, raw_model.parameters()),
                        lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay,
                    )
                    scheduler = WarmupCosineScheduler(
                        optimizer,
                        warmup_epochs = cfg.warmup_epochs,
                        T_max         = cfg.scheduler_t_max,
                        eta_min       = cfg.scheduler_eta_min,
                    )
                    scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
                    if rank == 0:
                        logger.info("[Resume] Encoder is frozen (Phase 1)")

                # Load states
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    if rank == 0:
                        logger.info("[Resume] Loaded optimizer state")
                if "scheduler" in checkpoint:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                    if rank == 0:
                        logger.info("[Resume] Loaded scheduler state")
                if "scaler" in checkpoint:
                    scaler.load_state_dict(checkpoint["scaler"])
                    if rank == 0:
                        logger.info("[Resume] Loaded AMP scaler state")

                # Restore RNG states
                if "torch_rng_state" in checkpoint:
                    torch.set_rng_state(checkpoint["torch_rng_state"].cpu() if isinstance(checkpoint["torch_rng_state"], torch.Tensor) else checkpoint["torch_rng_state"])
                    if rank == 0:
                        logger.info("[Resume] Restored PyTorch RNG state")
                if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                    try:
                        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
                        if rank == 0:
                            logger.info("[Resume] Restored CUDA RNG state")
                    except Exception as e:
                        if rank == 0:
                            logger.warning(f"[Resume] Failed to restore CUDA RNG state: {e}")
                if "numpy_rng_state" in checkpoint:
                    np.random.set_state(checkpoint["numpy_rng_state"])
                    if rank == 0:
                        logger.info("[Resume] Restored NumPy RNG state")
                if "python_rng_state" in checkpoint:
                    random.setstate(checkpoint["python_rng_state"])
                    if rank == 0:
                        logger.info("[Resume] Restored Python RNG state")
            else:
                # Weights-only state dict (e.g. best_model.pth weight-only file)
                state_dict = checkpoint["model"] if (isinstance(checkpoint, dict) and "model" in checkpoint) else checkpoint
                raw_model.load_state_dict(state_dict)
                if rank == 0:
                    logger.info("[Resume] Loaded weights-only state dict. Starting training from epoch 1 (Phase 1).")
                
                freeze_encoder(raw_model)
                optimizer = optim.AdamW(
                    filter(lambda p: p.requires_grad, raw_model.parameters()),
                    lr=cfg.learning_rate,
                    weight_decay=cfg.weight_decay,
                )
                scheduler = WarmupCosineScheduler(
                    optimizer,
                    warmup_epochs = cfg.warmup_epochs,
                    T_max         = cfg.scheduler_t_max,
                    eta_min       = cfg.scheduler_eta_min,
                )
                scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
                encoder_unfrozen = False
        else:
            if resume_path and rank == 0:
                logger.warning(f"[Resume] Warning: Checkpoint path '{resume_path}' does not exist. Starting training from scratch.")
            # Setup from scratch (Phase 1)
            freeze_encoder(raw_model)
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, raw_model.parameters()),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
            scheduler = WarmupCosineScheduler(
                optimizer,
                warmup_epochs = cfg.warmup_epochs,
                T_max         = cfg.scheduler_t_max,
                eta_min       = cfg.scheduler_eta_min,
            )
            scaler = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
            encoder_unfrozen = False

        # Set up DDP / SyncBatchNorm if running in DDP mode
        if is_ddp:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
            model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        else:
            model = raw_model

        # Early stopping
        early_stop = EarlyStopping(patience=cfg.early_stopping_patience, mode="max")
        if best_dice > 0.0:
            early_stop.best = best_dice

        history = []

        # Training loop
        for epoch in range(start_epoch, cfg.num_epochs + 1):
            if is_ddp:
                train_sampler.set_epoch(epoch)

            # Unfreeze at epoch UNFREEZE_EPOCH
            if epoch == UNFREEZE_EPOCH + 1 and not encoder_unfrozen:
                unfreeze_encoder(model)
                # Re-create optimizer with all params and lower LR for encoder
                raw_model_unwrapped = model.module if hasattr(model, "module") else model
                head_params = list(raw_model_unwrapped.segmentation_head.parameters()) if hasattr(raw_model_unwrapped, "segmentation_head") else []
                optimizer = optim.AdamW([
                    {"params": raw_model_unwrapped.encoder.parameters(), "lr": cfg.learning_rate * 0.1},
                    {"params": raw_model_unwrapped.decoder.parameters(), "lr": cfg.learning_rate},
                    {"params": head_params, "lr": cfg.learning_rate},
                ], weight_decay=cfg.weight_decay)
                scheduler = WarmupCosineScheduler(
                    optimizer,
                    warmup_epochs = 0,
                    T_max         = cfg.scheduler_t_max,
                    eta_min       = cfg.scheduler_eta_min,
                )
                scaler  = GradScaler(enabled=cfg.use_amp and device.type == "cuda")
                encoder_unfrozen = True
                if rank == 0:
                    logger.info(f"[Epoch {epoch}] Encoder unfrozen - full fine-tuning begins")

            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler,
                device, cfg.grad_accumulation_steps, cfg.grad_clip_norm,
                epoch, logger, is_ddp=is_ddp, rank=rank
            )

            # Validate
            val_metrics = validate(model, val_loader, criterion, device, cfg.threshold, is_ddp=is_ddp)

            # Scheduler step
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            # Log to TensorBoard (rank 0 only)
            if rank == 0 and writer is not None:
                writer.add_scalar("Loss/train",      train_metrics["loss"], epoch)
                writer.add_scalar("Loss/val",        val_metrics["loss"],   epoch)
                writer.add_scalar("Dice/val",        val_metrics["dice"],   epoch)
                writer.add_scalar("IoU/val",         val_metrics["iou"],    epoch)
                writer.add_scalar("Precision/val",   val_metrics["precision"], epoch)
                writer.add_scalar("Recall/val",      val_metrics["recall"],    epoch)
                writer.add_scalar("LR",              current_lr,            epoch)

            # Console summary (rank 0 only)
            if rank == 0:
                mem_info = ""
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / 1024**3
                    mem_info = f"  gpu_mem={mem:.2f}GB"
                logger.info(
                    f"Epoch {epoch:03d}/{cfg.num_epochs}  "
                    f"train_loss={train_metrics['loss']:.4f}  "
                    f"val_loss={val_metrics['loss']:.4f}  "
                    f"val_dice={val_metrics['dice']:.4f}  "
                    f"val_iou={val_metrics['iou']:.4f}  "
                    f"prec={val_metrics['precision']:.4f}  "
                    f"rec={val_metrics['recall']:.4f}  "
                    f"lr={current_lr:.2e}"
                    f"{mem_info}"
                )

            history.append({"epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}})

            # Save best checkpoint (rank 0 only)
            val_dice = val_metrics["dice"]
            if val_dice > best_dice:
                best_dice = val_dice
                if rank == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        epoch=epoch,
                        metrics=val_metrics,
                        best_dice=best_dice,
                        path=str(Path(cfg.checkpoint_dir) / "best_model.pth"),
                        encoder_unfrozen=encoder_unfrozen,
                    )
                    logger.info(f"  * New best Dice: {best_dice:.4f}")

            # Save latest checkpoint (rank 0 only)
            if rank == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch,
                    metrics=val_metrics,
                    best_dice=best_dice,
                    path=str(Path(cfg.checkpoint_dir) / "last_model.pth"),
                    encoder_unfrozen=encoder_unfrozen,
                )

            # Early stopping
            if early_stop.step(val_dice):
                if rank == 0:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        if rank == 0 and writer is not None:
            writer.close()

        # Save training history to JSON (rank 0 only)
        if rank == 0:
            import json
            history_path = Path(cfg.log_dir) / "history.json"
            try:
                with open(history_path, "w") as f:
                    json.dump(history, f, indent=2)
                logger.info(f"Training history saved to {history_path}")
            except Exception as e:
                logger.error(f"Failed to save training history to JSON: {e}")

            logger.info(f"\nTraining complete.  Best val Dice = {best_dice:.4f}")
            logger.info(f"Best model saved -> {cfg.checkpoint_dir}/best_model.pth")
            return history

    finally:
        if is_ddp:
            dist.destroy_process_group()
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wound Segmentation Model")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="checkpoints/last_model.pth",
        type=str,
        default=None,
        help="Path to checkpoint .pth file to resume training from. If flag is provided without path, defaults to checkpoints/last_model.pth"
    )
    args = parser.parse_args()

    cfg = Config()
    train(cfg, resume_path=args.resume)
