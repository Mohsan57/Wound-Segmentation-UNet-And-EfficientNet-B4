"""
config.py — Central configuration for wound segmentation training.
All hyperparameters and paths live here. Change them here only.
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ─────────────────────────────────────────────
    #  Paths
    # ─────────────────────────────────────────────
    data_root: str = "wound_dataset"                       # Root of your dataset folder
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    export_dir: str = "exports"                # For TFLite / ONNX exports

    # ─────────────────────────────────────────────
    #  Model & Mode
    # ─────────────────────────────────────────────
    encoder_name: str = "efficientnet-b4"      # SMP encoder
    encoder_weights: str = "imagenet"          # Pretrained weights
    architecture: str = "unet"                 # unet | unetplusplus | deeplabv3plus
    num_classes: int = 1                       # Binary segmentation
    activation: str = "sigmoid"
    decoder_attention_type: str = "scse"       # "scse" | "none" — channel+spatial squeeze-excitation on decoder blocks
    mobile_mode: bool = False                  # Optimization for mobile targets

    # ─────────────────────────────────────────────
    #  Input
    # ─────────────────────────────────────────────
    image_size: int = 512                      # Resize both H and W to this
    image_channels: int = 3

    # ─────────────────────────────────────────────
    #  Training
    # ─────────────────────────────────────────────
    batch_size: int = 8
    num_workers: int = 4
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 15
    grad_accumulation_steps: int = 2           # Effective batch = batch_size × grad_accumulation_steps

    # ─────────────────────────────────────────────
    #  Loss Weights (Dice / Tversky + Focal)
    # ─────────────────────────────────────────────
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    focal_gamma: float = 2.0                   # Focus on hard examples (overridden to 3.0 if Tversky is active)
    focal_alpha: float = 0.25                  # Class balance factor
    dice_smooth: float = 1e-6
    
    use_tversky_loss: bool = True              # Enable recall-prioritised loss
    tversky_alpha: float = 0.3                 # Penalises FP (false positives)
    tversky_beta: float = 0.7                  # Penalises FN (false negatives)

    # ─────────────────────────────────────────────
    #  Scheduler  (Cosine Annealing + Warmup)
    # ─────────────────────────────────────────────
    warmup_epochs: int = 5
    scheduler_t_max: int = 50
    scheduler_eta_min: float = 1e-6

    # ─────────────────────────────────────────────
    #  Regularization / Augmentation / Quantization
    # ─────────────────────────────────────────────
    use_augmentation: bool = True
    label_smoothing: float = 0.05              # Slight label smoothing
    use_amp: bool = True                       # Automatic Mixed Precision (fp16)
    num_calibration_images: int = 400          # Number of validation images for TFLite calibration

    # ─────────────────────────────────────────────
    #  Threshold & Metrics
    # ─────────────────────────────────────────────
    threshold: float = 0.5                     # Sigmoid → binary mask threshold

    # ─────────────────────────────────────────────
    #  Device
    # ─────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ─────────────────────────────────────────────
    #  Reproducibility
    # ─────────────────────────────────────────────
    seed: int = 42

    def __post_init__(self):
        # Override parameters if in mobile mode
        if self.mobile_mode:
            self.encoder_name = "mobilenet_v2"
            self.decoder_attention_type = None  # Disable scse for hardware NPU speed
            self.image_size = 384               # Save FLOPs quadratically
            
        # Pin focal_gamma when Tversky Loss is active to strengthen boundary focus
        if self.use_tversky_loss:
            self.focal_gamma = 3.0

        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        Path(self.export_dir).mkdir(parents=True, exist_ok=True)

