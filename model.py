"""
model.py — UNet + EfficientNet-B4 Segmentation Model
------------------------------------------------------
Built on top of segmentation_models_pytorch (smp).

Supports:
  • unet           — classic skip-connection UNet decoder
  • unetplusplus   — nested dense skip connections (slightly better, slightly slower)
  • deeplabv3plus  — ASPP + low-level features (good for thin structures)

Default: UNet + EfficientNet-B4 pretrained on ImageNet.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Optional

try:
    import segmentation_models_pytorch as smp
except ImportError:
    raise ImportError(
        "Install segmentation_models_pytorch:\n"
        "  pip install segmentation-models-pytorch"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Model factory
# ─────────────────────────────────────────────────────────────────────────────

_ARCHITECTURE_MAP = {
    "unet":          smp.Unet,
    "unetplusplus":  smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
}


def build_model(
    architecture:    str = "unet",
    encoder_name:    str = "efficientnet-b4",
    encoder_weights: str = "imagenet",
    in_channels:     int = 3,
    num_classes:     int = 1,
    activation:      Optional[str] = None,    # None = raw logits (recommended for training)
    decoder_attention_type: Optional[str] = "scse",  # "scse" | None
) -> nn.Module:
    """
    Build a segmentation model.

    Args:
        architecture    : "unet" | "unetplusplus" | "deeplabv3plus"
        encoder_name    : any timm/smp encoder, e.g. "efficientnet-b4"
        encoder_weights : "imagenet" or None
        in_channels     : 3 for RGB
        num_classes     : 1 for binary segmentation
        activation      : None (raw logits) | "sigmoid" | "softmax2d"
        decoder_attention_type : "scse" adds channel + spatial squeeze-excitation
                                 attention to every decoder block (UNet / UNet++).
                                 Ignored for DeepLabV3+ (not supported by smp).
                                 Set to None to disable.
    Returns:
        nn.Module
    """
    if architecture not in _ARCHITECTURE_MAP:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from {list(_ARCHITECTURE_MAP.keys())}"
        )

    ModelClass = _ARCHITECTURE_MAP[architecture]
 
    # DeepLabV3+ does not expose a decoder_attention_type argument in smp
    kwargs = dict(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=activation,
    )
    
    # Warn/guard if attention is active on mobile/lightweight encoders
    is_mobile_backbone = any(m in encoder_name.lower() for m in ["mobilenet", "ghostnet", "efficientnet-b0", "lcnet"])
    if is_mobile_backbone and decoder_attention_type is not None:
        import warnings
        warnings.warn(
            f"[Warning] Attention type '{decoder_attention_type}' is enabled with lightweight mobile backbone '{encoder_name}'. "
            f"Global pooling or fully connected attention layers can bottleneck execution on mobile NPUs/GPUs. "
            f"Consider setting decoder_attention_type=None or using mobile_mode=True.",
            UserWarning
        )

    if architecture != "deeplabv3plus" and decoder_attention_type is not None:
        kwargs["decoder_attention_type"] = decoder_attention_type
 
    model = ModelClass(**kwargs)
 
    # Print param summary
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    attn_label = decoder_attention_type if (architecture != "deeplabv3plus" and decoder_attention_type) else "none"
    print(f"[Model] Architecture  : {architecture} + {encoder_name}  (decoder attention: {attn_label})")
    print(f"[Model] Total params  : {total_params:,}")
    print(f"[Model] Trainable     : {trainable_params:,}")
 
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  Optional: freeze / unfreeze encoder for fine-tuning strategy
# ─────────────────────────────────────────────────────────────────────────────

def freeze_encoder(model: nn.Module) -> None:
    """Freeze encoder weights — train decoder only (phase 1)."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Encoder frozen. Trainable params: {trainable:,}")


def unfreeze_encoder(model: nn.Module) -> None:
    """Unfreeze all weights — full fine-tuning (phase 2)."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Encoder unfrozen. Trainable params: {trainable:,}")


# ─────────────────────────────────────────────────────────────────────────────
#  Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    metrics,
    best_dice,
    path,
    encoder_unfrozen: bool = False,
):
    checkpoint_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "metrics": metrics,
        "best_dice": best_dice,
        "encoder_unfrozen": encoder_unfrozen,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint_dict["cuda_rng_state"] = torch.cuda.get_rng_state_all()

    tmp_path = f"{path}.tmp"
    torch.save(checkpoint_dict, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer=None,
    device: str = "cpu",
) -> dict:
    print(f"[Checkpoint] Loading <- {path}  (device: {device})")
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as e:
        ckpt = torch.load(path, map_location=device, weights_only=False)


    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"[Checkpoint] Loaded <- {path}  (epoch {ckpt.get('epoch', '?')})")
    return ckpt.get("metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
#  Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = build_model(
        architecture="unet",
        encoder_name="efficientnet-b4",
        encoder_weights=None,       # skip download in CI
        in_channels=3,
        num_classes=1,
    )
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")        # expect (2, 1, 512, 512)
    print("Model OK ✓")
