"""
inference.py — Inference & Mobile Export
-----------------------------------------
Supports:
  • Single image inference (returns mask + overlay)
  • Batch inference on a folder
  • ONNX export  (cross-platform mobile)
  • TFLite export via onnx2tf  (Android)
  • CoreML export via coremltools  (iOS)

Usage:
    # Single image
    python inference.py --image path/to/image.jpg --checkpoint checkpoints/best_model.pth

    # Export to ONNX
    python inference.py --export onnx --checkpoint checkpoints/best_model.pth
"""

import argparse
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

from config import Config
from model  import build_model, load_checkpoint


# ─────────────────────────────────────────────────────────────────────────────
#  Pre / Post processing
# ─────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess(image_bgr: np.ndarray, image_size: int) -> torch.Tensor:
    """
    BGR numpy image → normalised (1, 3, H, W) float32 tensor.
    """
    img = cv2.resize(image_bgr, (image_size, image_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1)                  # HWC → CHW
    return torch.from_numpy(img).unsqueeze(0)      # → (1, 3, H, W)


def postprocess(
    logits:       torch.Tensor,
    original_hw:  Tuple[int, int],
    threshold:    float = 0.5,
) -> np.ndarray:
    """
    Logit tensor → resized binary mask (H_orig × W_orig).
    """
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()  # (H, W) float
    # Resize back to original resolution
    prob = cv2.resize(prob, (original_hw[1], original_hw[0]))
    return (prob >= threshold).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Overlay visualisation
# ─────────────────────────────────────────────────────────────────────────────

def draw_overlay(
    image_bgr: np.ndarray,
    mask:      np.ndarray,
    color:     Tuple[int, int, int] = (0, 255, 0),
    alpha:     float = 0.45,
) -> np.ndarray:
    """
    Draw semi-transparent mask overlay + green contour on image.
    Returns BGR image.
    """
    overlay  = image_bgr.copy()
    coloured = np.zeros_like(image_bgr)
    coloured[mask == 1] = color
    cv2.addWeighted(coloured, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw contour
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, color, 2)

    return overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Predictor class
# ─────────────────────────────────────────────────────────────────────────────

class WoundPredictor:
    """
    High-level inference wrapper.

    Example:
        predictor = WoundPredictor("checkpoints/best_model.pth")
        mask = predictor.predict("wound.jpg")
    """

    def __init__(
        self,
        checkpoint_path: str,
        config:          Optional[Config] = None,
        device:          Optional[str]    = None,
    ):
        self.cfg    = config or Config()
        self.device = torch.device(device or self.cfg.device)

        self.model = build_model(
            architecture    = self.cfg.architecture,
            encoder_name    = self.cfg.encoder_name,
            encoder_weights = None,             # weights loaded from checkpoint
            in_channels     = self.cfg.image_channels,
            num_classes     = self.cfg.num_classes,
            activation      = None,
        ).to(self.device)

        load_checkpoint(self.model, checkpoint_path, device=str(self.device))
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        threshold:  Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run inference on a single image.

        Returns:
            mask        : binary mask  H×W  uint8
            probability : probability map  H×W  float32
            overlay     : BGR image with coloured mask overlay
        """
        thr   = threshold or self.cfg.threshold
        image = cv2.imread(image_path)
        if image is None:
            raise IOError(f"Cannot read image: {image_path}")

        orig_h, orig_w = image.shape[:2]

        tensor = preprocess(image, self.cfg.image_size).to(self.device)
        logits = self.model(tensor)

        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
        prob   = cv2.resize(prob, (orig_w, orig_h))
        mask   = (prob >= thr).astype(np.uint8)
        overlay = draw_overlay(image, mask)

        return mask, prob, overlay

    @torch.no_grad()
    def predict_batch(
        self,
        images_dir:  str,
        output_dir:  str,
        threshold:   Optional[float] = None,
    ) -> None:
        """Run inference on all images in a folder and save results."""
        output_dir = Path(output_dir)
        (output_dir / "masks").mkdir(parents=True, exist_ok=True)
        (output_dir / "overlays").mkdir(parents=True, exist_ok=True)

        image_paths = sorted(Path(images_dir).glob("*"))
        image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        for i, img_path in enumerate(image_paths):
            mask, prob, overlay = self.predict(str(img_path), threshold)

            stem = img_path.stem
            cv2.imwrite(str(output_dir / "masks"   / f"{stem}_mask.png"),    mask * 255)
            cv2.imwrite(str(output_dir / "overlays" / f"{stem}_overlay.png"), overlay)

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(image_paths)}")

        print(f"Done. Results saved to {output_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX export
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx(
    checkpoint_path: str,
    output_path:     str = "exports/wound_seg.onnx",
    config:          Optional[Config] = None,
    opset:           int = 17,
) -> None:
    """
    Export model to ONNX.
    Compatible with TFLite (via onnx2tf) and CoreML (via onnx-coreml).
    """
    cfg    = config or Config()
    device = torch.device("cpu")

    model = build_model(
        architecture    = cfg.architecture,
        encoder_name    = cfg.encoder_name,
        encoder_weights = None,
        in_channels     = cfg.image_channels,
        num_classes     = cfg.num_classes,
        activation      = "sigmoid",            # include sigmoid in ONNX graph
    ).to(device)

    load_checkpoint(model, checkpoint_path, device="cpu")
    model.eval()

    dummy = torch.randn(1, cfg.image_channels, cfg.image_size, cfg.image_size)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        do_constant_folding=True,
    )
    print(f"ONNX model exported → {output_path}")

    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified ✓")
    except ImportError:
        print("Install onnx for verification:  pip install onnx")


# ─────────────────────────────────────────────────────────────────────────────
#  TFLite export  (Android)
# ─────────────────────────────────────────────────────────────────────────────

def export_tflite(
    onnx_path:   str = "exports/wound_seg.onnx",
    output_path: str = "exports/wound_seg.tflite",
) -> None:
    """
    Convert ONNX → TFLite using onnx2tf.
    Install: pip install onnx2tf
    """
    try:
        import onnx2tf
    except ImportError:
        print("Install onnx2tf:  pip install onnx2tf")
        return

    import subprocess
    cmd = [
        "onnx2tf",
        "-i", onnx_path,
        "-o", str(Path(output_path).parent),
        "-oiqt",                   # INT8 quantisation
        "--non_verbose",
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"TFLite model → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CoreML export  (iOS)
# ─────────────────────────────────────────────────────────────────────────────

def export_coreml(
    onnx_path:    str = "exports/wound_seg.onnx",
    output_path:  str = "exports/WoundSeg.mlmodel",
    image_size:   int = 512,
) -> None:
    """
    Convert ONNX → CoreML using coremltools.
    Install: pip install coremltools
    """
    try:
        import coremltools as ct
    except ImportError:
        print("Install coremltools:  pip install coremltools")
        return

    model = ct.converters.onnx.convert(
        model=onnx_path,
        minimum_ios_deployment_target="14",
    )

    model.short_description  = "Wound segmentation — UNet + EfficientNet-B4"
    model.input_description["input"]   = "RGB wound image"
    model.output_description["output"] = "Binary wound mask probability"

    model.save(output_path)
    print(f"CoreML model → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wound Segmentation Inference & Export")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--image",      type=str, default=None,  help="Single image path")
    parser.add_argument("--images_dir", type=str, default=None,  help="Folder of images")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--export",     type=str, default=None,
                        choices=["onnx", "tflite", "coreml"],
                        help="Export format for mobile deployment")
    args = parser.parse_args()

    cfg = Config()

    # ── Export mode ──────────────────────────────────────────────────────────
    if args.export == "onnx":
        export_onnx(args.checkpoint, output_path="exports/wound_seg.onnx", config=cfg)

    elif args.export == "tflite":
        export_onnx(args.checkpoint, output_path="exports/wound_seg.onnx", config=cfg)
        export_tflite("exports/wound_seg.onnx", "exports/wound_seg.tflite")

    elif args.export == "coreml":
        export_onnx(args.checkpoint, output_path="exports/wound_seg.onnx", config=cfg)
        export_coreml("exports/wound_seg.onnx", "exports/WoundSeg.mlmodel")

    # ── Inference mode ────────────────────────────────────────────────────────
    elif args.image:
        predictor = WoundPredictor(args.checkpoint, config=cfg)
        mask, prob, overlay = predictor.predict(args.image, threshold=args.threshold)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stem = Path(args.image).stem
        cv2.imwrite(str(out / f"{stem}_mask.png"),    mask * 255)
        cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)
        wound_pct = mask.mean() * 100
        print(f"Mask saved.  Wound coverage: {wound_pct:.1f}%")

    elif args.images_dir:
        predictor = WoundPredictor(args.checkpoint, config=cfg)
        predictor.predict_batch(args.images_dir, args.output_dir, args.threshold)

    else:
        parser.print_help()
