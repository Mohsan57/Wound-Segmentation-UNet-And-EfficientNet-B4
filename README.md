# Wound Segmentation — UNet + EfficientNet-B4

Production-grade binary segmentation of wound images using a UNet decoder
with an EfficientNet-B4 ImageNet-pretrained encoder and Hybrid Loss
(Dice + Focal).

---

## Project Structure

```
wound_seg/
├── config.py        ← All hyperparameters live here
├── dataset.py       ← WoundDataset: YOLO polygon → mask + augmentations
├── loss.py          ← HybridLoss (DiceLoss + FocalLoss)
├── model.py         ← UNet + EfficientNet-B4 via smp
├── metrics.py       ← Dice, IoU, Precision, Recall, Hausdorff
├── train.py         ← Full training loop
├── evaluate.py      ← Validation evaluation + visualisations
├── inference.py     ← Prediction + ONNX / TFLite / CoreML export
└── requirements.txt
```

---

## Expected Dataset Layout

```
.                          ← cfg.data_root
├── images/
│   ├── train/             ← training RGB images (.jpg / .png)
│   └── val/               ← validation RGB images
├── labels/
│   ├── train/             ← YOLO polygon .txt labels
│   └── val/
├── masks/                 ← (optional) pre-made binary PNG masks
└── wound.txt              ← YOLO dataset config (not used by this code)
```

### YOLO label format
Each `.txt` file mirrors its image:
```
<class_id>  x1 y1 x2 y2 ... xN yN
```
All coordinates are **normalised** [0, 1]. The dataset module converts
these polygon points into rasterised binary masks automatically.

---

## Installation

```bash
pip install -r requirements.txt
```

For GPU (CUDA 12.x):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Training

### 1. Configure

Edit `config.py` — key settings:

| Parameter | Default | Notes |
|---|---|---|
| `data_root` | `"."` | Root of your dataset |
| `image_size` | `512` | Resize all images to 512×512 |
| `batch_size` | `8` | Reduce to 4 if OOM |
| `num_epochs` | `100` | Early stopping at patience=15 |
| `learning_rate` | `1e-4` | AdamW |
| `dice_weight` | `0.5` | Hybrid Loss Dice weight |
| `focal_weight` | `0.5` | Hybrid Loss Focal weight |
| `use_amp` | `True` | Mixed precision (fp16) |
| `grad_accumulation_steps` | `2` | Effective batch = batch × steps |

### 2. Run training

```bash
python train.py
```

Training runs in **two phases** automatically:
- **Phase 1** (epochs 1–10): Encoder frozen, only decoder is trained.
  Fast convergence, avoids overwriting ImageNet features immediately.
- **Phase 2** (epoch 11+): Full fine-tuning with a 10× lower LR on encoder.

Checkpoints are saved to `checkpoints/`:
- `best_model.pth`  — best validation Dice ever seen
- `last_model.pth`  — latest epoch

TensorBoard logs are saved to `logs/`. View with:
```bash
tensorboard --logdir logs/
```

---

## Evaluation

```bash
python evaluate.py --checkpoint checkpoints/best_model.pth
```

Outputs in `eval_results/`:
- `per_image_metrics.csv`       — Dice, IoU, Precision, Recall per image
- `evaluation_summary.png`      — Bar chart of aggregate metrics
- `visualisations/`             — 5 best + 5 worst prediction panels
  - Each panel: `[Input | GT Mask | Predicted | Overlay]`

### Production targets
| Metric | Minimum | Production |
|---|---|---|
| Dice | > 0.80 | **> 0.88** |
| IoU | > 0.72 | **> 0.82** |
| Recall | > 0.80 | **> 0.85** |
| Inference | < 200 ms | **< 80 ms** |

---

## Inference

### Single image
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/wound.jpg
```
Saves `<stem>_mask.png` and `<stem>_overlay.png` to `outputs/`.

### Batch (folder of images)
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pth \
    --images_dir images/val \
    --output_dir results/
```

### Python API
```python
from inference import WoundPredictor

predictor = WoundPredictor("checkpoints/best_model.pth")
mask, probability_map, overlay_bgr = predictor.predict("wound.jpg")

wound_coverage = mask.mean() * 100
print(f"Wound area: {wound_coverage:.1f}%")
```

---

## Mobile Export

### Android (TFLite)
```bash
pip install onnx2tf tensorflow
python inference.py --export tflite --checkpoint checkpoints/best_model.pth
```
Output: `exports/wound_seg.tflite`

### iOS (CoreML)
```bash
pip install coremltools
python inference.py --export coreml --checkpoint checkpoints/best_model.pth
```
Output: `exports/WoundSeg.mlmodel`

### ONNX only (cross-platform)
```bash
python inference.py --export onnx --checkpoint checkpoints/best_model.pth
```

---

## Hybrid Loss — How It Works

```
L_hybrid = 0.5 × L_dice  +  0.5 × L_focal

L_dice  = 1 - (2·|P∩G| + ε) / (|P| + |G| + ε)

L_focal = -α · (1 - p_t)^γ · log(p_t)
```

| Loss | Role |
|---|---|
| **Dice** | Optimises overlap metric directly; robust to class imbalance |
| **Focal** | Focuses on hard pixels (wound boundaries); down-weights easy background |

Weights can be tuned in `config.py`:
- `dice_weight` + `focal_weight` must sum to 1.0
- `focal_gamma = 2.0` — increase for harder datasets
- `focal_alpha = 0.25` — foreground weight; increase if wound is rare

---

## Troubleshooting

| Problem | Fix |
|---|---|
| CUDA OOM | Reduce `batch_size` to 4 or `image_size` to 384 |
| Low Dice on tiny wounds | Increase `focal_gamma` to 3.0 |
| Training unstable | Reduce `learning_rate` to 5e-5 |
| Mask not matching image | Check label stem matches image stem |
| No masks in masks/ dir | The code falls back to YOLO polygon labels automatically |
