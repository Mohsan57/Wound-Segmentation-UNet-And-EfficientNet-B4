# Wound Segmentation — UNet + EfficientNet-B4

Production-grade binary segmentation of wound images using a UNet decoder
with an EfficientNet-B4 ImageNet-pretrained encoder and Hybrid Loss
(Dice + Focal).

**Kaggle Dataset**: [Wound Segmentation YOLO Format](https://www.kaggle.com/datasets/mohsanyaseen/wound-segmentation-yolo-format)

---

## Project Structure

```
wound_seg/
├── config.py        ← All hyperparameters live here
├── dataset.py       ← WoundDataset: YOLO polygon → mask + augmentations
├── loss.py          ← HybridLoss (Dice/TverskyLoss + FocalLoss)
├── model.py         ← UNet + EfficientNet-B4 via smp (MobileNetV2 on mobile_mode)
├── metrics.py       ← Dice, IoU, Precision, Recall, Hausdorff
├── train.py         ← Full training loop
├── evaluate.py      ← Validation evaluation + visualisations
├── inference.py     ← Prediction + ONNX / TFLite / CoreML export
├── benchmark.py     ← Latency benchmarking (PyTorch / ONNX / TFLite)
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
| `data_root` | `"wound_dataset"` | Root of your dataset |
| `image_size` | `512` | Target H and W resolution (overridden to `384` in mobile mode) |
| `batch_size` | `8` | Reduce to 4 if OOM |
| `num_epochs` | `100` | Early stopping at patience=15 |
| `learning_rate` | `1e-4` | AdamW |
| `use_amp` | `True` | Mixed precision (fp16) |
| `grad_accumulation_steps` | `2` | Effective batch = batch × steps |
| `mobile_mode` | `False` | Swaps backbone to `mobilenet_v2`, disables attention, and sets size to `384` |
| `use_tversky_loss` | `True` | Activates recall-focused Tversky Loss (pins `focal_gamma = 3.0`) |
| `tversky_alpha` | `0.3` | Penalises False Positives (FP) in Tversky Loss |
| `tversky_beta` | `0.7` | Penalises False Negatives (FN) in Tversky Loss |
| `num_calibration_images` | `400` | Calibration images loaded from validation split for INT8 quantization |

### 2. Run training

**Single-GPU or CPU Training**:
```bash
# Start training from scratch
python train.py
```

**Multi-GPU Training (DDP)**:
This project supports PyTorch's Distributed Data Parallel (DDP) for high-performance multi-GPU training (e.g. utilizing the 2 GPUs provided on Kaggle).

To train on 2 GPUs using `torchrun` in a terminal:
```bash
torchrun --nproc_per_node=2 train.py
```

Or inside a Kaggle / Jupyter Notebook cell:
```python
!torchrun --nproc_per_node=2 train.py
```

### 3. Resuming Training

You can resume training from a previously saved checkpoint or state dictionary. This is highly useful if training was interrupted or if you want to initialize training from a specific set of pre-trained weights.

We support three ways to use the `--resume` argument:

1. **Auto-resume from the last checkpoint**:
   Resumes training from the default `checkpoints/last_model.pth` file if it exists, restoring the optimizer state, learning rate scheduler state, AMP loss scaler, and starting from the next epoch:
   ```bash
   python train.py --resume
   ```

2. **Resume from a specific full checkpoint**:
   Resumes training from a specific path to a full checkpoint, restoring the exact training state (epoch, optimizer, scheduler, scaler):
   ```bash
   python train.py --resume checkpoints/best_model.pth
   ```

3. **Initialize from a weights-only model**:
   If the provided `.pth` file is a weights-only state dictionary (such as `best_model.pth` files exported for inference that do not contain optimizer metadata), the script will load the weights and start training from Epoch 1 (Phase 1, encoder frozen):
   ```bash
   python train.py --resume best_model.pth
   ```

---

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

### Android (TFLite INT8 Quantized)
```bash
pip install onnx2tf tensorflow onnxscript
# Set UTF-8 encoding environment variable to prevent Windows terminal print errors
$env:PYTHONIOENCODING="utf-8"  # PowerShell
# Export:
python inference.py --export tflite --checkpoint checkpoints/best_model.pth
```
Output: `exports/wound_seg.tflite`
*   **Calibration**: Automatically loads 400 validation split images as a representative dataset, calculating highly accurate integer scale ranges and preventing boundary noise degradation.

### iOS (CoreML FP16 Package)
```bash
pip install coremltools
python inference.py --export coreml --checkpoint checkpoints/best_model.pth
```
Output: `exports/WoundSeg.mlpackage`
*   **Neural Engine Optimization**: Targets iOS 15+ minimum deployment and exports directly to the modern `.mlpackage` format using `FLOAT16` precision for full hardware acceleration on Apple's Neural Engine.

### ONNX only (cross-platform)
```bash
python inference.py --export onnx --checkpoint checkpoints/best_model.pth
```

---

## Latency Benchmarking

Use the benchmarking utility to measure inference speeds on CPU, GPU, or exported formats (ONNX / TFLite):

```bash
# Benchmark the PyTorch checkpoint on CPU/GPU, along with ONNX and TFLite exports:
python benchmark.py \
    --model checkpoints/best_model.pth \
    --onnx exports/wound_seg.onnx \
    --tflite exports/wound_seg.tflite
```

---

## Hybrid Loss & Tversky Optimization

```
L_hybrid = 0.5 × L_dice_tversky  +  0.5 × L_focal

L_tversky = 1 - (TP + ε) / (TP + α·FP + β·FN + ε)

L_focal   = -α · (1 - p_t)^γ · log(p_t)
```

| Component | Default | Role |
|---|---|---|
| **Tversky Loss** | $\alpha=0.3$, $\beta=0.7$ | Penalizes False Negatives (FN) heavily, ensuring complete coverage of the wound boundaries (Recall focus). |
| **Focal Loss** | $\gamma=3.0$, $\alpha=0.25$ | Dynamically updates $\gamma$ to 3.0 when Tversky is active to push model focus towards hard boundary pixels. |

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
