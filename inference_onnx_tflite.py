"""
inference_onnx_tflite.py — Unified ONNX & TFLite Inference and Benchmark
-------------------------------------------------------------------------
Supports:
  1. ONNX  (onnxruntime)
  2. TFLite FP32
  3. TFLite INT8 full-integer-quant

FP16 TFLite is SKIPPED — TFLite CPU does not support native float16 tensors.
Run on GPU/NPU delegate if you need FP16.

Usage:
    python inference_onnx_tflite.py --image input/sample.png [--diagnose]

If INT8 output looks wrong, re-export with the fix in the docstring at the
bottom of this file.
"""

import argparse
import time
import cv2
import numpy as np
from pathlib import Path

# ── Must match training resolution AND onnx2tf calibration size ───────────────
MODEL_SIZE = 256

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Pre / Post Processing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_bgr, target_size=MODEL_SIZE):
    """BGR image -> normalised NCHW float32 tensor (1,3,H,W)."""
    img = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)   # (1,3,H,W)


def postprocess_mask(prob_map, orig_w, orig_h, threshold=0.5):
    """prob_map: (H,W) float32 in [0,1] -> binary mask at original size."""
    prob_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (prob_resized >= threshold).astype(np.uint8), prob_resized


def draw_overlay(image_bgr, mask, color=(0, 255, 0), alpha=0.4):
    overlay      = image_bgr.copy()
    colored_mask = np.zeros_like(image_bgr)
    colored_mask[mask == 1] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1.0 - alpha, 0, overlay)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a,  b).sum()
    return (inter / union) if union > 0 else (1.0 if inter == 0 else 0.0)

def dice(a, b):
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return (2.0 * inter / total) if total > 0 else (1.0 if inter == 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  ONNX Predictor
# ─────────────────────────────────────────────────────────────────────────────

class ONNXPredictor:
    def __init__(self, model_path):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("pip install onnxruntime")
        self.sess  = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name
        inp = self.sess.get_inputs()[0]
        out = self.sess.get_outputs()[0]
        print(f"  [ONNX] input  {inp.name}  shape={inp.shape}  dtype={inp.type}")
        print(f"  [ONNX] output {out.name}  shape={out.shape}  dtype={out.type}")

    def predict(self, nchw):
        """nchw: float32 (1,3,H,W) -> prob_map (H,W) in [0,1]"""
        raw = self.sess.run([self.oname], {self.iname: nchw})[0]  # (1,1,H,W) or (1,H,W)
        raw = raw.squeeze()                                         # -> (H,W)
        return sigmoid(raw) if (raw.min() < 0.0 or raw.max() > 1.0) else raw


# ─────────────────────────────────────────────────────────────────────────────
#  TFLite Predictor
# ─────────────────────────────────────────────────────────────────────────────

class TFLitePredictor:
    """
    Handles FP32 and INT8 full-integer-quant TFLite models produced by onnx2tf.

    Key facts about onnx2tf output:
      • Input  layout : always NHWC  (1, H, W, C)
      • Output layout : NHWC  (1, H, W, 1)  — but onnx2tf sometimes collapses
                        spatial dims to (1,1,1,1) due to a known squeeze bug.
                        We detect and raise a clear error if that happens.
      • INT8 quant    : full_integer_quant bakes sigmoid into the graph.
                        Output is already in [0,1] after dequantisation.
    """

    def __init__(self, model_path):
        try:
            import tensorflow as tf
            self._interp = tf.lite.Interpreter(model_path=str(model_path))
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
                self._interp = Interpreter(model_path=str(model_path))
            except ImportError:
                raise ImportError("pip install tensorflow  OR  pip install tflite-runtime")

        self._interp.allocate_tensors()

        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]

        self._inp_idx  = inp['index']
        self._out_idx  = out['index']
        self._inp_shape = inp['shape']          # e.g. [1,256,256,3]
        self._out_shape = out['shape']          # should be [1,H,W,1] or [1,1,H,W]
        self._inp_dtype = inp['dtype']
        self._out_dtype = out['dtype']

        self._inp_scale = float(inp['quantization'][0])
        self._inp_zp    = int(inp['quantization'][1])
        self._out_scale = float(out['quantization'][0])
        self._out_zp    = int(out['quantization'][1])

        # ── Layout detection ───────────────────────────────────────────────────
        # Confirmed from Step 4 verification:
        #   wound_seg_static_*.tflite input = [1, 256, 256, 3]  => NHWC
        # Generic rule: if dim[3] is channel count (1 or 3) => NHWC
        #               if dim[1] is channel count (1 or 3) and dim[2] != 3 => NCHW
        n, d1, d2, d3 = self._inp_shape
        if int(d3) in (1, 3):
            self._layout = 'NHWC'     # (1, H, W, C) — standard onnx2tf output
        elif int(d1) in (1, 3):
            self._layout = 'NCHW'     # (1, C, H, W) — rare, kept for safety
        else:
            self._layout = 'NHWC'     # fallback

        # ── Output shape sanity check ─────────────────────────────────────────
        # onnx2tf bug: sometimes collapses (1,1,H,W) -> (1,1,1,1)
        # This makes the model useless for segmentation — must re-export.
        spatial = [s for s in self._out_shape if s > 1]
        if len(spatial) < 2:
            raise ValueError(
                f"\n\n"
                f"  *** onnx2tf OUTPUT SHAPE BUG ***\n"
                f"  Model : {model_path}\n"
                f"  Output shape reported: {list(self._out_shape)}\n"
                f"  Expected: [1, {MODEL_SIZE}, {MODEL_SIZE}, 1] or [1, 1, {MODEL_SIZE}, {MODEL_SIZE}]\n\n"
                f"  onnx2tf collapsed your spatial output dimensions.\n"
                f"  Fix: re-export with the --output_unify_dynamic_shapes flag:\n\n"
                f"    onnx2tf -i exports/wound_seg.onnx \\\n"
                f"            -o exports \\\n"
                f"            -oiqt \\\n"
                f"            --output_unify_dynamic_shapes \\\n"
                f"            -cind input exports/calibration_data.npy \\\n"
                f"               '[[[[0.485,0.456,0.406]]]]' \\\n"
                f"               '[[[[0.229,0.224,0.225]]]]' \\\n"
                f"            --non_verbose\n\n"
                f"  If that flag is not available in your onnx2tf version, use:\n"
                f"    --keep_shape_absolutely_input_names input\n"
            )

        print(f"  [TFLite] input  shape={list(self._inp_shape)}  dtype={self._inp_dtype.__name__}  layout={self._layout}")
        print(f"  [TFLite] output shape={list(self._out_shape)}  dtype={self._out_dtype.__name__}")
        print(f"  [TFLite] quant  inp scale={self._inp_scale:.6f} zp={self._inp_zp}  |  out scale={self._out_scale:.6f} zp={self._out_zp}")

    def predict(self, nchw):
        """nchw: float32 (1,3,H,W) -> prob_map (H,W) in [0,1]"""

        # ── 1. Layout conversion ───────────────────────────────────────────────
        x = nchw.copy()
        if self._layout == 'NHWC':
            x = x.squeeze(0).transpose(1, 2, 0)[np.newaxis]   # (1,H,W,3)

        # ── 2. Quantise input (INT8 only) ──────────────────────────────────────
        is_int_in = self._inp_dtype in (np.int8, np.uint8)
        if is_int_in and self._inp_scale > 0.0:
            # TFLite spec: q = round(x / scale) + zero_point
            x = np.round(x / self._inp_scale) + self._inp_zp
            lo, hi = (-128, 127) if self._inp_dtype == np.int8 else (0, 255)
            x = np.clip(x, lo, hi).astype(self._inp_dtype)
        else:
            x = x.astype(self._inp_dtype)

        # ── 3. Invoke ──────────────────────────────────────────────────────────
        self._interp.set_tensor(self._inp_idx, x)
        self._interp.invoke()
        raw = self._interp.get_tensor(self._out_idx)

        # ── 4. Dequantise output (INT8 only) ───────────────────────────────────
        if raw.dtype in (np.int8, np.uint8) and self._out_scale > 0.0:
            # TFLite spec: x = (q - zero_point) * scale
            raw = (raw.astype(np.float32) - self._out_zp) * self._out_scale

        # ── 5. Squeeze to (H,W) ────────────────────────────────────────────────
        prob = raw.squeeze()   # (1,H,W,1) or (1,1,H,W) -> (H,W)

        # ── 6. Sigmoid only if output is clearly logit-space ──────────────────
        # full_integer_quant bakes sigmoid in -> output already in [0,1]
        vmin, vmax = float(prob.min()), float(prob.max())
        if vmin < -0.5 or vmax > 1.5:
            prob = sigmoid(prob)

        return prob


# ─────────────────────────────────────────────────────────────────────────────
#  Diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_tflite(model_path):
    """Print full tensor info. Use --diagnose flag to trigger this."""
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(model_path))
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
        interp = Interpreter(model_path=str(model_path))

    try:
        interp.allocate_tensors()
    except RuntimeError as e:
        print(f"\n  [Diagnose] allocate_tensors FAILED: {e}")
        print("  -> This model cannot run on CPU (likely FP16 native tensors). Skip it.\n")
        return

    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    print(f"\n── TFLite Diagnostic ── {Path(model_path).name}")
    print(f"  Input  idx={inp['index']}  shape={list(inp['shape'])}  dtype={inp['dtype'].__name__}")
    print(f"         scale={inp['quantization'][0]:.8f}  zp={inp['quantization'][1]}")
    print(f"  Output idx={out['index']}  shape={list(out['shape'])}  dtype={out['dtype'].__name__}")
    print(f"         scale={out['quantization'][0]:.8f}  zp={out['quantization'][1]}")
    print("─" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(image_path, models_dir, output_dir, threshold, diagnose=False):
    img_path    = Path(image_path)
    models_path = Path(models_dir)
    out_dir     = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_configs = {
        "ONNX":        models_path / "wound_seg.onnx",
        "TFLite FP32": models_path / "wound_seg_static_float32.tflite",
        # FP16 skipped — CPU doesn't support native float16 TFLite tensors
        "TFLite INT8": models_path / "wound_seg_static_full_integer_quant.tflite",
    }

    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    orig_h, orig_w = image.shape[:2]

    input_tensor = preprocess_image(image)
    print(f"\n[Input] shape={input_tensor.shape}  min={input_tensor.min():.3f}  max={input_tensor.max():.3f}")

    results        = {}
    reference_mask = None

    print("\n" + "="*70)
    print(f" Benchmark: {img_path.name}")
    print("="*70)

    for name, m_path in model_configs.items():
        if not m_path.exists():
            print(f"\n[Skip] {name}: {m_path} not found")
            continue

        print(f"\n[Load] {name}  ← {m_path.name}")

        if diagnose and name != "ONNX":
            diagnose_tflite(m_path)

        try:
            predictor = ONNXPredictor(str(m_path)) if name == "ONNX" else TFLitePredictor(str(m_path))
        except ValueError as e:
            # Shape bug — print the re-export instructions and continue
            print(e)
            continue
        except RuntimeError as e:
            if "kTfLiteFloat16" in str(e) or "conv.cc" in str(e):
                print(f"  [Skip] {name}: CPU does not support native FP16 tensors.")
            else:
                print(f"  [Error] {name}: {e}")
            continue
        except Exception as e:
            print(f"  [Error] {name}: {e}")
            continue

        runs = 1 if name == "TFLite INT8" else 5
        _ = predictor.predict(input_tensor)          # warm-up

        t0 = time.perf_counter()
        for _ in range(runs):
            prob_map = predictor.predict(input_tensor)
        latency_ms = ((time.perf_counter() - t0) / runs) * 1000

        print(f"  prob_map  shape={prob_map.shape}  min={prob_map.min():.4f}  max={prob_map.max():.4f}  mean={prob_map.mean():.4f}")

        binary_mask, _ = postprocess_mask(prob_map, orig_w, orig_h, threshold)
        coverage = binary_mask.sum() / binary_mask.size * 100

        results[name] = {"mask": binary_mask, "latency": latency_ms, "coverage": coverage}
        if name == "ONNX":
            reference_mask = binary_mask

        stem    = img_path.stem
        tag     = name.lower().replace(' ', '_')
        cv2.imwrite(str(out_dir / f"{stem}_{tag}_overlay.png"), draw_overlay(image, binary_mask))
        cv2.imwrite(str(out_dir / f"{stem}_{tag}_mask.png"),    binary_mask * 255)
        print(f"  Latency={latency_ms:.1f} ms  WoundArea={coverage:.2f}%")

    if not results:
        print("\n[Error] No models ran.")
        return

    print("\n" + "="*75)
    print(f" {'Model':<15} | {'ms':<8} | {'Dice vs ONNX':<14} | {'IoU vs ONNX':<13} | Wound%")
    print("="*75)
    for name, res in results.items():
        d = dice(reference_mask, res["mask"]) if reference_mask is not None else 1.0
        i = iou( reference_mask, res["mask"]) if reference_mask is not None else 1.0
        print(f" {name:<15} | {res['latency']:<8.1f} | {d:<14.4f} | {i:<13.4f} | {res['coverage']:.2f}%")
    print("="*75)
    print(f"\nOutputs → {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
#  Re-export instructions (if onnx2tf collapses output shape)
# ─────────────────────────────────────────────────────────────────────────────
"""
IF YOU SEE "onnx2tf OUTPUT SHAPE BUG" above, re-run the export like this:

    # Step 1: verify what your ONNX output shape actually is
    import onnxruntime as ort
    sess = ort.InferenceSession("exports/wound_seg.onnx")
    print(sess.get_outputs()[0].shape)   # should be ['batch_size', 1, 256, 256]

    # Step 2: re-export with shape preservation flag
    onnx2tf -i exports/wound_seg.onnx \
            -o exports \
            -oiqt \
            --output_unify_dynamic_shapes \
            -cind input exports/calibration_data.npy \
               '[[[[0.485,0.456,0.406]]]]' \
               '[[[[0.229,0.224,0.225]]]]' \
            --non_verbose

    # If --output_unify_dynamic_shapes is not recognised in your version:
    onnx2tf -i exports/wound_seg.onnx \
            -o exports \
            -oiqt \
            --keep_shape_absolutely_input_names input \
            -cind input exports/calibration_data.npy \
               '[[[[0.485,0.456,0.406]]]]' \
               '[[[[0.229,0.224,0.225]]]]' \
            --non_verbose

The key symptom: FP32 and INT8 .tflite both give output shape [1,1,1,1]
instead of [1,256,256,1]. The ONNX model is fine; only onnx2tf conversion
is broken.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True)
    parser.add_argument("--models_dir", default="exports")
    parser.add_argument("--output_dir", default="outputs/compare")
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--diagnose",   action="store_true",
                        help="Print full tensor info for each TFLite model")
    args = parser.parse_args()
    run_benchmark(args.image, args.models_dir, args.output_dir, args.threshold, args.diagnose)