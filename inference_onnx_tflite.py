"""
inference_onnx_tflite.py — Unified ONNX & TFLite Model Inference and Benchmark
-------------------------------------------------------------------------------
Loads and runs inference on:
  1. ONNX model (using onnxruntime)
  2. TFLite Float32 model (using tensorflow)
  3. TFLite Float16 model (using tensorflow)
  4. TFLite Full Integer Quantized INT8 model (using tensorflow)

This script preprocesses the input image, runs inference on all 4 models,
compares their latencies, computes prediction similarity (IoU & Dice vs ONNX),
and saves visual overlays for comparison.

Usage:
    python inference_onnx_tflite.py --image input/processed_42463520.png
"""

import argparse
import os
import time
import cv2
import numpy as np
from pathlib import Path

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Pre / Post Processing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_image(image_bgr, target_size=512):
    """
    Preprocess BGR image:
    1. Resize to target size (512x512)
    2. Convert BGR to RGB
    3. Normalise to [0, 1]
    4. Normalise with ImageNet mean/std
    5. Transpose HWC -> CHW (1, 3, H, W)
    """
    img = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = img.transpose(2, 0, 1) # NCHW
    img = np.expand_dims(img, axis=0)
    return img


def postprocess_mask(prob_map, orig_w, orig_h, threshold=0.5):
    """
    Resize probability map to original image shape and threshold to get binary mask.
    """
    prob_map_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_map_resized >= threshold).astype(np.uint8)
    return binary_mask, prob_map_resized


def draw_overlay(image_bgr, mask, color=(0, 255, 0), alpha=0.4):
    """
    Draw a semi-transparent colored mask overlay and contour on the original image.
    """
    overlay = image_bgr.copy()
    colored_mask = np.zeros_like(image_bgr)
    colored_mask[mask == 1] = color
    
    cv2.addWeighted(colored_mask, alpha, overlay, 1.0 - alpha, 0, overlay)
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    return overlay


def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def calculate_iou(mask_a, mask_b):
    """Calculate Intersection over Union (IoU) of two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def calculate_dice(mask_a, mask_b):
    """Calculate Dice Coefficient of two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    total_pixels = mask_a.sum() + mask_b.sum()
    if total_pixels == 0:
        return 1.0 if intersection == 0 else 0.0
    return (2.0 * intersection) / total_pixels


# ─────────────────────────────────────────────────────────────────────────────
#  Inference Runners
# ─────────────────────────────────────────────────────────────────────────────

class ONNXPredictor:
    """ONNX Model Predictor wrapper using ONNX Runtime."""
    def __init__(self, model_path):
        self.model_path = model_path
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError("onnxruntime is not installed. Install with: pip install onnxruntime")
            
        providers = ['CPUExecutionProvider']
        self.session = self.ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def predict(self, input_tensor):
        """
        Runs inference. 
        Input layout: NCHW (1, 3, 512, 512)
        Output range checked dynamically for Sigmoid activation.
        """
        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        raw_out = outputs[0].squeeze() # (H, W)
        
        # Apply sigmoid if model outputs logits
        min_val, max_val = raw_out.min(), raw_out.max()
        if min_val < 0.0 or max_val > 1.0:
            prob_map = sigmoid(raw_out)
        else:
            prob_map = raw_out
            
        return prob_map


class TFLitePredictor:
    """TFLite Model Predictor wrapper supporting FP32, FP16, and INT8 models."""
    def __init__(self, model_path):
        self.model_path = model_path
        try:
            import tensorflow as tf
            self.tf = tf
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
                self.interpreter = Interpreter(model_path=str(model_path))
            except ImportError:
                raise ImportError(
                    "Neither tensorflow nor tflite_runtime is installed. "
                    "Please install either 'tensorflow' or 'tflite-runtime'."
                )
        
        try:
            self.interpreter.allocate_tensors()
        except RuntimeError as e:
            # Catch known CPU float16 input type mismatch error
            if "conv.cc" in str(e) or "kTfLiteFloat16" in str(e):
                raise RuntimeError(
                    f"TFLite CPU interpreter does not support native float16 input/output tensors directly. "
                    f"To run FP16 models on CPU, the model should be converted with float32 inputs/outputs "
                    f"(with only internal weights quantized to float16). Original error: {e}"
                )
            else:
                raise e

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        
        # Parse inputs
        self.input_shape = self.input_details['shape']
        self.input_dtype = self.input_details['dtype']
        self.input_scale, self.input_zero_point = self.input_details['quantization']
        
        # Parse outputs
        self.output_dtype = self.output_details['dtype']
        self.output_scale, self.output_zero_point = self.output_details['quantization']
        
        # Determine layout
        # ONNX uses NCHW: (1, 3, H, W)
        # TFLite conversion might keep NCHW or transpose to NHWC: (1, H, W, 3)
        if self.input_shape[1] == 3 or self.input_shape[1] == 1:
            self.layout = 'NCHW'
        elif self.input_shape[3] == 3 or self.input_shape[3] == 1:
            self.layout = 'NHWC'
        else:
            self.layout = 'NCHW' # Default fallback
            
    def predict(self, input_tensor):
        """
        Runs TFLite inference, handling layouts and quantization.
        """
        # 1. Adapt layout if necessary
        # input_tensor is NCHW (1, 3, H, W)
        tflite_input = input_tensor.copy()
        if self.layout == 'NHWC':
            # Transpose: NCHW -> NHWC
            tflite_input = tflite_input.squeeze(0).transpose(1, 2, 0)
            tflite_input = np.expand_dims(tflite_input, axis=0)
            
        # 2. Apply Quantization for INT8 models
        is_quantized = (self.input_dtype == np.int8 or self.input_dtype == np.uint8)
        if is_quantized:
            # Scale and shift float values to quantized integer values
            tflite_input = (tflite_input / self.input_scale) + self.input_zero_point
            # Clip and cast
            if self.input_dtype == np.int8:
                tflite_input = np.clip(np.round(tflite_input), -128, 127).astype(np.int8)
            else:
                tflite_input = np.clip(np.round(tflite_input), 0, 255).astype(np.uint8)
        else:
            tflite_input = tflite_input.astype(self.input_dtype)
            
        # 3. Set input tensor and invoke
        self.interpreter.set_tensor(self.input_details['index'], tflite_input)
        self.interpreter.invoke()
        
        # 4. Get output tensor
        raw_output = self.interpreter.get_tensor(self.output_details['index'])
        
        # 5. Apply Dequantization for INT8 models
        if raw_output.dtype == np.int8 or raw_output.dtype == np.uint8:
            raw_output = raw_output.astype(np.float32)
            # Map quantized integer back to float32
            raw_output = (raw_output - self.output_zero_point) * self.output_scale
            
        # 6. Squeeze and format output map -> shape (H, W)
        prob_map = raw_output.squeeze()
        
        # Apply sigmoid if output contains logits
        min_val, max_val = prob_map.min(), prob_map.max()
        if min_val < 0.0 or max_val > 1.0:
            prob_map = sigmoid(prob_map)
            
        return prob_map


# ─────────────────────────────────────────────────────────────────────────────
#  Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(image_path, models_dir, output_dir, threshold):
    # Set paths
    img_path = Path(image_path)
    models_path = Path(models_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Model paths (restore all models)
    model_configs = {
        "ONNX":        models_path / "wound_seg.onnx",
        "TFLite FP32": models_path / "wound_seg_float32.tflite",
        "TFLite FP16": models_path / "wound_seg_float16.tflite",
        "TFLite INT8": models_path / "wound_seg_full_integer_quant.tflite"
    }
    
    # Load input image
    image = cv2.imread(str(img_path))
    if image is None:
        raise FileNotFoundError(f"Input image not found: {img_path}")
    orig_h, orig_w = image.shape[:2]
    
    # Preprocess once for all models
    preprocessed_img = preprocess_image(image)
    
    results = {}
    reference_mask = None # We will use the ONNX mask as reference for similarity metrics
    
    print("\n" + "="*70)
    print(f" Wound Segmentation Inference & Benchmark on: {img_path.name}")
    print("="*70)
    
    for name, m_path in model_configs.items():
        if not m_path.exists():
            print(f"[Warning] Skipping {name}: Model file not found at {m_path}")
            continue
            
        print(f"Loading {name} model...")
        try:
            if name == "ONNX":
                predictor = ONNXPredictor(str(m_path))
            else:
                predictor = TFLitePredictor(str(m_path))
        except Exception as e:
            print(f"[Error] Failed to load {name}: {e}")
            continue
            
        # Benchmark inference time (Warm-up + N runs)
        # INT8 on CPU is extremely slow (~90s per run). We run 1 pass to avoid hanging the process.
        runs = 1 if name == "TFLite INT8" else 5
        
        print(f"Profiling {name} inference ({runs} run{'s' if runs > 1 else ''})...")
        if name == "TFLite INT8":
            print("⚠️  [NOTICE] TFLite INT8 CPU inference is extremely heavy for this model (EfficientNet-B4 + UNet).")
            print("   It will take approximately 1.5 minutes to complete 1 execution pass. Please wait...")
            
        # Warm-up run
        _ = predictor.predict(preprocessed_img)
        
        start_time = time.perf_counter()
        for i in range(runs):
            prob_map = predictor.predict(preprocessed_img)
            if name == "TFLite INT8":
                print("   -> [INT8 Pass Completed]")
        avg_latency_ms = ((time.perf_counter() - start_time) / runs) * 1000
        
        # Postprocess outputs
        binary_mask, prob_map_resized = postprocess_mask(prob_map, orig_w, orig_h, threshold)
        
        # Calculate coverage area
        coverage = (binary_mask.sum() / binary_mask.size) * 100
        
        # Store results
        results[name] = {
            "mask": binary_mask,
            "prob": prob_map_resized,
            "latency": avg_latency_ms,
            "coverage": coverage
        }
        
        # Set reference mask if ONNX
        if name == "ONNX":
            reference_mask = binary_mask
            
        # Draw overlay and save image
        overlay = draw_overlay(image, binary_mask)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_{name.lower().replace(' ', '_')}_overlay.png"), overlay)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_{name.lower().replace(' ', '_')}_mask.png"), binary_mask * 255)
        print(f"  - Latency: {avg_latency_ms:.2f} ms | Wound Area: {coverage:.2f}%")
        
    if not results:
        print("[Error] No models were successfully run.")
        return
        
    # Print comparison table (use standard ASCII characters for Windows console support)
    print("\n" + "="*75)
    print(f" {'Model Type':<15} | {'Latency (ms)':<14} | {'Dice (vs ONNX)':<16} | {'IoU (vs ONNX)':<15} | {'Wound %':<8}")
    print("="*75)
    
    for name, res in results.items():
        if reference_mask is not None:
            dice = calculate_dice(reference_mask, res["mask"])
            iou = calculate_iou(reference_mask, res["mask"])
        else:
            dice, iou = 1.0, 1.0
            
        print(f" {name:<15} | {res['latency']:<14.2f} | {dice:<16.4f} | {iou:<15.4f} | {res['coverage']:<8.2f}%")
    print("="*75)
    print(f"Results and visual overlays saved to: {out_dir}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX and TFLite Inference & Benchmarking")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--models_dir", type=str, default="checkpoints", help="Directory where models are stored")
    parser.add_argument("--output_dir", type=str, default="outputs/comparison", help="Directory to save visual overlays")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification probability threshold")
    args = parser.parse_args()
    
    run_benchmark(args.image, args.models_dir, args.output_dir, args.threshold)
