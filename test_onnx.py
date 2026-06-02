"""
test_onnx.py — Test Wound Segmentation ONNX Model
---------------------------------------------------
Loads an input image, runs inference using ONNX Runtime,
and saves the binary mask and visual overlay.

Usage:
    python test_onnx.py --image input/processed_42463520.png --model checkpoints/wound_seg.onnx
"""

import argparse
import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# ImageNet normalization statistics
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess(image_bgr, target_size=512):
    """
    Preprocess BGR image to match model requirements:
    1. Resize to target size (512x512)
    2. Convert BGR to RGB
    3. Normalise to [0, 1]
    4. Normalise with ImageNet mean/std
    5. Transpose HWC -> CHW
    6. Expand dimensions -> (1, C, H, W)
    """
    # Resize
    img = cv2.resize(image_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Normalise
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    # HWC to CHW
    img = img.transpose(2, 0, 1)
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def sigmoid(x):
    """Numerically stable sigmoid function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def draw_overlay(image_bgr, mask, color=(0, 255, 0), alpha=0.4):
    """
    Draw a semi-transparent colored mask and solid contour on the original image.
    """
    overlay = image_bgr.copy()
    colored_mask = np.zeros_like(image_bgr)
    colored_mask[mask == 1] = color
    
    # Blend mask with original image
    cv2.addWeighted(colored_mask, alpha, overlay, 1.0 - alpha, 0, overlay)
    
    # Draw contours
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    
    return overlay

def main():
    parser = argparse.ArgumentParser(description="Inference on Wound Segmentation ONNX Model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="checkpoints/wound_seg.onnx", help="Path to ONNX model file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary classification threshold")
    args = parser.parse_args()

    # Verify model path
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Verify input image path
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found at: {image_path}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image at: {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    print(f"Original resolution: {orig_w}x{orig_h}")

    # 2. Preprocess
    print("Preprocessing image...")
    input_tensor = preprocess(image)

    # 3. Load ONNX model and run session
    print(f"Loading ONNX model: {model_path}")
    # Specify CPU provider explicitly for portability
    providers = ['CPUExecutionProvider']
    session = ort.InferenceSession(str(model_path), providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    print(f"Model Inputs: {[x.name for x in session.get_inputs()]} (shape: {session.get_inputs()[0].shape})")
    print(f"Model Outputs: {[x.name for x in session.get_outputs()]} (shape: {session.get_outputs()[0].shape})")

    # Run inference and measure time
    print("Running inference...")
    start_time = time.perf_counter()
    outputs = session.run([output_name], {input_name: input_tensor})
    inference_time = (time.perf_counter() - start_time) * 1000
    print(f"ONNX Inference Latency: {inference_time:.2f} ms")

    # 4. Post-process
    raw_output = outputs[0].squeeze() # Remove batch & channel dimensions -> (H, W)
    
    # Check if we need to apply sigmoid (if raw output contains values outside [0, 1])
    min_val, max_val = raw_output.min(), raw_output.max()
    print(f"Raw model output range: [{min_val:.4f}, {max_val:.4f}]")
    if min_val < 0.0 or max_val > 1.0:
        print("Output detected as raw logits. Applying Sigmoid...")
        prob_map = sigmoid(raw_output)
    else:
        print("Output detected as probabilities (Sigmoid already applied in graph).")
        prob_map = raw_output

    # Resize probability map back to original size
    prob_map_resized = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    
    # Threshold to get binary mask
    binary_mask = (prob_map_resized >= args.threshold).astype(np.uint8)

    # Calculate coverage
    wound_area_percentage = (binary_mask.sum() / binary_mask.size) * 100
    print(f"Detected Wound Coverage: {wound_area_percentage:.2f}%")

    # Draw overlay
    overlay = draw_overlay(image, binary_mask)

    # 5. Save results
    mask_out_path = output_dir / f"{image_path.stem}_onnx_mask.png"
    overlay_out_path = output_dir / f"{image_path.stem}_onnx_overlay.png"

    cv2.imwrite(str(mask_out_path), binary_mask * 255)
    cv2.imwrite(str(overlay_out_path), overlay)
    
    print(f"Results saved:")
    print(f"  - Mask: {mask_out_path}")
    print(f"  - Overlay: {overlay_out_path}")
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()
