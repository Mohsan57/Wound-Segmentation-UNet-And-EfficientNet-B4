"""
benchmark.py — Latency benchmarking utility for PyTorch, ONNX, and TFLite models.
---------------------------------------------------------------------------------
Measures average inference latency (in ms) across CPU, GPU, and quantized models.
Usage:
    python benchmark.py --model checkpoints/best_model.pth
"""

import argparse
import time
import os
import numpy as np
import torch
from pathlib import Path

from config import Config
from model import build_model


def benchmark_pytorch(model, device, image_size: int, runs: int = 200, warmup: int = 50):
    print(f"\n--- Benchmarking PyTorch ({device}) ---")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
        
    start_time = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy_input)
            
    if device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    total_time = (end_time - start_time) * 1000.0  # in ms
    avg_latency = total_time / runs
    print(f"Device: {device}")
    print(f"Input size: {image_size}x{image_size}")
    print(f"Warmup runs: {warmup}, Timed runs: {runs}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {1000.0 / avg_latency:.1f} FPS")
    return avg_latency


def benchmark_onnx(onnx_path: str, runs: int = 200, warmup: int = 50):
    print(f"\n--- Benchmarking ONNX Runtime (CPU) ---")
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found at {onnx_path}. Skipping.")
        return None
        
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed. Skip benchmarking ONNX.")
        return None
        
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # Extract resolution
    # Input shape might contain dynamic batch size (e.g. ['batch_size', 3, 512, 512] or [1, 3, 512, 512])
    h = input_shape[2] if isinstance(input_shape[2], int) else 512
    w = input_shape[3] if isinstance(input_shape[3], int) else 512
    
    dummy_input = np.random.randn(1, 3, h, w).astype(np.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})
        
    start_time = time.perf_counter()
    for _ in range(runs):
        _ = session.run(None, {input_name: dummy_input})
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) * 1000.0) / runs
    print(f"Model path: {onnx_path}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {1000.0 / avg_latency:.1f} FPS")
    return avg_latency


def benchmark_tflite(tflite_path: str, runs: int = 200, warmup: int = 50):
    print(f"\n--- Benchmarking TFLite (CPU Interpreter) ---")
    if not os.path.exists(tflite_path):
        print(f"TFLite model not found at {tflite_path}. Skipping.")
        return None
        
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            print("tensorflow or tflite_runtime is not installed. Skip benchmarking TFLite.")
            return None
            
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = input_details[0]['shape']
    input_type = input_details[0]['dtype']
    
    # Generate dummy input matching dtype
    if input_type == np.uint8 or input_type == np.int8:
        dummy_input = np.random.randint(0, 255, size=input_shape, dtype=input_type)
    else:
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
    # Warmup
    for _ in range(warmup):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        
    start_time = time.perf_counter()
    for _ in range(runs):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    end_time = time.perf_counter()
    
    avg_latency = ((end_time - start_time) * 1000.0) / runs
    print(f"Model path: {tflite_path}")
    print(f"Input shape: {dummy_input.shape} ({input_type})")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {1000.0 / avg_latency:.1f} FPS")
    return avg_latency


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Latency Benchmark")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth", help="PyTorch model checkpoint")
    parser.add_argument("--onnx", type=str, default="exports/wound_seg.onnx", help="Exported ONNX model path")
    parser.add_argument("--tflite", type=str, default="exports/wound_seg.tflite", help="Exported TFLite model path")
    parser.add_argument("--runs", type=int, default=200, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=50, help="Number of warmup runs")
    args = parser.parse_args()
    
    cfg = Config()
    
    # PyTorch Model Latency
    if os.path.exists(args.model):
        device = torch.device(cfg.device)
        model = build_model(
            architecture=cfg.architecture,
            encoder_name=cfg.encoder_name,
            encoder_weights=None,
            in_channels=cfg.image_channels,
            num_classes=cfg.num_classes,
            activation=None
        ).to(device)
        
        # Load weights
        try:
            try:
                ckpt = torch.load(args.model, map_location=device)
            except Exception:
                ckpt = torch.load(args.model, map_location=device, weights_only=False)
            state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
            model.load_state_dict(state_dict)
            model.eval()
            
            # Benchmark on CPU
            benchmark_pytorch(model.to("cpu"), torch.device("cpu"), cfg.image_size, args.runs, args.warmup)
            
            # Benchmark on GPU (if available)
            if torch.cuda.is_available():
                benchmark_pytorch(model.to("cuda"), torch.device("cuda"), cfg.image_size, args.runs, args.warmup)
        except Exception as e:
            print(f"Error loading PyTorch model for benchmarking: {e}")
    else:
        print(f"PyTorch model not found at {args.model}.")
        
    # ONNX Model Latency
    benchmark_onnx(args.onnx, args.runs, args.warmup)
    
    # TFLite Model Latency
    benchmark_tflite(args.tflite, args.runs, args.warmup)
