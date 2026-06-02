import onnxruntime as ort
import numpy as np

def inspect_onnx(path):
    print("=== ONNX Model ===")
    try:
        session = ort.InferenceSession(path)
        for i, input_meta in enumerate(session.get_inputs()):
            print(f"Input {i}: name={input_meta.name}, shape={input_meta.shape}, type={input_meta.type}")
        for i, output_meta in enumerate(session.get_outputs()):
            print(f"Output {i}: name={output_meta.name}, shape={output_meta.shape}, type={output_meta.type}")
    except Exception as e:
        print(f"Error inspecting ONNX model: {e}")

def inspect_tflite(path):
    print(f"\n=== TFLite Model: {path} ===")
    try:
        # Try importing tensorflow or tflite_runtime
        try:
            import tensorflow as tf
            interpreter = tf.lite.Interpreter(model_path=path)
        except ImportError:
            from tflite_runtime.interpreter import Interpreter
            interpreter = Interpreter(model_path=path)
            
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        for i, detail in enumerate(input_details):
            print(f"Input {i}: name={detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}, quantization={detail['quantization']}")
        for i, detail in enumerate(output_details):
            print(f"Output {i}: name={detail['name']}, shape={detail['shape']}, dtype={detail['dtype']}, quantization={detail['quantization']}")
    except Exception as e:
        print(f"Error inspecting TFLite model: {e}")

if __name__ == "__main__":
    inspect_onnx("checkpoints/wound_seg.onnx")
    inspect_tflite("checkpoints/wound_seg_float32.tflite")
    inspect_tflite("checkpoints/wound_seg_float16.tflite")
    inspect_tflite("checkpoints/wound_seg_full_integer_quant.tflite")