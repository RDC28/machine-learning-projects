"""
Convert Keras H5 model to ONNX format for lightweight production deployment.

Run this script ONCE before deploying to convert the model.
After conversion, you can deploy without TensorFlow dependency.

Requirements (only needed for conversion):
    pip install tensorflow tf2onnx onnx
"""

import os
import sys

def convert_h5_to_onnx():
    try:
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        print("Error: Please install conversion dependencies:")
        print("  pip install tensorflow tf2onnx onnx")
        sys.exit(1)
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    H5_PATH = os.path.join(BASE_DIR, 'model', 'skin_cancer_model_best.h5')
    ONNX_PATH = os.path.join(BASE_DIR, 'model', 'skin_cancer_model.onnx')
    
    if not os.path.exists(H5_PATH):
        print(f"Error: Model not found at {H5_PATH}")
        sys.exit(1)
    
    print(f"Loading model from: {H5_PATH}")
    model = tf.keras.models.load_model(H5_PATH)
    
    print("Converting to ONNX format...")
    
    # Get input signature
    input_signature = [tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name='input')]
    
    # Convert to ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=input_signature, opset=13)
    
    # Save ONNX model
    import onnx
    onnx.save(onnx_model, ONNX_PATH)
    
    print(f"\n✅ Model converted successfully!")
    print(f"   ONNX model saved to: {ONNX_PATH}")
    
    # Show size comparison
    h5_size = os.path.getsize(H5_PATH) / (1024 * 1024)
    onnx_size = os.path.getsize(ONNX_PATH) / (1024 * 1024)
    print(f"\n📊 Size comparison:")
    print(f"   H5 model:   {h5_size:.2f} MB")
    print(f"   ONNX model: {onnx_size:.2f} MB")


if __name__ == '__main__':
    convert_h5_to_onnx()
