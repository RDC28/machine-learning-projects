"""
Skin Cancer Detection Flask Web Application - Production Version

Lightweight production version using ONNX Runtime for inference.
No TensorFlow dependency required.
"""

import os
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# Try ONNX Runtime first (lightweight), fallback to TensorFlow if needed
try:
    import onnxruntime as ort
    USE_ONNX = True
    print("Using ONNX Runtime for inference (lightweight)")
except ImportError:
    USE_ONNX = False
    try:
        import tensorflow as tf
        print("Using TensorFlow for inference")
    except ImportError:
        raise ImportError("Please install either onnxruntime or tensorflow for inference")


# ================================
# Configuration
# ================================

app = Flask(__name__)
app.secret_key = 'skin-cancer-detection-prod-key-2026'

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Model configuration
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
H5_MODEL_PATH = os.path.join(MODEL_DIR, 'skin_cancer_model_best.h5')
ONNX_MODEL_PATH = os.path.join(MODEL_DIR, 'skin_cancer_model.onnx')
IMG_SIZE = 224

# Global model variable
model = None
ort_session = None


# ================================
# Helper Functions
# ================================

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model."""
    global model, ort_session, USE_ONNX
    
    # AUTO-DOWNLOAD: Ensure model exists before attempting to load
    # This handles the case where the model is not in the repo (Render deployment)
    if not os.path.exists(H5_MODEL_PATH) and not os.path.exists(ONNX_MODEL_PATH):
        print("Model files not found. Attempting to download...")
        try:
            from download_model import ensure_model_exists
            if ensure_model_exists():
                print("Download successful/verified.")
            else:
                print("Download failed.")
        except Exception as e:
            print(f"Error triggering model download: {e}")

    if USE_ONNX:
        if ort_session is None:
            if os.path.exists(ONNX_MODEL_PATH):
                print(f"Loading ONNX model from: {ONNX_MODEL_PATH}")
                ort_session = ort.InferenceSession(ONNX_MODEL_PATH)
                print("ONNX model loaded successfully!")
            else:
                print(f"ONNX model not found at {ONNX_MODEL_PATH}")
                print("Falling back to TensorFlow...")
                USE_ONNX = False
                return load_model()
        return ort_session
    else:
        if model is None:
            if os.path.exists(H5_MODEL_PATH):
                print(f"Loading H5 model from: {H5_MODEL_PATH}")
                model = tf.keras.models.load_model(H5_MODEL_PATH)
                print("Model loaded successfully!")
            else:
                print(f"Model not found at {H5_MODEL_PATH}")
        return model


def preprocess_image(image_path):
    """
    Preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    img = Image.open(image_path)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_image(image_path):
    """
    Make a prediction on a skin lesion image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        dict: Prediction result with class and confidence
    """
    global USE_ONNX, ort_session, model
    
    loaded_model = load_model()
    
    if loaded_model is None:
        return {
            'class': 'Unknown',
            'confidence': 0,
            'error': 'Model not loaded. Please check model files.'
        }
    
    img_array = preprocess_image(image_path)
    
    if USE_ONNX and ort_session is not None:
        # ONNX Runtime inference
        input_name = ort_session.get_inputs()[0].name
        output_name = ort_session.get_outputs()[0].name
        prediction = ort_session.run([output_name], {input_name: img_array})[0][0][0]
    else:
        # TensorFlow inference
        prediction = model.predict(img_array, verbose=0)[0][0]
    
    if prediction > 0.5:
        predicted_class = 'Malignant'
        confidence = prediction * 100
    else:
        predicted_class = 'Benign'
        confidence = (1 - prediction) * 100
    
    return {
        'class': predicted_class,
        'confidence': round(float(confidence), 2),
        'raw_prediction': float(prediction)
    }


# ================================
# Routes
# ================================

@app.route('/')
def index():
    """Home page route."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page route."""
    prediction = None
    uploaded_image = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'warning')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            file.save(filepath)
            uploaded_image = unique_filename
            
            try:
                prediction = predict_image(filepath)
                
                if 'error' in prediction:
                    flash(prediction['error'], 'danger')
                    prediction = None
                else:
                    flash('Image analyzed successfully!', 'success')
                    
            except Exception as e:
                flash(f'Error processing image: {str(e)}', 'danger')
                prediction = None
        else:
            flash('Invalid file type. Please upload an image (PNG, JPG, JPEG, GIF, BMP)', 'danger')
    
    return render_template('predict.html', prediction=prediction, uploaded_image=uploaded_image)


@app.route('/about')
def about():
    """About page route."""
    return render_template('about.html')


# ================================
# Error Handlers
# ================================

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash('File is too large. Maximum size is 16MB.', 'danger')
    return redirect(url_for('predict'))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 error."""
    return render_template('layout.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 error."""
    flash('An unexpected error occurred. Please try again.', 'danger')
    return redirect(url_for('index'))


# ================================
# Main Entry Point
# ================================

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Download model if not present
    from download_model import ensure_model_exists
    if not ensure_model_exists():
        print("\n⚠️  Warning: Model not available. Predictions will fail.")
        print("    Set HF_REPO, GDRIVE_FILE_ID, or MODEL_URL environment variable.")
    
    # Pre-load model
    load_model()
    
    print("\n" + "="*50)
    print(" Skin Cancer Detection - Production Server ")
    print("="*50)
    print(f"Server starting at: http://127.0.0.1:5000")
    print("="*50 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000)
