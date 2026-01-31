"""
Skin Cancer Detection Flask Web Application

This is the main Flask application that serves the web interface
for skin cancer prediction using a trained CNN model.
"""

import os
import uuid
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf


# ================================
# Configuration
# ================================

app = Flask(__name__)
app.secret_key = 'skin-cancer-detection-secret-key-2026'

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'skin_cancer_model_best.h5')
IMG_SIZE = 224
CLASSES = ['Benign', 'Malignant']

# Global model variable
model = None


# ================================
# Helper Functions
# ================================

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_model():
    """Load the trained model."""
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully!")
        else:
            print(f"Warning: Model not found at {MODEL_PATH}")
            print("Please train the model first using: python -m module.train_model")
    return model


def preprocess_image(image_path):
    """
    Preprocess an image for prediction.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array
    """
    # Load and resize image
    img = Image.open(image_path)
    
    # Convert to RGB if necessary (handles PNG with alpha, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Resize to model input size
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
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
    # Load model if not already loaded
    loaded_model = load_model()
    
    if loaded_model is None:
        return {
            'class': 'Unknown',
            'confidence': 0,
            'error': 'Model not loaded. Please train the model first.'
        }
    
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Make prediction
    prediction = loaded_model.predict(img_array, verbose=0)[0][0]
    
    # Interpret prediction
    # Model outputs probability of malignant (class 1)
    # If prediction > 0.5, it's malignant; otherwise, it's benign
    if prediction > 0.5:
        predicted_class = 'Malignant'
        confidence = prediction * 100
    else:
        predicted_class = 'Benign'
        confidence = (1 - prediction) * 100
    
    return {
        'class': predicted_class,
        'confidence': round(confidence, 2),
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
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'warning')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            flash('No file selected', 'warning')
            return redirect(request.url)
        
        # Check if file is allowed
        if file and allowed_file(file.filename):
            # Create upload folder if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Generate unique filename
            ext = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            
            # Save the file
            file.save(filepath)
            uploaded_image = unique_filename
            
            try:
                # Make prediction
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
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'), exist_ok=True)
    
    # Try to load the model at startup
    load_model()
    
    # Run the Flask app
    print("\n" + "="*50)
    print(" Skin Cancer Detection Web App ")
    print("="*50)
    print(f"Server starting at: http://127.0.0.1:5000")
    print("Press Ctrl+C to stop the server")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
