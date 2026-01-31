# 🔬 Skin Cancer Detection using CNN

AI-powered web application for skin cancer classification using Convolutional Neural Networks.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Overview

This web application uses a trained CNN model to classify skin lesion images as **Benign** or **Malignant**. The model is trained on the ISIC (International Skin Imaging Collaboration) dataset.

> ⚠️ **Disclaimer**: This tool is for educational purposes only and should NOT replace professional medical advice.

## ✨ Features

- 🧠 Deep Learning model trained on ISIC dataset
- ⚡ Instant predictions with confidence scores
- 🖼️ Drag-and-drop image upload
- 📱 Responsive modern UI design
- 🔒 Privacy-first (images not stored permanently)
- 🚀 Lightweight production deployment option

## 🏗️ CNN Architecture

```
Input: 224 × 224 × 3
    ↓
Conv2D (32 filters) + ReLU + MaxPool
    ↓
Conv2D (64 filters) + ReLU + MaxPool
    ↓
Conv2D (128 filters) + ReLU + MaxPool
    ↓
Flatten → Dense (512) + Dropout (0.5)
    ↓
Output: Sigmoid (Binary Classification)
```

## 📊 Dataset

- **Training**: 2,637 images (1,440 benign, 1,197 malignant)
- **Testing**: 660 images (360 benign, 300 malignant)

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/RDC28/machine-learning-projects.git
cd machine-learning-projects/major-projects/skin-cancer-cnn

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
flask run --debug
```

Visit `http://127.0.0.1:5000` in your browser.

### Training the Model (Optional)

```bash
python -m module.train
```

## 📁 Project Structure

```
skin-cancer-cnn/
├── app.py                  # Flask application
├── requirements.txt        # Dependencies
├── module/
│   └── train.py           # Training script
├── model/
│   └── *.h5               # Trained model (not in git)
├── templates/
│   ├── layout.html        # Base template
│   ├── index.html         # Home page
│   ├── predict.html       # Prediction page
│   └── about.html         # About page
├── static/
│   └── uploads/           # Uploaded images
├── prod/                   # Production deployment
│   ├── app.py             # Lightweight Flask app
│   ├── requirements.txt   # Minimal dependencies
│   └── convert_to_onnx.py # Model converter
└── data/                   # Training data (not in git)
```

## 🌐 Production Deployment

For lightweight deployment (~50MB instead of ~500MB):

```bash
cd prod

# Convert model to ONNX (one-time)
pip install tensorflow tf2onnx onnx
python convert_to_onnx.py

# Deploy with ONNX Runtime
pip install -r requirements.txt
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML Framework**: TensorFlow / Keras
- **Inference**: ONNX Runtime (production)
- **Frontend**: HTML5, CSS3, Bootstrap 5
- **Image Processing**: Pillow, NumPy

## 📸 Screenshots

### Home Page
Modern dark theme with animated background and feature highlights.

### Prediction Page  
Drag-and-drop image upload with instant AI analysis and confidence scores.

### About Page
Detailed information about the model architecture and dataset.

## 👤 Author

**@RDC28**

- GitHub: [@RDC28](https://github.com/RDC28)
- LinkedIn: [rchavda28](https://www.linkedin.com/in/rchavda28)

## 📝 License

This project is for educational purposes. See [LICENSE](LICENSE) for details.

---

⭐ Star this repo if you found it helpful!
