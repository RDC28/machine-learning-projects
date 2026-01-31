# Skin Cancer Detection - Production Deployment

Lightweight production version of the Skin Cancer Detection web app.

## 📦 What's Included

```
prod/
├── app.py              # Flask application (production)
├── requirements.txt    # Lightweight dependencies
├── convert_to_onnx.py  # Model conversion script (run once)
├── templates/          # HTML templates
├── static/
│   └── uploads/        # Uploaded images
└── model/
    └── skin_cancer_model_best.h5   # Trained model
```

## 🚀 Deployment Options

### Option 1: ONNX Runtime (Recommended - Lightweight)

**Step 1:** Convert model to ONNX format (run once, requires TensorFlow):
```bash
pip install tensorflow tf2onnx onnx
python convert_to_onnx.py
```

**Step 2:** Deploy with ONNX Runtime only:
```bash
pip install -r requirements.txt
python app.py
```

**Benefits:**
- ONNX Runtime: ~50MB vs TensorFlow: ~500MB+
- Faster cold starts
- Lower memory usage
- No TensorFlow dependency in production

### Option 2: TensorFlow Fallback

If you skip ONNX conversion, the app will use TensorFlow:
```bash
pip install flask pillow numpy tensorflow-cpu gunicorn
python app.py
```

## 🌐 Production Server

For production, use Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 📊 Size Comparison

| Dependency      | Size     |
|-----------------|----------|
| ONNX Runtime    | ~50 MB   |
| TensorFlow CPU  | ~200 MB  |
| Full TensorFlow | ~500+ MB |

## ⚠️ Notes

- The `uploads/` folder stores temporary uploaded images
- Consider adding a cleanup job for old uploads
- For HTTPS, use a reverse proxy like Nginx

## 👤 Author

Made by @RDC28
