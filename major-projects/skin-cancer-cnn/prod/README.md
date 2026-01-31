# Skin Cancer Detection - Production Deployment

Lightweight production version of the Skin Cancer Detection web app.

## 📦 What's Included

```
prod/
├── app.py              # Flask application (production)
├── requirements.txt    # Lightweight dependencies
├── download_model.py   # Auto-download model on startup
├── convert_to_onnx.py  # Model conversion script (optional)
├── templates/          # HTML templates
├── static/uploads/     # Uploaded images
└── model/              # Model files (downloaded on startup)
```

## 🚀 Deploy to Render.com

### Step 1: Upload Model to Hugging Face (Recommended)

1. Create account at [huggingface.co](https://huggingface.co)
2. Create new model repository: https://huggingface.co/new
3. Upload `skin_cancer_model_best.h5` to your repo
4. Note your repo name: `your-username/skin-cancer-model`

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com) → New → Web Service
2. Connect your GitHub repo
3. Configure:
   - **Root Directory**: `major-projects/skin-cancer-cnn/prod`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
4. Add Environment Variable:
   - `HF_REPO` = `your-username/skin-cancer-model`
5. Deploy!

## 🔧 Model Storage Options

### Option 1: Hugging Face Hub ⭐ (Recommended)

```bash
# Set environment variable on Render
HF_REPO=RDC28/skin-cancer-model
```

Best for: ML models, free, versioned, fast CDN

### Option 2: Google Drive

1. Upload model to Google Drive
2. Share → "Anyone with link"
3. Copy file ID from URL: `https://drive.google.com/file/d/FILE_ID/view`

```bash
# Set environment variable on Render
GDRIVE_FILE_ID=your_file_id_here
```

### Option 3: GitHub Releases

1. Go to your repo → Releases → Create new release
2. Attach `skin_cancer_model_best.h5` as asset
3. Copy direct download URL

```bash
# Set environment variable on Render
MODEL_URL=https://github.com/RDC28/machine-learning-projects/releases/download/v1.0/skin_cancer_model_best.h5
```

## 📊 Size Comparison

| Component | With TensorFlow | With ONNX Runtime |
|-----------|-----------------|-------------------|
| Dependencies | ~500 MB | ~50 MB |
| Cold Start | ~30 sec | ~5 sec |
| Memory | ~500 MB | ~200 MB |

## 🌐 Local Testing

```bash
cd prod

# Set model source (choose one)
export HF_REPO="RDC28/skin-cancer-model"
# OR
export GDRIVE_FILE_ID="your_file_id"
# OR
export MODEL_URL="https://..."

# Install and run
pip install -r requirements.txt
python app.py
```

## 📝 Render.yaml (Optional)

Create `render.yaml` in repo root for auto-deploy:

```yaml
services:
  - type: web
    name: skin-cancer-detection
    env: python
    rootDir: major-projects/skin-cancer-cnn/prod
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: HF_REPO
        value: RDC28/skin-cancer-model
```

## 👤 Author

Made by @RDC28
