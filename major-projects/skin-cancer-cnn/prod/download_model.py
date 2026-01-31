"""
Model Download Utility for Production Deployment

Downloads the trained model from external storage on startup.
Supports: Hugging Face Hub, Google Drive, GitHub Releases, or direct URL.
"""

import os
import sys
import hashlib

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'skin_cancer_model_best.h5')

# ========================================
# CONFIGURATION - Choose ONE option below
# ========================================

# Option 1: Hugging Face Hub (Recommended)
# Upload your model to: https://huggingface.co/new
# Then set: HF_REPO = "your-username/skin-cancer-model"
HF_REPO = os.environ.get('HF_REPO', 'rdc28/skin-canc-classi')  # Default to your repo
HF_FILENAME = "skin_cancer_model_best.h5"

# Option 2: Google Drive
# Share file -> "Anyone with link" -> Copy file ID from URL
# URL format: https://drive.google.com/file/d/FILE_ID/view
GDRIVE_FILE_ID = os.environ.get('GDRIVE_FILE_ID', None)

# Option 3: Direct URL (GitHub Releases, S3, etc.)
MODEL_URL = os.environ.get('MODEL_URL', None)

# ========================================


def download_from_huggingface():
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download
        print(f"📥 Downloading model from Hugging Face: {HF_REPO}")
        
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILENAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False
        )
        print(f"✅ Model downloaded to: {downloaded_path}")
        return True
    except Exception as e:
        print(f"❌ Hugging Face download failed: {e}")
        return False


def download_from_gdrive():
    """Download model from Google Drive."""
    try:
        import gdown
        print(f"📥 Downloading model from Google Drive...")
        
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        
        print(f"✅ Model downloaded to: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ Google Drive download failed: {e}")
        return False


def download_from_url():
    """Download model from direct URL."""
    try:
        import requests
        print(f"📥 Downloading model from URL...")
        
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    progress = (downloaded / total_size) * 100
                    print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
        
        print(f"\n✅ Model downloaded to: {MODEL_PATH}")
        return True
    except Exception as e:
        print(f"❌ URL download failed: {e}")
        return False


def ensure_model_exists():
    """Ensure the model file exists, downloading if necessary."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"✅ Model already exists: {MODEL_PATH} ({size_mb:.1f} MB)")
        return True
    
    print("🔍 Model not found locally. Attempting download...")
    
    # Try download methods in order of preference
    if HF_REPO:
        if download_from_huggingface():
            return True
    
    if GDRIVE_FILE_ID:
        if download_from_gdrive():
            return True
    
    if MODEL_URL:
        if download_from_url():
            return True
    
    print("\n" + "="*50)
    print("❌ ERROR: Could not download model!")
    print("="*50)
    print("\nPlease set ONE of these environment variables:")
    print("  - HF_REPO: Hugging Face repository (e.g., 'RDC28/skin-cancer-model')")
    print("  - GDRIVE_FILE_ID: Google Drive file ID")
    print("  - MODEL_URL: Direct download URL")
    print("\nOr manually place the model at:")
    print(f"  {MODEL_PATH}")
    print("="*50)
    
    return False


if __name__ == '__main__':
    success = ensure_model_exists()
    sys.exit(0 if success else 1)
