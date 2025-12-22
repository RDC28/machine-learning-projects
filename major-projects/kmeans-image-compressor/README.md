# 🖼️ K-Means Image Compressor

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-blueviolet)](https://kmeans-image-compressor.onrender.com)

**Fast, Intelligent Image Compression using Unsupervised Learning**

A web-based image compression tool that reduces file size while preserving visual quality using **unsupervised machine learning**, **information theory**, and **perceptual color modeling**.

Built with **Python, Flask, and Scikit-learn**, this project demonstrates how classical ML and mathematics can be combined to solve real-world optimization problems — without deep learning or labeled data.

---

## 🚀 Key Features

* 🔍 **Unsupervised K-Means Compression**
  Clusters pixel colors to reduce palette size while retaining structure.

* 🧠 **Auto-K Mode (Entropy-Based)**
  Automatically selects optimal compression strength using **Shannon entropy**.

* ⚡ **Fast Processing (~2–5s per image)**
  Single-pass MiniBatchKMeans + adaptive heuristics — no brute-force loops.

* 🎨 **Perceptual Color Space (LAB)**
  Preserves human-visible details better than raw RGB clustering.

* 📉 **50–60% File Size Reduction**
  Achieved through a combination of spatial scaling, palette reduction, and adaptive encoding.

* 💾 **One-Click Download**
  Instantly save the compressed image.

* 🧩 **Clean Modular Architecture**
  ML logic isolated from Flask routes for maintainability.

---

## 🧠 How It Works (High-Level)

1. **Image Analysis (Fast & Cheap)**

   * Compute grayscale entropy
   * Measure spatial variance

2. **Automatic Parameter Selection**

   * Decide:

     * Downscaling factor (spatial redundancy)
     * Number of clusters (K)
     * JPEG quality level

3. **Unsupervised Learning**

   * Convert image to LAB color space
   * Normalize channel variance
   * Apply **MiniBatchKMeans** once

4. **Reconstruction & Encoding**

   * Rebuild image using learned centroids
   * Encode using optimized JPEG / PNG settings

All decisions are **mathematical and unsupervised** — no labels, no pretrained models.

---

## 🗂️ Project Structure

```
.
├── app.py                     # Flask application & routes
├── modules/
│   └── image_compression.py   # Core unsupervised compression logic
│
├── templates/
│   ├── layout.html            # Base layout + navbar
│   ├── index.html             # Homepage + demo examples
│   ├── compress.html          # Upload, compress, download UI
│   └── about.html             # Algorithm & theory explanation
│
├── static/
│   ├── demo/                  # Example compressed images
│   └── outputs/               # Overwritten compressed output
│
├── uploads/                   # Overwritten input image
└── README.md
```

---

## 🧪 Core Algorithm (Summary)

The compression engine (`modules/image_compression.py`) is responsible for:

* Entropy-driven parameter selection
* LAB color conversion
* Variance-aware clustering
* Fast MiniBatchKMeans execution
* Adaptive JPEG / PNG encoding

Source: 

---

## 🌐 Web Application

The Flask app (`app.py`) provides:

* `/` — Home & visual demos
* `/compress` — Upload, manual K or Auto-K, compression & download
* `/about` — Explanation of algorithm & theory

Source: 

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/kmeans-image-compressor.git
cd kmeans-image-compressor
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install flask numpy pillow scikit-learn scikit-image
```

### 4️⃣ Run the app

```bash
python app.py
```

Open your browser at:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🎛️ Manual vs Auto-K

| Mode     | Description                              |
| -------- | ---------------------------------------- |
| Manual K | User selects number of colors explicitly |
| Auto-K   | System estimates optimal K using entropy |

Auto-K remains **fully unsupervised** and deterministic.

---

## 📊 Example Results

| Setting          | Reduction | Notes                 |
| ---------------- | --------- | --------------------- |
| K=16, Auto-K     | ~50–60%   | Balanced quality      |
| K=8              | ~65%+     | Strong compression    |
| Auto-K + scaling | ~70%      | Best for large photos |

Actual results vary with image complexity.

---

## 🎯 Why This Project Matters

This project demonstrates:

* Practical use of **unsupervised learning**
* Understanding of **information theory**
* Performance-aware ML system design
* Clean separation of ML logic and web UI

It intentionally avoids deep learning to highlight **foundational ML and math** — the kind used in real compression systems.

---

## 🔮 Possible Extensions

* PSNR / SSIM quality metrics
* Batch compression mode
* GPU acceleration
* Target compression percentage slider
* Side-by-side visual comparison

---

## 👨‍💻 Author

**Rohit Chavda**

* GitHub: [https://github.com/RDC28](https://github.com/RDC28)
* LinkedIn: [https://www.linkedin.com/in/rchavda28](https://www.linkedin.com/in/rchavda28)

---

## 📜 License

This project is open-source and intended for educational and portfolio use.