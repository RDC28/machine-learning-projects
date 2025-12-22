import os
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from skimage import color
from skimage.measure import shannon_entropy

SUPPORTED_FORMATS = ("jpg", "jpeg", "png")


# -------------------------------------------------
# Fast unsupervised parameter decision
# -------------------------------------------------
def decide_params(img_np, manual_k=None):
    """
    Decide scale, K, and JPEG quality using cheap image statistics.
    Fully unsupervised and fast.
    """
    gray = np.mean(img_np, axis=2).astype(np.uint8)
    entropy = shannon_entropy(gray)

    # Spatial complexity proxy
    variance = np.var(gray)

    # --- Decide scale ---
    if entropy > 6.0 or variance > 5000:
        scale = 0.75
    else:
        scale = 0.6

    # --- Decide K ---
    if manual_k is not None:
        k = manual_k
    else:
        if entropy < 4.5:
            k = 8
        elif entropy < 5.5:
            k = 12
        else:
            k = 16

    # --- Decide JPEG quality ---
    if entropy < 4.5:
        quality = 35
    elif entropy < 5.5:
        quality = 40
    else:
        quality = 45

    return {
        "entropy": float(entropy),
        "scale": scale,
        "k": int(k),
        "quality": int(quality)
    }


# -------------------------------------------------
# Main fast compression function
# -------------------------------------------------
def compress_image(
    input_path,
    output_path,
    k=16,
    auto_k=False,
    max_iter=50,
    random_state=42
):
    """
    Fast intelligent unsupervised image compression.

    Guarantees:
    - Single K-Means run
    - Single encode
    - 5s-ish runtime for typical images
    """

    # -----------------------------
    # Load image
    # -----------------------------
    img = Image.open(input_path).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]

    # -----------------------------
    # Decide parameters (FAST)
    # -----------------------------
    params = decide_params(
        img_np,
        manual_k=None if auto_k else k
    )

    scale = params["scale"]
    selected_k = params["k"]
    jpeg_quality = params["quality"]

    # -----------------------------
    # Resize (spatial compression)
    # -----------------------------
    if scale != 1.0:
        img = img.resize(
            (int(W * scale), int(H * scale)),
            Image.BILINEAR
        )
        img_np = np.array(img)

    # -----------------------------
    # RGB -> LAB
    # -----------------------------
    lab = color.rgb2lab(img_np)
    pixels = lab.reshape(-1, 3)

    # -----------------------------
    # Variance normalization
    # -----------------------------
    std = np.std(pixels, axis=0) + 1e-6
    pixels_norm = pixels / std

    # -----------------------------
    # Single MiniBatchKMeans
    # -----------------------------
    kmeans = MiniBatchKMeans(
        n_clusters=selected_k,
        batch_size=4096,
        max_iter=max_iter,
        random_state=random_state
    )

    labels = kmeans.fit_predict(pixels_norm)

    # -----------------------------
    # Reconstruct
    # -----------------------------
    recon_lab = (kmeans.cluster_centers_[labels] * std).reshape(lab.shape)
    recon_rgb = color.lab2rgb(recon_lab)
    recon_rgb = np.clip(recon_rgb * 255, 0, 255).astype("uint8")

    recon_img = Image.fromarray(recon_rgb)

    # -----------------------------
    # Upscale back (if resized)
    # -----------------------------
    if scale != 1.0:
        recon_img = recon_img.resize((W, H), Image.BILINEAR)

    # -----------------------------
    # Encode
    # -----------------------------
    ext = output_path.rsplit(".", 1)[1].lower()

    if ext in ("jpg", "jpeg"):
        recon_img.save(
            output_path,
            format="JPEG",
            quality=jpeg_quality,
            optimize=True,
            progressive=True
        )
    else:
        recon_img = recon_img.convert(
            "P",
            palette=Image.ADAPTIVE,
            colors=selected_k
        )
        recon_img.save(output_path, optimize=True)

    # -----------------------------
    # Metadata
    # -----------------------------
    original_size = os.path.getsize(input_path)
    compressed_size = os.path.getsize(output_path)

    return {
        "k_used": selected_k,
        "auto_k": auto_k,
        "entropy": round(params["entropy"], 3),
        "scale": scale,
        "jpeg_quality": jpeg_quality if ext in ("jpg", "jpeg") else None,
        "original_size_kb": round(original_size / 1024, 2),
        "compressed_size_kb": round(compressed_size / 1024, 2),
        "reduction_percent": round(
            (1 - compressed_size / original_size) * 100, 2
        ),
        "encoder": "JPEG (lossy)" if ext in ("jpg", "jpeg") else "PNG (palette)"
    }
