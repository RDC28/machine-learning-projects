import os
from flask import Flask, render_template, request
from modules.image_compression import compress_image

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "static/outputs"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------
# Helpers
# -----------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html", active="home")


@app.route("/compress", methods=["GET", "POST"])
def compress():
    if request.method == "POST":
        file = request.files.get("image")
        k = int(request.form.get("k", 16))
        auto_k = request.form.get("auto_k") == "on"

        if not file or not allowed_file(file.filename):
            return "Invalid file format", 400

        ext = file.filename.rsplit(".", 1)[1].lower()

        input_path = os.path.join(UPLOAD_DIR, f"input.{ext}")
        output_path = os.path.join(OUTPUT_DIR, f"output.{ext}")

        # Overwrite input image
        file.save(input_path)

        # Run intelligent compression
        stats = compress_image(
            input_path=input_path,
            output_path=output_path,
            k=k,
            auto_k=auto_k
        )

        return render_template(
            "compress.html",
            active="compress",
            output_image=f"outputs/output.{ext}",
            stats=stats,
            k=stats["k_used"],
            auto_k=auto_k
        )

    return render_template(
        "compress.html",
        active="compress",
        k=16,
        auto_k=False
    )


@app.route("/about")
def about():
    return render_template("about.html", active="about")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
