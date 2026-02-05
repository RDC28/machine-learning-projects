import os
import io
import base64
from flask import Flask, render_template, request
from modules.image_compression import compress_image

app = Flask(__name__)

# -----------------------------
# Config
# -----------------------------
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


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
        
        # Read input into memory
        input_buffer = io.BytesIO()
        file.save(input_buffer)
        input_buffer.seek(0)

        # Prepare output buffer
        output_buffer = io.BytesIO()

        # Run intelligent compression (using buffers instead of paths)
        stats = compress_image(
            input_file=input_buffer,
            output_file=output_buffer,
            k=k,
            auto_k=auto_k
        )

        # Convert output to base64 for display
        output_buffer.seek(0)
        base64_data = base64.b64encode(output_buffer.getvalue()).decode("utf-8")
        
        # Determine MIME type
        mime_type = "image/jpeg" if ext in ("jpg", "jpeg") else "image/png"
        output_image_data = f"data:{mime_type};base64,{base64_data}"

        return render_template(
            "compress.html",
            active="compress",
            output_image=output_image_data,
            ext=ext,
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
