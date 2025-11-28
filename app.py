# app.py

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

from model_utils import SignLanguageModel

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "super-secret-key"  # for flash messages

# Load model once at startup
model = SignLanguageModel()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "video" not in request.files:
            flash("No video file part")
            return redirect(request.url)

        file = request.files["video"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            try:
                label, confidence = model.predict_video(save_path)
            except Exception as e:
                flash(f"Error processing video: {str(e)}")
                return redirect(request.url)

            return render_template(
                "result.html",
                predicted_label=label,
                confidence=round(confidence * 100, 2),
                filename=filename,
            )
        else:
            flash("Invalid file type. Please upload a video file (mp4, avi, mov, mkv, webm).")
            return redirect(request.url)

    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    # Debug for development; use gunicorn in production
    app.run(host="0.0.0.0", port=5000, debug=True)
