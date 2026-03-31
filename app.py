from flask import Flask, request, render_template
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# ✅ Smart model loading
if os.path.exists("best.pt"):
    print("✅ Loading custom model (best.pt)")
    model = YOLO("best.pt")
else:
    print("⚠️ best.pt not found → using yolov8n.pt")
    model = YOLO("yolov8n.pt")

# Folders
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


@app.route("/", methods=["GET", "POST"])
def index():
    uploaded_image = None
    output_image = None
    prediction = None

    if request.method == "POST":

        if "image" not in request.files:
            return "No file uploaded"

        file = request.files["image"]

        if file.filename == "":
            return "No file selected"

        try:
            # Save upload
            upload_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            upload_path = upload_path.replace("\\", "/")
            file.save(upload_path)

            # Predict
            results = model(upload_path)

            result_img = results[0].plot()

            output_path = os.path.join(app.config["OUTPUT_FOLDER"], file.filename)
            output_path = output_path.replace("\\", "/")
            cv2.imwrite(output_path, result_img)

            # Prediction text
            if len(results[0].boxes) > 0:
                prediction = f"{len(results[0].boxes)} Objects Detected 🌿"
            else:
                prediction = "No Objects Detected ❌"

            uploaded_image = "/" + upload_path
            output_image = "/" + output_path

        except Exception as e:
            return f"Error: {str(e)}"

    return render_template(
        "index.html",
        uploaded_image=uploaded_image,
        output_image=output_image,
        prediction=prediction
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
