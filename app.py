from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
import os

app = Flask(__name__)

model = tf.keras.models.load_model("waste_classifier_mobilenet.keras")

class_names = [
    'battery',
    'biological',
    'cardboard',
    'clothes',
    'glass',
    'metal',
    'paper',
    'plastic',
    'shoes',
    'trash'
]

UPLOAD_FOLDER = "static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction) * 100)

        if confidence < 50:
            predicted_class = "Non-waste / Unknown object"

        return render_template(
            "index.html",
            prediction=predicted_class,
            confidence=round(confidence, 2),
            image_path=filepath
        )

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)