from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)
CORS(app)

# Locate model file robustly:
# 1) backend/models/sign_model.h5
# 2) repo-root model/sign_model.h5
base_dir = os.path.dirname(os.path.abspath(__file__))
candidate_paths = [
    os.path.join(base_dir, 'models', 'sign_model.h5'),
    os.path.join(os.path.dirname(base_dir), 'model', 'sign_model.h5')
]

model_path = None
for p in candidate_paths:
    if os.path.exists(p):
        model_path = p
        break

if model_path is None:
    raise FileNotFoundError(f"Could not find model file. Checked: {candidate_paths}")

model = tf.keras.models.load_model(model_path)

def preprocess_image(image_b64):
    img_data = base64.b64decode(image_b64.split(',')[1])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_b64 = data['image']
    img = preprocess_image(img_b64)
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    return jsonify({'prediction': str(class_index)})

if __name__ == '__main__':
    app.run(debug=True)
