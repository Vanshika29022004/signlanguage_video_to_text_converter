# model_utils.py

import os
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_HEIGHT = 64
IMG_WIDTH = 64

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "sign_model.h5")
LABEL_MAP_PATH = os.path.join(MODEL_DIR, "label_map.pkl")


class SignLanguageModel:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run train_model.py first.")
        if not os.path.exists(LABEL_MAP_PATH):
            raise FileNotFoundError(f"Label map not found at {LABEL_MAP_PATH}. Run train_model.py first.")

        self.model = load_model(MODEL_PATH)

        with open(LABEL_MAP_PATH, "rb") as f:
            self.index_to_class = pickle.load(f)

    def preprocess_frame(self, frame):
        frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frame_resized = frame_resized.astype("float32") / 255.0
        frame_resized = np.expand_dims(frame_resized, axis=0)
        return frame_resized

    def predict_frame(self, frame):
        processed = self.preprocess_frame(frame)
        preds = self.model.predict(processed, verbose=0)[0]
        class_index = int(np.argmax(preds))
        class_label = self.index_to_class[class_index]
        confidence = float(preds[class_index])
        return class_label, confidence

    def predict_video(self, video_path, frame_skip=5, min_frames=10):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Error opening video file.")

        predictions = []

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce processing
            if frame_count % frame_skip == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                label, conf = self.predict_frame(frame_rgb)
                predictions.append(label)
            frame_count += 1

        cap.release()

        if len(predictions) == 0:
            return None, 0.0

        # Majority voting
        unique, counts = np.unique(predictions, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        total = np.sum(counts)
        majority_count = np.max(counts)
        confidence = majority_count / total

        return majority_label, confidence
