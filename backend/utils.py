import cv2
import numpy as np

def preprocess_frame(frame, size=(64, 64)):
    img = cv2.resize(frame, size)
    img = img.astype('float32') / 255.0
    return img
