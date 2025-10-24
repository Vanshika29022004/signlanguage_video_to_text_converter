"""Create and save a small dummy CNN to `model/sign_model.h5`.
This is used as a fallback if training is slow or dataset parsing is complicated.
"""
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# small model
num_classes = 5
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# create model dir at repo root
repo_root = os.path.dirname(os.path.dirname(__file__))
out_dir = os.path.join(repo_root, 'model')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'sign_model.h5')

# save untrained model weights (it's fine for a quick predict API test)
model.save(out_path)
print("✅ Dummy model saved at:", out_path)
