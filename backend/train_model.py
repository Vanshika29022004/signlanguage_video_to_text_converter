"""Train a small image-based CNN and save to `model/sign_model.h5`.

Notes:
- The original script used TimeDistributed+LSTM but the preprocessing returns single images.
- This simplified trainer is faster and compatible with `app.py` which expects single-image input.
"""
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from preprocess import load_data
import numpy as np


# load dataset (default path used by original script)
data_path = r"C:\Users\USER\.cache\kagglehub\datasets\drblack00\isl-csltr-indian-sign-language-dataset\versions\1"
(X_train, X_test, y_train, y_test), label_dict = load_data(data_path)

# Configuration via environment variables for flexibility
MAX_SAMPLES = int(os.environ.get('MAX_SAMPLES', '2000'))
EPOCHS = int(os.environ.get('EPOCHS', '3'))

# Cap training/validation set sizes for quicker runs
if X_train.shape[0] > MAX_SAMPLES:
    idx = np.random.choice(X_train.shape[0], MAX_SAMPLES, replace=False)
    X_train = X_train[idx]
    y_train = y_train[idx]

if X_test.shape[0] > 500:
    idx = np.random.choice(X_test.shape[0], 500, replace=False)
    X_test = X_test[idx]
    y_test = y_test[idx]

# simple CNN
num_classes = len(label_dict)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(f"Training with MAX_SAMPLES={MAX_SAMPLES}, EPOCHS={EPOCHS}, num_classes={num_classes}")
model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))

# Save model to both repo-root `model/` and backend `models/` so the app can find it
repo_root = os.path.dirname(os.path.dirname(__file__))
out_dir1 = os.path.join(repo_root, 'model')
out_dir2 = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(out_dir1, exist_ok=True)
os.makedirs(out_dir2, exist_ok=True)
out_path1 = os.path.join(out_dir1, 'sign_model.h5')
out_path2 = os.path.join(out_dir2, 'sign_model.h5')
model.save(out_path1)
model.save(out_path2)
print("✅ Model trained and saved at:", out_path1)
print("✅ Model also saved at:", out_path2)
