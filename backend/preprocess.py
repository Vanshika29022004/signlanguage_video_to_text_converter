import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def _find_label_dir(root):
    """Try to locate the directory that directly contains label subfolders.

    The dataset may be nested (e.g. root/ISL_CSLRT_Corpus/<labels>), so
    walk down single-child directories until we find a directory whose
    immediate children look like label folders.
    """
    cur = root
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    for _ in range(6):  # avoid infinite loops
        try:
            children = [d for d in os.listdir(cur) if os.path.isdir(os.path.join(cur, d))]
        except FileNotFoundError:
            return None

        # If any child directory contains image files, assume current is label root
        for c in children:
            child_path = os.path.join(cur, c)
            try:
                for fname in os.listdir(child_path):
                    if os.path.isfile(os.path.join(child_path, fname)) and os.path.splitext(fname)[1].lower() in IMAGE_EXTS:
                        return cur
            except Exception:
                continue

        # if there's exactly one directory, descend into it and continue searching
        if len(children) == 1:
            cur = os.path.join(cur, children[0])
            continue
        # otherwise assume current is the label dir (best effort)
        return cur
    return cur


def load_data(data_dir, img_size=(64, 64)):
    X, y = [], []

    label_root = _find_label_dir(data_dir)
    if label_root is None:
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    labels = [d for d in os.listdir(label_root) if os.path.isdir(os.path.join(label_root, d))]
    labels = sorted(labels)
    label_dict = {label: i for i, label in enumerate(labels)}

    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    for label in labels:
        folder_path = os.path.join(label_root, label)
        if not os.path.isdir(folder_path):
            continue
        # walk nested directories to find image files (dataset contains nested folders)
        for root, dirs, files in os.walk(folder_path):
            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in IMAGE_EXTS:
                    continue
                img_path = os.path.join(root, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, img_size)
                    X.append(img)
                    y.append(label_dict[label])

    X = np.array(X, dtype='float32') / 255.0
    y = np.array(y)
    if X.shape[0] == 0:
        raise ValueError(f'No image files found under {label_root}')
    return train_test_split(X, y, test_size=0.2, random_state=42), label_dict
