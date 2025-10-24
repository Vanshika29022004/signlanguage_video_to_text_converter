"""Small helper to inspect the downloaded dataset structure and count image files.
Run from repository root using the venv python.
"""
import os
from collections import Counter

DATA_PATH = r"C:\Users\USER\.cache\kagglehub\datasets\drblack00\isl-csltr-indian-sign-language-dataset\versions\1"
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def summarize(path, max_show=20):
    print('Scanning', path)
    if not os.path.exists(path):
        print('Path does not exist')
        return

    top_children = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    print('Top-level dirs:', len(top_children))
    for name in top_children[:max_show]:
        full = os.path.join(path, name)
        # count images under this child (recursively)
        img_count = 0
        subdirs = set()
        for root, dirs, files in os.walk(full):
            subdirs.update(dirs)
            for f in files:
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS:
                    img_count += 1
        print(f'- {name}: subdirs={len(subdirs)}, images={img_count}')

    # global counts
    total_images = 0
    total_dirs = 0
    counter = Counter()
    for root, dirs, files in os.walk(path):
        total_dirs += len(dirs)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            counter[ext] += 1
            if ext in IMAGE_EXTS:
                total_images += 1

    print('\nTotal dirs:', total_dirs)
    print('Total image files:', total_images)
    print('File type counts (top 10):')
    for k, v in counter.most_common(10):
        print(' ', k, v)

if __name__ == '__main__':
    summarize(DATA_PATH)
