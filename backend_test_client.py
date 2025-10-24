import base64
import io
import os
from PIL import Image

# ensure working directory is repo root so relative model path resolves
repo_root = os.path.dirname(__file__)
os.chdir(repo_root)

import importlib
mod = importlib.import_module('backend.app')
flask_app = getattr(mod, 'app')

# Create a simple image
img = Image.new('RGB', (64,64), color=(0,0,255))
buffer = io.BytesIO()
img.save(buffer, format='PNG')
img_bytes = buffer.getvalue()
img_b64 = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')

with flask_app.test_client() as client:
    resp = client.post('/predict', json={'image': img_b64})
    print('status', resp.status_code)
    print('data', resp.get_json())
