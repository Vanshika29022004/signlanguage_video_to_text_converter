import base64
import io
from PIL import Image
import requests

# create a simple RGB image 64x64
img = Image.new('RGB', (64,64), color=(255,0,0))
buffer = io.BytesIO()
img.save(buffer, format='PNG')
img_bytes = buffer.getvalue()
img_b64 = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')

payload = {'image': img_b64}
resp = requests.post('http://127.0.0.1:5000/predict', json=payload)
print('status:', resp.status_code)
print('json:', resp.json())
