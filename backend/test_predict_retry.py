import base64
import io
from PIL import Image
import requests
import time

# create a simple RGB image 64x64
img = Image.new('RGB', (64,64), color=(0,255,0))
buffer = io.BytesIO()
img.save(buffer, format='PNG')
img_bytes = buffer.getvalue()
img_b64 = 'data:image/png;base64,' + base64.b64encode(img_bytes).decode('utf-8')

payload = {'image': img_b64}

url = 'http://127.0.0.1:5000/predict'
for i in range(12):
    try:
        resp = requests.post(url, json=payload, timeout=5)
        print('status:', resp.status_code)
        print('json:', resp.json())
        break
    except Exception as e:
        print('attempt', i+1, 'failed:', str(e))
        time.sleep(1)
else:
    print('All attempts failed')
