#!/usr/bin/env bash
set -e

echo "=== [1/3] Building Frontend Production Assets ==="
cd frontend
npm install
npm run build
cd ..

echo "=== [2/3] Installing Production Python Dependencies ==="
pip install --upgrade pip
pip install -r requirements-prod.txt

echo "=== [3/3] Fetching Production ONNX Model Weights (5.9 MB) ==="
python -c "
import os, shutil
from huggingface_hub import hf_hub_download
models_dir = 'models'
os.makedirs(models_dir, exist_ok=True)
dest = os.path.join(models_dir, 'yolo_esi_fp16.onnx')
if not os.path.exists(dest):
    print('Downloading production weights from Hugging Face...')
    dl = hf_hub_download('Dinoman1221/sonarvision-yolov8-esi-v6', 'yolo_esi_v6_fp16.onnx', local_dir=models_dir)
    shutil.copy(dl, dest)
    print('Model weights ready at:', dest)
else:
    print('Model weights already present at:', dest)
"

echo "=== Build Complete Successfully ==="
