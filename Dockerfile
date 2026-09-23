# Multi-stage Dockerfile for KADAL Production Deployment
# Builds React frontend and bundles lightweight Python backend

# Stage 1: Build React/TypeScript frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit
COPY frontend/ ./
RUN npm run build

# Stage 2: Minimal runtime image
FROM python:3.11-slim
WORKDIR /app

# Install runtime dependencies for OpenCV & ONNX Runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install production Python dependencies (no PyTorch / CUDA overhead)
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-prod.txt

# Copy source directories
COPY backend/ ./backend/
COPY models/ ./models/
COPY run_app.py .

# Copy built frontend assets
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Download and lock production ONNX model weights (5.9 MB)
RUN python -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
os.makedirs('models', exist_ok=True); \
dl = hf_hub_download('Dinoman1221/sonarvision-yolov8-esi-v6', 'yolo_esi_v6_fp16.onnx', local_dir='models'); \
shutil.copy(dl, 'models/yolo_esi_fp16.onnx')"

ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
