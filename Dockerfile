# =======================================
# FHP Detection API — Docker Image
# Optimized for Render free-tier deployment
# =======================================

FROM python:3.11-slim

# System dependencies for OpenCV + MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps — CPU-only PyTorch to keep image small
COPY requirements-api.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-api.txt

# Copy source code
COPY src/ src/
COPY api/ api/
COPY config.yaml .

# Copy model weights (exported model + MediaPipe task file)
COPY models/exported/ models/exported/
COPY models/pose_landmarker_lite.task models/pose_landmarker_lite.task

# Copy web frontend (served by FastAPI)
COPY web/ web/

# Render provides PORT env var at runtime (typically 10000)
ENV PORT=8000
EXPOSE ${PORT}

# Health check using runtime PORT
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\",\"8000\")}/health')" || exit 1

# Shell form so $PORT is expanded at runtime by Render
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 120
