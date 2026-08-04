FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    NUMBA_THREADING_LAYER=tbb \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
ENV PYTHONPATH=/app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    curl \
    libtbb12 \
    libtbb-dev \
    gcc \
    g++ \
    cmake \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir --no-compile typing-extensions==4.12.2 jinja2==3.1.3
RUN git clone https://github.com/TencentARC/InstantMesh.git /opt/InstantMesh
RUN pip install --no-cache-dir --no-compile \
     torch==2.4.1 \
    torchvision==0.19.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --no-compile tbb
RUN pip install --no-cache-dir git+https://github.com/tatsy/torchmcubes.git
RUN git clone https://github.com/VAST-AI-Research/TripoSR.git /tmp/TripoSR && \
   mv /tmp/TripoSR/tsr /usr/local/lib/python3.10/site-packages/tsr
RUN pip install --no-cache-dir --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
RUN pip install --no-cache-dir --no-compile -c constraints.txt -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
RUN mkdir -p /root/.cache/torch/hub/checkpoints && \
    curl -L --retry 3 -o /root/.cache/torch/hub/checkpoints/big-lama.pt \
    https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('microsoft/Florence-2-large')"
RUN python -c "import easyocr; reader = easyocr.Reader(['en']); del reader"
# Pre-download YOLO watermark detector
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='qfisch/yolov8n-watermark-detection', filename='best.pt')"

# Pre-download 7 watermark segmenters
RUN for spec in logo centered_text overlay_text repeated_text tiny_corner line_pattern universal; do \
        python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='christophernavas/watermark-remover', filename='segmenter_${spec}.pth')"; \
    done
COPY . .
RUN mkdir -p app/static/uploads app/static/processed app/static/rooms
CMD sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"