FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    NUMBA_THREADING_LAYER=tbb \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    libtbb12 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Do NOT download the ONNX model here – let the app download it at runtime
# (saves ~180 MB in the image)

COPY requirements.txt constraints.txt ./

# Install all Python dependencies in a single layer with exact versions
RUN pip install --no-cache-dir --no-compile \
    -c constraints.txt \
    -r requirements.txt

COPY . .

RUN mkdir -p static/uploads static/processed

CMD sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"