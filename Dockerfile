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
# This saves ~180 MB in the image
# RUN mkdir -p /root/.u2net \
#     && curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -o /root/.u2net/u2net.onnx

COPY requirements.txt .

# Install typing-extensions and jinja2 first
RUN pip install --no-cache-dir --no-compile typing-extensions==4.12.2 jinja2==3.1.3

# Install torch (CPU version) – use EXISTING versions
RUN pip install --no-cache-dir --no-compile \
    torch==2.2.1 \
    torchvision==0.17.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install tbb (though already installed via apt)
RUN pip install --no-cache-dir --no-compile tbb

# Install all other dependencies
RUN pip install --no-cache-dir --no-compile -r requirements.txt

COPY . .

RUN mkdir -p static/uploads static/processed

CMD sh -c "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"