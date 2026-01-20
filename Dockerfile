# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

# Set the working directory in the container
WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Pre-download the U2Net AI Model
RUN mkdir -p /root/.u2net \
    && curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -o /root/.u2net/u2net.onnx

# 3. Install Python dependencies
COPY requirements.txt .
ENV CUDA_VISIBLE_DEVICES=""
ENV FORCE_CPU=1

# Install PyTorch CPU version FIRST
RUN pip install --no-cache-dir \
    torch==2.9.1+cpu \
    torchvision==0.24.1+cpu \
    -f https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code
COPY . .

# 5. Create directory for static files
RUN mkdir -p static/uploads static/processed

# 6. Start Command (The Fix)
# We modify DATABASE_URL on the fly to support AsyncPG
CMD sh -c "export DATABASE_URL=\$(echo \$DATABASE_URL | sed 's/postgres:\/\//postgresql+asyncpg:\/\//') && alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port \${PORT}"