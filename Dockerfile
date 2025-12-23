# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Default port if Render doesn't set one
    PORT=10000

# Set the working directory in the container
WORKDIR /app

# 1. Install system dependencies
# Added 'curl' so we can download the AI model manually
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. CRITICAL FIX: Pre-download the U2Net AI Model
# This prevents the app from downloading 176MB on startup (which causes timeouts)
RUN mkdir -p /root/.u2net \
    && curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -o /root/.u2net/u2net.onnx

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code
COPY . .

# 5. Create directory for static files so permission errors don't occur
RUN mkdir -p static/uploads static/processed

# 6. Command to run the application
# We use shell format here so ${PORT} is expanded correctly
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}