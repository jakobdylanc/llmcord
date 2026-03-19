FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install FFmpeg for voice playback and build deps for PyNaCl
RUN apt-get update && apt-get install -y ffmpeg libffi-dev libsodium-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY bot/ bot/
COPY llmcord.py .
COPY __init__.py .
COPY web/ web/

# Copy config (can be overridden via volume)
COPY config-example.yaml config.yaml

# Environment variables for portal
# PORT - set to 8080 by default for web portal
ENV PORT=8080

CMD ["python", "llmcord.py"]
