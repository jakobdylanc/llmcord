FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install dependencies first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY bot/ bot/
COPY llmcord.py .
COPY __init__.py .

# Copy config (can be overridden via volume)
COPY config-example.yaml config.yaml

CMD ["python", "llmcord.py"]
