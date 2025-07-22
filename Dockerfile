FROM python:3.13-slim

ARG DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY llmcord.py .

# Create a directory for configuration files
# if it does not already exist
RUN mkdir -p /config

CMD ["python", "llmcord.py", "--config", "/config/config.yaml"]
