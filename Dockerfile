mFROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p audio_chunks output_audio

# Expose ports
EXPOSE 5005/udp
EXPOSE 5006

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the server


CMD ["python", "server.py"]
