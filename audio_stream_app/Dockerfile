FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories with appropriate permissions
RUN mkdir -p audio_chunks output_audio/finished temp_client_audio && \
    chmod -R 755 audio_chunks output_audio temp_client_audio

# Expose ports (App Platform will map these)
EXPOSE 5006
EXPOSE 5005/udp

# Create entrypoint script
RUN echo '#!/bin/bash\n\
# Start Gunicorn\n\
gunicorn --worker-class eventlet \\\n\
         --workers 1 \\\n\
         --bind 0.0.0.0:$PORT \\\n\
         --log-level info \\\n\
         --access-logfile - \\\n\
         --error-logfile - \\\n\
         server:flask_app &\n\
\n\
# Start the main server\n\
python server.py\n'\
> /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
