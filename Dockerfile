# Multilingual AI Voice Assistant
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY logs/ ./logs/

# Create necessary directories
RUN mkdir -p data/scraped data/vector_db data/audio logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/healthz || exit 1

# Default command (can be overridden)
# Option 1: API + Frontend (current)
CMD ["python", "-m", "src.api.main"]

# Option 2: Streamlit only (simpler)
# CMD ["streamlit", "run", "src/frontend/streamlit_app.py", "--server.address=0.0.0.0"]

# Option 3: Full system (both)
# CMD ["python", "app.py", "run"]
