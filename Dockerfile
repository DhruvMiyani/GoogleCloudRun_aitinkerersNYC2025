# Multi-stage build for efficient Gemma model serving
FROM nvidia/cuda:12.1.0-devel-ubuntu20.04 as builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Production stage
FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/models/joe /app/models/lex

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the FastAPI server
CMD ["python3", "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]