# RadarX API Server Dockerfile
# Multi-stage build for production-ready image

FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 radarx

# Create app directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/radarx/.local

# Copy application code
COPY --chown=radarx:radarx . .

# Switch to non-root user
USER radarx

# Add local bin to PATH
ENV PATH=/home/radarx/.local/bin:$PATH

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["python", "-m", "radarx.api.server"]
