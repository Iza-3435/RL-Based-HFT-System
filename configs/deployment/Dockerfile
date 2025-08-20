# Multi-stage Dockerfile for HFT Trading System
# Optimized for production deployment with security and performance

# Build stage - compile C++ components and prepare dependencies
FROM python:3.11-slim as builder

# Set build environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create build directory
WORKDIR /build

# Copy only requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies in a separate layer
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code for C++ compilation
COPY src/ ./src/
COPY Makefile ./

# Build C++ components for optimal performance
RUN make build-cpp || echo "C++ build skipped - components may not be available"

# Production stage - minimal runtime image
FROM python:3.11-slim as production

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TZ=America/New_York
ENV HFT_MODE=production
ENV LOG_LEVEL=INFO

# Create non-root user for security
RUN groupadd -r hft && useradd -r -g hft hft

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    netcat-openbsd \
    iputils-ping \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create build directory  
RUN mkdir -p ./build

# Copy application code
COPY --chown=hft:hft src/ ./src/
COPY --chown=hft:hft configs/ ./configs/
COPY --chown=hft:hft examples/ ./examples/
COPY --chown=hft:hft tests/ ./tests/
COPY --chown=hft:hft run.py ./
COPY --chown=hft:hft requirements.txt ./

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/reports /app/data /app/outputs && \
    chown -R hft:hft /app

# Switch to non-root user
USER hft

# Add health check to ensure system is functioning
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from src.python.data.advanced_technical_indicators import AdvancedTechnicalEngine; AdvancedTechnicalEngine()" || exit 1

# Expose port for monitoring (if needed)
EXPOSE 8080

# Set default command - can be overridden at runtime
CMD ["python", "run.py"]