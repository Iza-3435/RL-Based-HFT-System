FROM python:3.11-slim

LABEL maintainer="HFT Trading Systems"
LABEL description="Production HFT Network Optimizer"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    iputils-ping \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 hftuser && \
    chown -R hftuser:hftuser /app && \
    mkdir -p /app/logs /app/data && \
    chown -R hftuser:hftuser /app/logs /app/data

USER hftuser

ENV PYTHONPATH=/app

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

CMD ["python", "main.py", "--mode", "production", "--log-level", "normal"]
