# syntax=docker/dockerfile:1

# ---- Stage 1: build dependencies into a venv ----
FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: slim runtime image ----
FROM python:3.12-slim AS runtime

# Non-root user
RUN groupadd --gid 1000 fraudshield && \
    useradd --uid 1000 --gid fraudshield --shell /bin/bash --create-home fraudshield

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Only copy what's needed at runtime — not notebooks/, tests/, raw data, .git
COPY src/ ./src/
COPY data/features/ ./data/features/
COPY configs/ ./configs/
COPY deployment/healthcheck.sh ./deployment/healthcheck.sh
RUN chmod +x ./deployment/healthcheck.sh && chown -R fraudshield:fraudshield /app

USER fraudshield

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ./deployment/healthcheck.sh

CMD ["uvicorn", "src.fraudshield.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
