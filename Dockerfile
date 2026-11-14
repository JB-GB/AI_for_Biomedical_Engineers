# RAUSNet inference API for research servers (CPU only, ONNX Runtime, no PyTorch).
#
#   docker build -t rausnet-api .
#   docker run --rm -p 127.0.0.1:8000:8000 -v "$PWD/models:/models:ro" rausnet-api
#
# The model registry (registry.json + <version>/model.onnx + model.json) is NOT
# baked into the image: mount it read-only at /models. See docs/04_deployment.md.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Only the serving extra: fastapi, uvicorn, onnxruntime (CPU), python-multipart.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[serve]" \
    && rm -rf /app/src /root/.cache

# Unprivileged runtime user; /models is a read-only mount point.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin rausnet \
    && mkdir -p /models \
    && chown rausnet:rausnet /models
USER rausnet
WORKDIR /home/rausnet

VOLUME ["/models"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

# Inside the container the API listens on all interfaces of the container network;
# the host-side exposure is decided by the port mapping (127.0.0.1 by default in compose).
ENTRYPOINT ["rausnet-api"]
CMD ["--models-dir", "/models", "--host", "0.0.0.0", "--port", "8000"]
