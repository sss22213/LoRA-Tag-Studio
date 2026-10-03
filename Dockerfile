# syntax=docker/dockerfile:1
FROM python:3.12-slim

# CPU：onnxruntime；GPU：onnxruntime-gpu[cuda,cudnn]（CUDA/cuDNN 由 pip 安裝，需 NVIDIA 驅動 >= 580）
ARG ORT_PACKAGE=onnxruntime==1.30.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    DATA_DIR=/data \
    IMPORT_DIR=/import \
    WEB_DIR=/app/web

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt "${ORT_PACKAGE}"

COPY app ./app
COPY web ./web
RUN mkdir -p /data /import /models

EXPOSE 7870
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7870/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7870", "--proxy-headers", "--forwarded-allow-ips", "*"]
