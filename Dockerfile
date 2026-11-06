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

# Claude Code CLI：用已登入的訂閱帳號產生 / 修正 caption（設定的「VLM 引擎」選 Claude）。登入資訊由 compose 掛載主機的
# ~/.claude；Codex 的執行檔和登入資訊都在掛載進來的 ~/.codex 裡。安裝失敗（例如 build 時沒有網路）不影響其他功能，
# Claude 引擎只會顯示為未安裝。CLI 以登入資料夾擁有者的身分執行，HOME 是 /cli-home
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && (curl -fsSL https://claude.ai/install.sh | bash \
     && cp -L /root/.local/bin/claude /usr/local/bin/claude \
     && rm -rf /root/.local/share/claude /root/.local/bin/claude \
     || echo "claude install skipped") \
 && mkdir -p /cli-home && chmod 1777 /cli-home
ENV CLI_HOME=/cli-home \
    DISABLE_AUTOUPDATER=1

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
