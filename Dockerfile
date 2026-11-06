# syntax=docker/dockerfile:1
# H2-3: CPU-only образ для локального запуска HippoRAG.
# LM Studio не входит в образ: он работает на хосте, см. docker-compose.yml.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/app/cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

# CPU-колёса torch: не тянем CUDA в образ.
ARG TORCH_VERSION=2.5.1+cpu
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}"

COPY pyproject.toml README_RU.md .env.example constraints.txt ./
COPY src ./src
COPY scripts ./scripts
COPY main.py ./

# -c constraints.txt: пиним проверенный стек (transformers/sentence-transformers),
# иначе резолвер берёт последние версии, несовместимые с torch 2.5.1 (см. файл).
RUN python -m pip install . -c constraints.txt

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/outputs /app/data /app/temp_uploads /app/cache \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/ui/config >/dev/null || exit 1

CMD ["python", "scripts/start.py", "--host", "0.0.0.0", "--port", "8000"]
