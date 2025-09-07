# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      tini ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖再拷源码，利用缓存
COPY requirement.txt /app/requirement.txt
RUN pip install --no-cache-dir -r /app/requirement.txt

COPY . /app

EXPOSE 8000

ENV PORT=8000 \
    UVICORN_WORKERS=1 \
    UVICORN_LOG_LEVEL=info

ENTRYPOINT ["/usr/bin/tini","--"]
CMD ["sh","-lc","uvicorn app:app --host 0.0.0.0 --port ${PORT} --log-level ${UVICORN_LOG_LEVEL} --workers ${UVICORN_WORKERS}"]