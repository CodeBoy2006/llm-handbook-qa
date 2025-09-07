# ---- Base ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 基础依赖（尽量精简；chromadb 提供预编译轮子，一般无需编译工具链）
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 预装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝代码
COPY build_index_chroma.py app.py ./
COPY data ./data
COPY .env ./.env
COPY frontend ./.frontend

# Chroma 持久化目录
VOLUME ["/app/chroma"]

EXPOSE 8000

# 使用 tini 作为 init（处理信号更稳）
ENTRYPOINT ["/usr/bin/tini", "--"]

# 默认启动 Web 服务；构建索引用 run/exec 覆写命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]