# ── Stage 1: 依赖构建层 ──────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# 先复制项目元数据和源码，由 pyproject.toml 统一解析依赖。
COPY pyproject.toml README.md ./
COPY multi_agent_research/ ./multi_agent_research/
RUN python -m pip install --no-cache-dir --prefix=/install .


# ── Stage 2: 运行层（精简） ──────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# 从 builder 层复制已安装的包，避免跨 Python 版本复制 site-packages。
COPY --from=builder /install /usr/local

# 运行时系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目代码
COPY multi_agent_research/ ./multi_agent_research/
COPY config.yaml ./config.yaml

# 不复制 .env！通过环境变量注入（安全原则）
# 本项目不包含知识库和本地检索模型，通过 knowledge-service HTTP 接口检索

# 健康检查（对应你的 /api/health 接口）
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

EXPOSE 8000

# 生产环境去掉 --reload
CMD ["uvicorn", "multi_agent_research.api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--loop", "multi_agent_research.api.event_loop:selector_loop_factory", \
     "--workers", "1"]
