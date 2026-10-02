# FastAPI 完全解析：从基础到这段代码

---

## 一、FastAPI 是什么，为什么用它？

FastAPI 是一个现代 Python Web 框架，核心特点：

- **基于类型提示**：用 Python 的 `type hint` 自动完成参数校验、文档生成
- **异步优先**：原生支持 `async/await`，天生适合 IO 密集型任务（调用 LLM、数据库）
- **自动文档**：代码写完，Swagger UI 和 Redoc 文档自动生成，Agent 系统调试神器
- **性能极高**：基于 Starlette + Pydantic，性能接近 Node.js

对 Agent 开发来说，FastAPI 的意义是：**让你的 Agent 成为一个可被调用的服务**，前端、其他服务、甚至其他 Agent 都能通过 HTTP 请求驱动它。

---

## 二、FastAPI 核心语法地图

### 2.1 最小可运行示例

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
async def hello():
    return {"message": "world"}
```

用 `uvicorn main:app --reload` 启动，访问 `http://localhost:8000/hello` 就能看到响应。`uvicorn` 是 ASGI 服务器，FastAPI 的运行容器。

---

### 2.2 三种参数来源（非常重要）

```python
from fastapi import FastAPI, Query, Path
from pydantic import BaseModel

app = FastAPI()

# ① 路径参数：URL 里的变量
@app.get("/users/{user_id}")
async def get_user(user_id: int):  # FastAPI 自动把字符串转成 int，类型错误自动 422
    return {"user_id": user_id}

# ② 查询参数：?key=value
@app.get("/search")
async def search(
    q: str = Query(..., min_length=2, max_length=100),  # ... 表示必填
    page: int = Query(default=1, ge=1),                 # ge=1 表示 >=1
):
    return {"q": q, "page": page}

# ③ 请求体：POST/PUT 时用 Pydantic 模型接收 JSON
class ResearchRequest(BaseModel):
    question: str
    max_iterations: int = 5

@app.post("/research")
async def create_research(body: ResearchRequest):
    return {"received": body.question}
```

**关键点**：FastAPI 看参数名和类型，自动判断它是路径参数还是查询参数还是请求体。`Query()`、`Path()` 是显式声明 + 增加校验规则。

---

### 2.3 响应类型

```python
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse

# 默认返回 dict，FastAPI 自动序列化为 JSON
@app.get("/json")
async def json_endpoint():
    return {"key": "value"}

# 自定义响应
@app.get("/custom")
async def custom():
    return JSONResponse(content={"key": "value"}, status_code=201)

# 流式响应（SSE / 文件下载）
@app.get("/stream")
async def stream():
    async def generator():
        for i in range(10):
            yield f"data: {i}\n\n"
    return StreamingResponse(generator(), media_type="text/event-stream")
```

---

### 2.4 中间件（Middleware）

中间件是**拦截所有请求和响应的钩子**，在路由处理之前/之后执行：

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 允许哪些域名跨域
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

CORS（跨域资源共享）是浏览器安全机制。前端页面在 `localhost:3000`，后端在 `localhost:8000`，浏览器会拦截请求，加了 CORS 中间件才能放行。

---

### 2.5 Lifespan（生命周期钩子）⭐

这是代码里的核心结构，新版 FastAPI 推荐的写法：

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # === 服务启动时执行 ===
    print("服务正在启动，初始化资源...")
    await load_ml_models()
    yield                        # ← yield 之前是 startup，之后是 shutdown
    # === 服务关闭时执行 ===
    print("服务正在关闭，释放资源...")
    await cleanup()

app = FastAPI(lifespan=lifespan)
```

`yield` 是这里的关键词：它把一个函数变成"上下文管理器"，`yield` 之前的代码在**启动**时跑，`yield` 之后的代码在**关闭**时跑。这比老版本的 `@app.on_event("startup")` 更优雅，因为启动和关闭逻辑在同一个函数里，资源的申请和释放写在一起，一目了然。

---

## 三、SSE（Server-Sent Events）深度解析

SSE 是这段代码最核心的技术，也是 Agent 系统里展示实时进度的标配方案。

### 3.1 SSE vs WebSocket vs 轮询

| 方案 | 方向 | 复杂度 | 适用场景 |
|------|------|--------|---------|
| 轮询 | 客户端不停问 | 简单但低效 | 更新不频繁的状态 |
| WebSocket | 双向实时 | 复杂 | 聊天、协同编辑 |
| SSE | 服务器单向推送 | 简单 | 进度流、日志流、LLM 流式输出 |

**Agent 系统天然适合 SSE**：研究流程是单向的——服务器跑着，不断把进度推给前端，前端只需要展示。

### 3.2 SSE 协议格式

SSE 其实就是一个永不关闭的 HTTP 响应，内容遵循固定格式：

```
event: search_complete\n
data: {"count": 10}\n
\n                          ← 空行表示这条消息结束
```

完整格式：
```
id: 消息ID（可选，用于断线重连）
event: 事件名（可选，默认是 message）
data: JSON 字符串或文本
retry: 重连间隔毫秒（可选）

（空行）
```

### 3.3 SSE 协议对应代码中的函数

```python
def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
```

这个函数把事件名和数据字典，格式化成符合 W3C SSE 规范的字符串。`ensure_ascii=False` 是为了让中文不被转成 `\uXXXX` 转义序列。

前端这样消费：

```javascript
const es = new EventSource('/api/research/stream?question=AI最新进展');

es.addEventListener('search_complete', (e) => {
    const data = JSON.parse(e.data);
    console.log(`找到 ${data.new_count} 条结果`);
});

es.addEventListener('done', (e) => {
    const data = JSON.parse(e.data);
    console.log('研究完成:', data.report);
    es.close();
});

es.addEventListener('error', (e) => {
    console.error('出错了:', JSON.parse(e.data).message);
});
```

---

## 四、逐行解析这段代码

### 4.1 整体结构

```
server.py
├── lifespan()          服务启动/关闭钩子
├── app = FastAPI()     应用实例
├── CORS 中间件          处理跨域
├── _sse()              SSE 消息格式化工具
└── 路由
    ├── /api/health     健康检查
    ├── /api/research/stream   核心 SSE 流
    └── /               Demo HTML 页面
```

### 4.2 lifespan 预热逻辑

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Server] 正在预热 RAG 模型（BM25 + Reranker）...")
    await asyncio.gather(
        asyncio.to_thread(_get_app),   # ① 构建 LangGraph 图
        warmup_models(),               # ② 加载 Reranker 模型
    )
    logger.info("[Server] 预热完成，服务就绪 ✅")
    yield
```

**`asyncio.gather()`**：并发执行多个协程，相当于"同时做两件事"，比串行执行更快。

**`asyncio.to_thread()`**：把同步函数（`_get_app` 是同步的）放进线程池执行，避免阻塞事件循环。这是 async 代码调用 sync 代码的标准做法。

为什么要预热？BM25 索引构建和 Reranker 模型加载都很耗时（可能几秒到十几秒），放在启动时做，避免第一个用户请求时卡住。这是生产系统的标准做法。

### 4.3 核心路由：SSE 研究流

```python
@app.get("/api/research/stream", summary="研究进度 SSE 流")
async def research_stream(
    question: str = Query(..., description="研究问题", min_length=5, max_length=500),
    thread_id: str = Query(default="", description="会话 ID，空则自动生成"),
):
    tid = thread_id.strip() or f"sess_{uuid.uuid4().hex[:8]}"
```

- `Query(...)` 中的 `...` 表示必填参数，FastAPI 会自动验证，不传就返回 422 错误
- `uuid.uuid4().hex[:8]` 生成随机 8 位十六进制字符串作为会话 ID，用于 LangGraph 的线程隔离（不同用户的研究状态不互相干扰）

```python
    async def generate() -> AsyncGenerator[str, None]:
        try:
            async for event_name, event_data in astream_research(question, tid):
                yield _sse(event_name, event_data)
        except Exception as exc:
            logger.error("[SSE] 研究任务异常：%s", exc, exc_info=True)
            yield _sse("error", {"message": str(exc), "type": type(exc).__name__})
```

`generate()` 是一个**异步生成器函数**：
- 它从 `astream_research`（你的 LangGraph 多 Agent 系统）不断拿事件
- 每拿到一个，立刻格式化成 SSE 字符串 `yield` 出去
- 出错时不让连接直接断掉，而是推一个 `error` 事件，让前端能优雅处理

```python
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",   # 关闭 Nginx 缓冲
            "Connection":        "keep-alive",
        },
    )
```

`StreamingResponse` 包裹异步生成器，FastAPI 会持续从生成器取内容推给客户端。

三个 Header 都有实际意义：
- `no-cache`：禁止代理服务器缓存这个响应
- `X-Accel-Buffering: no`：告诉 Nginx 不要缓冲，立刻转发给客户端（否则 Nginx 会攒够一定量再发，实时性全毁）
- `keep-alive`：保持 TCP 连接不断开

### 4.4 健康检查端点

```python
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "multi-agent-research"}
```

这是运维标配。Kubernetes、Docker Compose、负载均衡器都会定期 GET 这个接口，返回 200 表示服务存活，否则自动重启或摘流量。别小看这两行，生产环境缺了它很麻烦。

---

## 五、Agent 开发视角：这套架构的意义

你正在做的事情，是一个典型的 **Agent-as-a-Service** 架构：

```
前端 / 其他服务
      │
      │ GET /api/research/stream?question=...
      ▼
  FastAPI (server.py)          ← 你现在看的这个文件
      │
      │ astream_research()
      ▼
  LangGraph Multi-Agent        ← 你的 Agent 核心
  ┌──────────────────┐
  │ Supervisor Agent  │
  │ Search Agent      │
  │ Analyst Agent     │
  │ Writer Agent      │
  └──────────────────┘
      │
      │ SSE 事件流
      ▼
  前端实时展示进度
```

这套架构的优点：
1. **解耦**：Agent 逻辑和 HTTP 服务分离，各自可以独立迭代
2. **可观测**：每个 Agent 的决策都通过 SSE 事件暴露出来，调试方便
3. **用户体验好**：研究可能要几十秒，SSE 让用户实时看到进展，不会以为卡死了

面试时提到这套架构，是很加分的亮点。

---

## 六、几个值得深挖的知识点

**ASGI vs WSGI**：老框架 Flask/Django 用 WSGI（同步），FastAPI 用 ASGI（异步）。ASGI 能在一个线程里处理上千个并发连接，对 LLM 调用这种高延迟 IO 场景优势巨大。

**Pydantic v2**：FastAPI 底层用 Pydantic 做数据验证，现在是 v2 版本，用 Rust 实现，比 v1 快 5-50 倍。你在代码里看到的 `Query(..., min_length=5)` 校验规则，底层都是 Pydantic 在跑。

**`async for` 和异步生成器**：`astream_research` 返回的是异步生成器，`async for` 才能消费它。普通的 `for` 循环消费异步生成器会报错。这是 Python 异步编程里容易搞混的点，值得专门练习。

**`asynccontextmanager`**：这个装饰器来自 `contextlib`，把一个带 `yield` 的 async 函数变成异步上下文管理器，是 Python 里非常优雅的模式，在 Agent 框架里到处都是这个用法。