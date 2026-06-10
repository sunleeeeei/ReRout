# LLM 智能路由方案设计文档

## 目录

- [第一章 需求与选型分析](#第一章-需求与选型分析)
- [第二章 架构设计](#第二章-架构设计)
- [第三章 测试环境部署（Docker Compose）](#第三章-测试环境部署docker-compose)
- [第四章 生产环境部署（Kubernetes）](#第四章-生产环境部署kubernetes)
- [附录](#附录)

---

## 第一章 需求与选型分析

### 1.1 业务背景

公司内网部署了多个不同参数规模的 LLM 模型，需要根据用户请求的复杂度和任务类型，自动将请求分发到最合适的模型，以降低资源消耗、保障响应质量、控制推理成本。

当前模型资源：

| 模型 | 参数量 | 适用场景 |
|------|--------|---------|
| Qwen2.5-7B | 7B | 简单问答、摘要、闲聊 |
| Qwen2.5-Coder-7B | 7B | 代码生成 |
| Qwen2.5-32B | 32B | 文档撰写、头脑风暴 |
| DeepSeek-R1-671B | 671B | 复杂推理、多步分析、架构设计 |

### 1.2 约束条件

| 约束 | 说明 |
|------|------|
| 纯内网环境 | 无法访问外部云服务和公网 API |
| 已有 Higress 网关 | 模型通过 Higress 统一对外提供 OpenAI 兼容 API |
| 不同模型不同 API Key | Higress 为每类模型分配独立的认证密钥 |
| 路由维度为模型选择 | 非 GPU 实例级负载均衡 |
| 低延迟 | 路由层延迟不应显著增加整体响应时间 |
| 生产就绪 | 方案需具备监控、降级、安全等生产特性 |

### 1.3 候选方案对比

经过筛选，以下 4 个方案进入详细评估：

| 维度 | NVIDIA LLM Router v1 | NVIDIA LLM Router v2 | LLMRouter (ulab-uiuc) | RouteLLM (lm-sys) | **ReRout + Sidecar** |
|------|---------------------|---------------------|----------------------|-------------------|---------------------|
| 分类模型 | BERT（预训练） | Qwen 1.7B（LLM） | 16+ 算法可选 | BERT/MF/SW（预训练） | 规则 / LLM Sidecar |
| GPU 要求 | ✅ 必须（Triton） | ✅ 必须（16GB） | ❌ CPU 可跑 | BERT: CPU / MF: 需 OpenAI API | ❌ CPU 可跑 |
| 分类延迟 | ~5ms（GPU） | ~90ms（GPU） | 5ms-2s | BERT: ~20ms | 规则: <1ms / Sidecar: ~200ms |
| 代理转发 | ✅ 内置 Rust 代理 | ❌ 仅分类，不转发 | ⚠️ 需 OpenClaw 包装 | ✅ 通过 LiteLLM | ✅ Controller 内置 |
| 安全护栏 | ❌ | ❌ | ❌ | ❌ | ✅ PII 检测 |
| 监控体系 | ❌ | ❌ | ❌ | ❌ | ✅ Prometheus + Grafana |
| 降级策略 | ❌ | ❌ | ❌ | ❌ | ✅ 多层降级 |
| A/B 测试 | ❌ | ❌ | ❌ | ❌ | ✅ 内置 |
| 反馈闭环 | ❌ | ❌ | ⚠️ 部分 | ❌ | ✅ Bandit /reward |
| Higress 对接 | ⚠️ 需写代码 | ⚠️ 需写转发层 | ⚠️ 需写代码 | ⚠️ 需配置 LiteLLM | ✅ config.yaml 纯配置 |
| 内网部署 | ⚠️ NGC 镜像问题 | ✅ Docker | ✅ pip install | ⚠️ MF/SW 需 OpenAI | ✅ Docker / K8s |
| 项目成熟度 | v1 稳定 / v2 Experimental | v2 Experimental | Alpha | 学术框架 | 微服务架构完善 |
| 中文支持 | ⚠️ 英文 BERT | ✅ Qwen | ⚠️ 英文 benchmark 为主 | ⚠️ 英文 Arena 数据 | ✅ LLM Sidecar 调中文模型 |
| 路由粒度 | 13 类意图 | 5 类意图 | 取决于算法 | 强/弱 二选一 | **6 意图 × 3 复杂度 → 任意模型** |

### 1.4 淘汰原因

| 方案 | 淘汰核心原因 |
|------|------------|
| **NVIDIA v1** | 强制要求 NVIDIA GPU（Triton 架构）；BERT 预训练数据以英文为主，中文分类需微调；内网首次部署需从 NGC 拉取镜像 |
| **NVIDIA v2** | Experimental 状态（官方标注）；不代理转发请求，需额外写转发层；Qwen 1.7B 独占 16GB GPU |
| **LLMRouter** | Alpha 状态，无 SLA；大部分算法需要训练数据；生产 API 服务（OpenClaw）需自行包装；监控/安全/降级全缺 |
| **RouteLLM（原版）** | 路由粒度为强/弱二选一，不支持多模型路由；MF/SW 路由器依赖 OpenAI embedding API（内网不可用）；BERT 路由器可用但粒度不够；生产功能全缺 |

### 1.5 RouteLLM 预训练模型的适用性分析

RouteLLM 确实提供了 3 个预训练路由模型（BERT、MF、Causal LLM），但存在以下限制：

- **MF（矩阵分解）和 SW Ranking**：依赖 OpenAI `text-embedding-3-small` API，该模型为闭源、仅通过 OpenAI API 提供，内网无法使用
- **BERT**：不依赖外部 API，纯本地推理，但输出为"强模型胜率"（0-1 浮点数），路由粒度只有"用大模型 or 小模型"二选一，无法支持 4+ 个模型的多级路由
- **泛化性**：预训练数据基于英文 Chatbot Arena 对话，中文场景效果未经验证

结论：RouteLLM 的预训练模型无法满足内网多模型路由需求。

### 1.6 最终选型

**ReRout + LLM Sidecar**，理由：

1. **零 GPU 依赖**：全链路 CPU 可运行，GPU 资源全部留给业务模型
2. **生产功能完整**：监控、安全护栏、降级策略、A/B 测试、反馈闭环开箱即用
3. **Higress 纯配置对接**：修改 `config.yaml` 即可，无需写适配代码
4. **路由粒度精细**：6 种意图 × 3 级复杂度，可路由到任意数量的模型
5. **模式灵活切换**：规则模式（<1ms）和 LLM Sidecar 模式（~200ms）通过配置切换
6. **多 API Key 支持**：不同模型使用不同密钥，通过多 backend 配置实现
7. **微服务架构**：天然适配 K8s，每个服务可独立升级

---

## 第二章 架构设计

### 2.1 整体架构

```
                              ┌─────────────────────────────────────────┐
                              │            用户 / 业务系统              │
                              └────────────────┬────────────────────────┘
                                               │
                              ┌────────────────▼────────────────────────┐
                              │        ReRout Controller (:8084)        │
                              │   OpenAI 兼容 API / 路由编排 / 代理转发   │
                              └────────────────┬────────────────────────┘
                                               │
                           ┌───────────────────┼───────────────────┐
                           │                   │                   │
              ┌────────────▼──────┐  ┌─────────▼────────┐  ┌──────▼──────────┐
              │  Intent Classifier │  │   Complexity     │  │   Guardrails    │
              │  意图分类 (:8000)  │  │  复杂度评估       │  │  安全护栏        │
              │                   │  │  (:8001)         │  │  (:8002)        │
              │  模式 A: 规则匹配  │  │                  │  │  PII 检测        │
              │  模式 B: LLM Sidecar│  │                  │  │                  │
              │  (:9000)          │  │                  │  │                  │
              └───────────────────┘  └──────────────────┘  └──────────────────┘
                           │                   │                   │
                           └───────────────────┼───────────────────┘
                                               │
                              ┌────────────────▼────────────────────────┐
                              │         Policy Engine (:8003)           │
                              │   模型选择策略 / A/B 测试 / Bandit       │
                              └────────────────┬────────────────────────┘
                                               │
                              ┌────────────────▼────────────────────────┐
                              │              Higress (:8080)            │
                              │         OpenAI 兼容 API 网关             │
                              └────────────────┬────────────────────────┘
                                               │
                    ┌──────────────┬────────────┼────────────┬──────────────┐
                    ▼              ▼            ▼            ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
              │ Qwen2.5  │  │ Qwen2.5  │  │ Qwen2.5  │  │ DeepSeek │  │  其他     │
              │   7B     │  │ Coder 7B │  │   32B    │  │  R1 671B │  │  模型     │
              └──────────┘  └──────────┘  └──────────┘  └──────────┘  └──────────┘
```

### 2.2 组件职责

| 组件 | 端口 | 职责 | 部署方式 |
|------|------|------|---------|
| **Controller** | 8084 | API 编排、路由调度、代理转发、成本追踪 | 主服务 Dockerfile |
| **Intent Classifier** | 8000 | 意图分类（规则模式） | 主服务 Dockerfile |
| **LLM Classifier** | 9000 | 意图分类（LLM Sidecar 模式） | 独立 Dockerfile |
| **Complexity Estimator** | 8001 | 复杂度评估（低/中/高） | 主服务 Dockerfile |
| **Guardrails** | 8002 | PII 检测、安全检查 | 主服务 Dockerfile |
| **Policy Engine** | 8003 | 模型选择决策、A/B 测试、Bandit | 主服务 Dockerfile |
| **Prometheus** | 9090 | 指标采集 | 官方镜像 |
| **Grafana** | 3000 | 监控仪表盘 | 官方镜像 |
| **Redis** | 6379 | 对话记忆存储（可选） | 官方镜像 |

### 2.3 请求处理流程

```
1. 用户发送请求到 ReRout Controller (:8084)
   POST /v1/chat/completions
   {"model": "", "messages": [...], "extra_body": {"routing_policy": "task_router"}}

2. Controller 提取最后一条 user message

3. Intent Classifier 分类意图（6 选 1）
   "帮我写一个排序算法" → {"label": "code_generation", "confidence": 0.85}

4. Complexity Estimator 评估复杂度（3 选 1）
   → {"level": "medium", "confidence": 0.75}

5. Guardrails 安全检查
   → {"passed": true}

6. Policy Engine 做最终模型选择
   intent=code_generation + complexity=medium → 选择 higress-qwen/qwen2.5-coder-7b

7. Controller 改写请求 model 字段，转发到 Higress
   原始 model="" → 改写 model="qwen2.5-coder-7b"
   使用 higress-qwen backend 配置的 API Key 认证

8. Higress 路由到实际的模型服务

9. 模型返回结果 → Higress → Controller → 用户
   响应中附加 routing_explain 字段说明路由决策过程
```

### 2.4 两种路由模式

#### 模式 A：规则匹配（默认）

```
分类延迟: <1ms
准确度: 中等（关键词匹配，中文覆盖有限）
适用: 上线初期、流量验证
```

通过 ReRout 内置的 Intent Classifier，基于关键词规则匹配意图类别。

#### 模式 B：LLM Sidecar

```
分类延迟: ~200ms（取决于小模型推理速度）
准确度: 高（利用小模型的语义理解能力）
适用: 追求分类准确率、中文场景
```

通过独立的 `llm-classifier` 服务，调用 Higress 后的小模型做意图分类。内置中文分类 prompt，支持 mock 模式用于流程测试。

#### 切换方式

修改 `config.yaml` 中一行 URL，重启 Controller：

```yaml
pipeline:
  # 规则模式
  intent_classifier: http://intent:8000/classify
  # LLM 模式（注释上行，取消注释下行）
  # intent_classifier: http://llm-classifier:9000/classify
```

两种模式返回相同的 JSON 格式，Controller 无感知切换。通过响应中的 `routing_explain.pipeline.intent.used` 字段可区分使用了哪种方式（`heuristic` / `llm` / `mock`）。

### 2.5 多 API Key 支持

Higress 为不同模型分配不同的 API Key。通过拆分 backend 实现：

```yaml
backends:
  - name: higress-qwen
    prefix: higress-qwen/
    base_url: http://higress:8080/v1
    api_key_env: HIGRESS_QWEN_API_KEY
    require_api_key: true

  - name: higress-deepseek
    prefix: higress-deepseek/
    base_url: http://higress:8080/v1
    api_key_env: HIGRESS_DEEPSEEK_API_KEY
    require_api_key: true

routing_rules:
  task_router:
    code_generation: higress-qwen/qwen2.5-coder-7b
    reasoning: higress-deepseek/deepseek-r1-67b
    chatbot: higress-qwen/qwen2.5-7b
```

路由后 `higress-qwen/qwen2.5-coder-7b` 被拆分为：
- 前缀 `higress-qwen/` → 匹配对应 backend，使用 `HIGRESS_QWEN_API_KEY`
- 模型名 `qwen2.5-coder-7b` → 作为 `model` 字段发给 Higress

### 2.6 Higress 对接

```
部署前: 用户 → Higress → 模型
部署后: 用户 → ReRout Controller → Higress → 模型
```

- Higress 无需任何配置变更
- 通过网络策略关闭用户对 Higress 的直接访问，只允许 ReRout 内部调用
- 用户只需将 API 地址从 Higress 改为 ReRout Controller

### 2.7 降级策略

```
正常流程: LLM Sidecar → ONNX → 规则匹配 → 默认模型

1. LLM Sidecar 模式下，小模型不可用 → 降级到规则匹配
2. 规则匹配失败 → 降级到 routing_rules 配置的默认映射
3. routing_rules 也无匹配 → 降级到 mock/gpt-4o-mini
```

每一层降级都会在 `routing_explain` 中记录实际使用的分类方式和模型来源。

### 2.8 监控体系

- **Prometheus**：采集每个服务的请求量、延迟、错误率、分类结果分布
- **Grafana**：可视化仪表盘，监控路由效果
- **日志**：每个服务输出结构化 JSON 日志，包含分类结果、耗时、决策路径

---

## 第三章 测试环境部署（Docker Compose）

### 3.1 前置条件

| 要求 | 说明 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 22.04+） |
| Docker | 20.10+ |
| Docker Compose | v2+ |
| 硬件 | 4C 8G 以上，无需 GPU |
| 网络 | 可访问 Higress 网关地址 |

### 3.2 获取代码

```bash
# 从 GitHub 下载仓库（通过安全浏览器下载 ZIP 或 git clone）
git clone https://github.com/sunleeeeei/ReRout.git
cd ReRout
```

### 3.3 配置环境变量

```bash
cp .env.example .env
vim .env
```

关键配置项：

```bash
# Higress 地址（必填）
HIGRESS_BASE_URL=http://your-higress:8080/v1

# Higress API Key（按模型供应商配置）
HIGRESS_QWEN_API_KEY=sk-qwen-xxxxx
HIGRESS_DEEPSEEK_API_KEY=sk-deepseek-xxxxx

# LLM Sidecar 模式（先 mock 验证流程）
LLM_CLASSIFIER_MODE=mock

# 后续接入真实模型时改为：
# LLM_CLASSIFIER_MODE=llm
# LLM_CLASSIFIER_URL=http://your-higress:8080/v1/chat/completions
# LLM_CLASSIFIER_MODEL=qwen2.5-7b
```

### 3.4 配置路由规则

编辑 `config.yaml`：

```yaml
backends:
  - name: higress-qwen
    prefix: higress-qwen/
    base_url: ${HIGRESS_BASE_URL}
    api_key_env: HIGRESS_QWEN_API_KEY
    require_api_key: true

  - name: higress-deepseek
    prefix: higress-deepseek/
    base_url: ${HIGRESS_BASE_URL}
    api_key_env: HIGRESS_DEEPSEEK_API_KEY
    require_api_key: true

  - name: mock
    prefix: mock/
    mock: true
    require_api_key: false

routing_rules:
  task_router:
    code_generation: higress-qwen/qwen2.5-coder-7b
    reasoning: higress-deepseek/deepseek-r1-67b
    summarization: higress-qwen/qwen2.5-7b
    brainstorming: higress-qwen/qwen2.5-32b
    chatbot: higress-qwen/qwen2.5-7b
    open_qa: higress-qwen/qwen2.5-7b
```

### 3.5 启动服务

```bash
docker-compose build
docker-compose up -d
```

### 3.6 验证健康状态

```bash
# 检查所有服务是否正常运行
docker-compose ps

# Controller 健康检查
curl http://localhost:8084/health

# LLM Sidecar 健康检查
curl http://localhost:9000/health
```

### 3.7 测试路由功能

#### 规则模式测试

```bash
curl -X POST http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [{"role": "user", "content": "帮我写一个快速排序算法"}],
    "extra_body": {"routing_policy": "task_router"}
  }'
```

预期：`routing_explain.pipeline.intent.label` 为 `code_generation`，`used` 为 `heuristic`。

```bash
# 测试复杂推理
curl -X POST http://localhost:8084/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "",
    "messages": [{"role": "user", "content": "请分析这段代码的时间复杂度并给出优化建议"}],
    "extra_body": {"routing_policy": "task_router"}
  }'
```

#### LLM Sidecar 模式测试

1. 修改 `config.yaml`：

```yaml
pipeline:
  # intent_classifier: http://intent:8000/classify
  intent_classifier: http://llm-classifier:9000/classify
```

2. 重启 Controller：

```bash
docker-compose restart controller
```

3. 发送相同请求，验证 `used` 字段变为 `mock`（Mock 模式）或 `llm`（真实模型模式）。

### 3.8 监控访问

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000（默认 admin/admin）

### 3.9 常见问题

| 问题 | 排查 |
|------|------|
| Controller 启动报 `Config not loaded` | 检查 `config.yaml` 是否在正确位置 |
| Intent Classifier 分类全返回 `chatbot` | 正常现象（规则模式对未匹配的请求默认归类为 chatbot） |
| 请求转发到 Higress 报 401 | 检查 `.env` 中 API Key 是否正确 |
| LLM Sidecar 连接超时 | 检查 `LLM_CLASSIFIER_URL` 是否可达，增加 `LLM_CLASSIFIER_TIMEOUT` |
| 模型名不匹配 | 确保 `routing_rules` 中的模型名与 Higress 注册名一致 |

---

## 第四章 生产环境部署（Kubernetes）

### 4.1 架构拓扑

```
                    ┌─────────────────────── Kubernetes Cluster ────────────────────────┐
                    │                                                                  │
                    │  ┌─────────────────┐    ┌─────────────────┐                      │
                    │  │  Ingress /       │    │  Higress         │                      │
                    │  │  LoadBalancer    │    │  (已独立部署)     │                      │
                    │  └────────┬────────┘    └────────▲────────┘                      │
                    │           │                      │                               │
                    │  ┌────────▼────────┐             │                               │
                    │  │  ReRout         │             │                               │
                    │  │  Controller     │─────────────┘                               │
                    │  │  (:8084)        │                                             │
                    │  └────────┬────────┘                                             │
                    │           │                                                      │
                    │    ┌──────┼──────┬───────────┬───────────┐                       │
                    │    │      │      │           │           │                       │
                    │  ┌─▼──┐ ┌─▼──┐ ┌─▼────────┐ ┌▼────────┐ ┌▼──────────┐          │
                    │  │意图 │ │复杂│ │安全护栏   │ │策略引擎 │ │LLM Sidecar│          │
                    │  │分类 │ │度  │ │Guardrails │ │Policy   │ │(可选)     │          │
                    │  │    │ │    │ │           │ │         │ │           │          │
                    │  └────┘ └────┘ └───────────┘ └─────────┘ └───────────┘          │
                    │                                                                  │
                    │  ┌────────────┐  ┌────────────┐  ┌────────────┐                  │
                    │  │ Prometheus │  │  Grafana    │  │   Redis    │                  │
                    │  └────────────┘  └────────────┘  └────────────┘                  │
                    └──────────────────────────────────────────────────────────────────┘
```

### 4.2 镜像准备

```bash
# 在外网机器上构建镜像
git clone https://github.com/sunleeeeei/ReRout.git
cd ReRout

# 构建主服务镜像（Controller + Classifiers + Guardrails + Policy Engine）
docker build -t rerout-main:latest .

# 构建 LLM Sidecar 镜像
docker build -t rerout-llm-classifier:latest ./llm-classifier/

# 导出镜像
docker save rerout-main:latest -o rerout-main.tar
docker save rerout-llm-classifier:latest -o rerout-llm-classifier.tar

# SFTP 传到内网后导入
docker load -i rerout-main.tar
docker load -i rerout-llm-classifier.tar

# 推送到内网镜像仓库
docker tag rerout-main:latest your-registry/rerout-main:latest
docker tag rerout-llm-classifier:latest your-registry/rerout-llm-classifier:latest
docker push your-registry/rerout-main:latest
docker push your-registry/rerout-llm-classifier:latest
```

### 4.3 K8s 资源清单

#### Namespace

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: llm-router
```

#### ConfigMap

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rerout-config
  namespace: llm-router
data:
  config.yaml: |
    backends:
      - name: higress-qwen
        prefix: higress-qwen/
        base_url: http://higress.higress-system:8080/v1
        api_key_env: HIGRESS_QWEN_API_KEY
        require_api_key: true
      - name: higress-deepseek
        prefix: higress-deepseek/
        base_url: http://higress.higress-system:8080/v1
        api_key_env: HIGRESS_DEEPSEEK_API_KEY
        require_api_key: true
      - name: mock
        prefix: mock/
        mock: true
        require_api_key: false

    aliases:
      demo: mock/gpt-4o-mini

    routing_rules:
      task_router:
        code_generation: higress-qwen/qwen2.5-coder-7b
        reasoning: higress-deepseek/deepseek-r1-67b
        summarization: higress-qwen/qwen2.5-7b
        brainstorming: higress-qwen/qwen2.5-32b
        chatbot: higress-qwen/qwen2.5-7b
        open_qa: higress-qwen/qwen2.5-7b

    pipeline:
      intent_classifier: http://intent:8000/classify
      # intent_classifier: http://llm-classifier:9000/classify
      complexity_estimator: http://complexity:8001/classify
      guardrails: http://guardrails:8002/check
      policy_engine: http://policy:8003/decide

    memory:
      enabled: false
```

#### Secret

```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rerout-secrets
  namespace: llm-router
type: Opaque
stringData:
  HIGRESS_QWEN_API_KEY: "sk-qwen-xxxxx"
  HIGRESS_DEEPSEEK_API_KEY: "sk-deepseek-xxxxx"
  LLM_CLASSIFIER_API_KEY: ""
```

#### Controller Deployment

```yaml
# k8s/controller.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: controller
  namespace: llm-router
spec:
  replicas: 2
  selector:
    matchLabels:
      app: controller
  template:
    metadata:
      labels:
        app: controller
    spec:
      containers:
        - name: controller
          image: your-registry/rerout-main:latest
          command: ["python", "-m", "controller.main"]
          ports:
            - containerPort: 8084
          envFrom:
            - secretRef:
                name: rerout-secrets
          volumeMounts:
            - name: config
              mountPath: /app/config.yaml
              subPath: config.yaml
          resources:
            requests:
              cpu: "500m"
              memory: "512Mi"
            limits:
              cpu: "1000m"
              memory: "1Gi"
          readinessProbe:
            httpGet:
              path: /health
              port: 8084
            initialDelaySeconds: 10
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 8084
            initialDelaySeconds: 15
            periodSeconds: 20
      volumes:
        - name: config
          configMap:
            name: rerout-config
---
apiVersion: v1
kind: Service
metadata:
  name: controller
  namespace: llm-router
spec:
  selector:
    app: controller
  ports:
    - port: 8084
      targetPort: 8084
  type: ClusterIP
```

#### Intent Classifier Deployment

```yaml
# k8s/intent.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: intent
  namespace: llm-router
spec:
  replicas: 2
  selector:
    matchLabels:
      app: intent
  template:
    metadata:
      labels:
        app: intent
    spec:
      containers:
        - name: intent
          image: your-registry/rerout-main:latest
          command: ["python", "classifiers/intent/app.py"]
          ports:
            - containerPort: 8000
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /metrics
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /metrics
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: intent
  namespace: llm-router
spec:
  selector:
    app: intent
  ports:
    - port: 8000
      targetPort: 8000
```

#### LLM Classifier Deployment（Sidecar）

```yaml
# k8s/llm-classifier.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-classifier
  namespace: llm-router
spec:
  replicas: 2
  selector:
    matchLabels:
      app: llm-classifier
  template:
    metadata:
      labels:
        app: llm-classifier
    spec:
      containers:
        - name: llm-classifier
          image: your-registry/rerout-llm-classifier:latest
          command: ["python", "app.py"]
          ports:
            - containerPort: 9000
          env:
            - name: LLM_CLASSIFIER_MODE
              value: "mock"
            - name: LLM_CLASSIFIER_URL
              value: "http://higress.higress-system:8080/v1/chat/completions"
            - name: LLM_CLASSIFIER_MODEL
              value: "qwen2.5-7b"
          envFrom:
            - secretRef:
                name: rerout-secrets
          resources:
            requests:
              cpu: "250m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          readinessProbe:
            httpGet:
              path: /health
              port: 9000
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet:
              path: /health
              port: 9000
            initialDelaySeconds: 10
            periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: llm-classifier
  namespace: llm-router
spec:
  selector:
    app: llm-classifier
  ports:
    - port: 9000
      targetPort: 9000
```

#### Complexity Estimator Deployment

```yaml
# k8s/complexity.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: complexity
  namespace: llm-router
spec:
  replicas: 1
  selector:
    matchLabels:
      app: complexity
  template:
    metadata:
      labels:
        app: complexity
    spec:
      containers:
        - name: complexity
          image: your-registry/rerout-main:latest
          command: ["python", "classifiers/complexity/app.py"]
          ports:
            - containerPort: 8001
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "250m"
              memory: "256Mi"
          readinessProbe:
            httpGet:
              path: /metrics
              port: 8001
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: complexity
  namespace: llm-router
spec:
  selector:
    app: complexity
  ports:
    - port: 8001
      targetPort: 8001
```

#### Guardrails Deployment

```yaml
# k8s/guardrails.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: guardrails
  namespace: llm-router
spec:
  replicas: 1
  selector:
    matchLabels:
      app: guardrails
  template:
    metadata:
      labels:
        app: guardrails
    spec:
      containers:
        - name: guardrails
          image: your-registry/rerout-main:latest
          command: ["python", "guardrails/app.py"]
          ports:
            - containerPort: 8002
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "250m"
              memory: "256Mi"
          readinessProbe:
            httpGet:
              path: /check
              port: 8002
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: guardrails
  namespace: llm-router
spec:
  selector:
    app: guardrails
  ports:
    - port: 8002
      targetPort: 8002
```

#### Policy Engine Deployment

```yaml
# k8s/policy.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: policy
  namespace: llm-router
spec:
  replicas: 1
  selector:
    matchLabels:
      app: policy
  template:
    metadata:
      labels:
        app: policy
    spec:
      containers:
        - name: policy
          image: your-registry/rerout-main:latest
          command: ["python", "policy_engine/app.py"]
          ports:
            - containerPort: 8003
          env:
            - name: POLICY_MODEL_MAP
              value: '{"code_generation":"higress-qwen/qwen2.5-coder-7b","reasoning":"higress-deepseek/deepseek-r1-67b","summarization":"higress-qwen/qwen2.5-7b","brainstorming":"higress-qwen/qwen2.5-32b","open_qa":"higress-qwen/qwen2.5-7b","chatbot":"higress-qwen/qwen2.5-7b"}'
          resources:
            requests:
              cpu: "100m"
              memory: "128Mi"
            limits:
              cpu: "250m"
              memory: "256Mi"
          readinessProbe:
            httpGet:
              path: /metrics
              port: 8003
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: policy
  namespace: llm-router
spec:
  selector:
    app: policy
  ports:
    - port: 8003
      targetPort: 8003
```

### 4.4 部署步骤

```bash
# 1. 创建 Namespace
kubectl apply -f k8s/namespace.yaml

# 2. 创建配置和密钥
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# 3. 部署各微服务（按依赖顺序）
kubectl apply -f k8s/intent.yaml
kubectl apply -f k8s/complexity.yaml
kubectl apply -f k8s/guardrails.yaml
kubectl apply -f k8s/policy.yaml
kubectl apply -f k8s/llm-classifier.yaml
kubectl apply -f k8s/controller.yaml

# 4. 验证所有 Pod 正常运行
kubectl get pods -n llm-router

# 5. 测试路由
kubectl exec -n llm-router deploy/controller -- curl -s http://localhost:8084/health
```

### 4.5 滚动升级

```bash
# 更新镜像版本（触发滚动升级）
kubectl set image deployment/controller controller=your-registry/rerout-main:v1.1 -n llm-router

# 只更新配置（重启 Pod 重新加载 ConfigMap）
kubectl rollout restart deployment/controller -n llm-router

# 查看升级状态
kubectl rollout status deployment/controller -n llm-router

# 回滚
kubectl rollout undo deployment/controller -n llm-router
```

单独升级某个服务不影响其他服务：

```bash
# 只升级 Intent Classifier
kubectl set image deployment/intent intent=your-registry/rerout-main:v1.1 -n llm-router

# 只升级 LLM Sidecar
kubectl set image deployment/llm-classifier llm-classifier=your-registry/rerout-llm-classifier:v1.1 -n llm-router
```

### 4.6 网络策略

限制只有 ReRout 可以访问 Higress：

```yaml
# k8s/networkpolicy.yaml
# 在 Higress 所在 Namespace 应用
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-only-rerout
  namespace: higress-system
spec:
  podSelector:
    matchLabels:
      app: higress
  policyTypes:
    - Ingress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: llm-router
      ports:
        - port: 8080
```

### 4.7 资源规划参考

| 服务 | 副本数 | CPU 请求/限制 | 内存请求/限制 | 说明 |
|------|--------|--------------|-------------|------|
| Controller | 2 | 500m / 1 | 512Mi / 1Gi | 主要入口，建议多副本 |
| Intent Classifier | 2 | 250m / 500m | 256Mi / 512Mi | 分类服务，建议多副本 |
| LLM Classifier | 2 | 250m / 500m | 256Mi / 512Mi | Sidecar，建议多副本 |
| Complexity Estimator | 1 | 100m / 250m | 128Mi / 256Mi | 轻量计算，单副本即可 |
| Guardrails | 1 | 100m / 250m | 128Mi / 256Mi | 正则匹配，单副本即可 |
| Policy Engine | 1 | 100m / 250m | 128Mi / 256Mi | 决策逻辑，单副本即可 |
| **总计** | **9 Pod** | **~2.5 / 5 Core** | **~2.5 / 5 GB** | |

---

## 附录

### A. 环境变量完整参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `HIGRESS_BASE_URL` | - | Higress OpenAI 兼容 API 地址 |
| `HIGRESS_QWEN_API_KEY` | - | Qwen 系列模型的 API Key |
| `HIGRESS_DEEPSEEK_API_KEY` | - | DeepSeek 系列模型的 API Key |
| `LLM_CLASSIFIER_MODE` | mock | 分类模式：mock / llm |
| `LLM_CLASSIFIER_URL` | - | LLM 分类服务的 API 地址 |
| `LLM_CLASSIFIER_MODEL` | qwen2.5-7b | 分类使用的模型名 |
| `LLM_CLASSIFIER_API_KEY` | - | 分类服务 API Key |
| `LLM_CLASSIFIER_TIMEOUT` | 10 | 分类请求超时（秒） |
| `LLM_CLASSIFIER_LABELS` | code_generation,reasoning,... | 分类标签（逗号分隔） |
| `POLICY_MODEL_MAP` | - | 自定义 intent→model 映射（JSON） |
| `AB_ENABLED` | false | 是否启用 A/B 测试 |
| `BANDIT_EPSILON` | 0.1 | Bandit 探索率 |

### B. 路由规则配置参考

`config.yaml` 中 `routing_rules.task_router` 定义了意图类别到模型的映射。每个值的格式为 `backend前缀/模型名`：

```yaml
routing_rules:
  task_router:
    code_generation: higress-qwen/qwen2.5-coder-7b    # 代码生成 → Qwen Coder
    reasoning: higress-deepseek/deepseek-r1-67b        # 逻辑推理 → DeepSeek R1
    summarization: higress-qwen/qwen2.5-7b            # 摘要 → Qwen 7B
    brainstorming: higress-qwen/qwen2.5-32b           # 头脑风暴 → Qwen 32B
    chatbot: higress-qwen/qwen2.5-7b                  # 闲聊 → Qwen 7B
    open_qa: higress-qwen/qwen2.5-7b                  # 开放问答 → Qwen 7B
```

### C. 监控指标说明

| 指标 | 来源 | 含义 |
|------|------|------|
| `orchestrator_requests_total` | Controller | 总请求数 |
| `orchestrator_request_latency_seconds` | Controller | 请求总延迟 |
| `orchestrator_cost_total` | Controller | 累计成本 |
| `orchestrator_tokens_total` | Controller | Token 消耗 |
| `intent_requests_total` | Intent | 意图分类请求数 |
| `intent_request_latency_seconds` | Intent | 分类延迟 |
| `llm_classifier_requests_total` | LLM Sidecar | 分类请求数 |
| `llm_classifier_latency_seconds` | LLM Sidecar | 分类延迟 |
| `policy_requests_total` | Policy Engine | 策略决策数 |
| `policy_rewards_total` | Policy Engine | 反馈奖励记录 |
