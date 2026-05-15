# rsLLM PRD（产品需求文档）

> 本文回答："**我们做什么产品，为谁做，他们能得到什么价值。**"
>
> **技术怎么实现** → 见 [`03-HLD.md`](03-HLD.md)
> **详细 feature 设计与编号** → 见 [`FEATURE_LIST.md`](FEATURE_LIST.md) 与 [`features/`](features/)
> **整体愿景与战略** → 见 [`00-overview.md`](00-overview.md)
> **目标用户画像** → 见 [`01-product-profile.md`](01-product-profile.md)

---

## 0. 一页纸（One-Pager）

| 项 | 内容 |
|---|---|
| **产品** | rsLLM |
| **一句话定位** | Rust 原生 LLM 推理引擎，跨硬件等级、跨模型家族、跨操作系统 |
| **解决的问题** | 让用户在自己的硬件上，用 OpenAI / Anthropic 兼容 API 跑主流开源 LLM，不用容忍 Python 生态部署复杂度，也不用接受 ollama 的黑盒 |
| **北极星** | "同一二进制、同一 API，在 8GB 笔记本和 8×H100 上都跑得动 / 跑得好" |
| **核心差异化** | 单二进制 / 内存安全 / 异构 CPU+GPU / 磁盘 KV / Agent 友好 |
| **当前阶段** | v0.0.1，workspace 骨架 + GGUF parser Phase 4 已完成；正在按"ds4 复刻"重定位 v0.1.0 设计（首发硬件 Mac + AMD Ryzen AI Max+ 395 Linux；首发模型 DeepSeek V4 Flash） |
| **v1.0 时间窗口** | 12-18 个月 |

---

## 1. 产品愿景

让任何想运行开源 LLM 的人，都能在自己的硬件上得到「**最大化的性能 / 质量比**」：

- 手里只有 8GB iGPU 笔记本 → 跑 Qwen 7B INT4，体感流畅
- 一张 4090 → 跑 Llama 70B 或 Mixtral 量级模型，逐字输出快过人类阅读
- 双路 Xeon + 4090 → 跑 DeepSeek-V3 671B，长上下文 agent 工作流可用
- 8×H100 集群 → 跑 DeepSeek-V3 FP8 服务化，承载真实业务流量
- Mac Studio 512GB → 跑 70B+ 模型做创作 / 编码助手

愿景的核心不是"做更快的推理引擎"，而是「**让硬件不再是用户使用开源 LLM 的门槛**」。

完整愿景见 [`00-overview.md`](00-overview.md)。

> **2026-05-14 重定位说明**：v1.0 终点（≥10 个家族 / 跨硬件 / OpenAI+Anthropic）保持不变，但路径分两阶段——
> - **v0.1.0-v0.1.3**：**严格复刻 ds4**——首发硬件 **Mac (Apple Silicon M3+)** + **AMD Ryzen AI Max+ 395 (Strix Halo) Linux**；首发模型 **DeepSeek V4 Flash 单模型**
> - **v0.1.4+**：开始通用化扩展——多模型家族、NVIDIA CUDA、Continuous Batching、Tensor Parallel 等
>
> 选择"ds4 复刻"作为 v0.1.0 路径，是因为它能让我们用最小的不确定性把工程 idiom 立起来（ds4 已经把 Apple Silicon 路径走通过，AMD AI Max+ 是 idiom 等价硬件——统一内存 + 集成 GPU）。

---

## 2. 我们在解决什么问题

### 2.1 当前用户痛点

| 用户类型 | 当前痛点 |
|---|---|
| 自托管开发者 | Python 推理栈（vLLM/transformers）部署链长；ollama 黑盒、性能不可调 |
| 边缘 / 嵌入式 | llama.cpp 跨平台编译矩阵复杂；客户机器五花八门 |
| 异构工作站用户 | ktransformers 是唯一能跑超大 MoE 的方案，但 Python + Linux only |
| 小集群运维 | vLLM/SGLang 跑得动但内存抖动、长尾延迟难控制 |
| Mac 创作者 | LM Studio / ollama 性能不及 MLX / ds4；ds4 又锁死单模型 |
| Agent 工作流 | 系统提示 25k token 每次重 prefill，体验差 |

### 2.2 现有方案为什么不够好

```
       │ 跨平台 │ 多模型 │ 异构 CPU+GPU │ 磁盘 KV │ 内存安全 │ 单二进制
─────────┼───────┼───────┼─────────────┼─────────┼─────────┼─────────
llama.cpp │  ✓   │   ✓   │     基础     │    ✗    │    ✗    │   ✓
ollama    │  ✓   │   ✓   │     基础     │    ✗    │   部分   │   ✓
vLLM      │ Linux │  部分  │     ✗       │    ✗    │    ✗    │   ✗
ktrans    │ Linux │  部分  │     ✓       │    ✗    │    ✗    │   ✗
ds4       │ macOS │  ✗    │     ✗       │    ✓    │    ✗    │   ✓
rsLLM     │  ✓   │   ✓   │     ✓       │    ✓    │    ✓    │   ✓
```

**rsLLM 是唯一同时具备这六项的开源引擎**——这是产品的根本立足点。

### 2.3 核心洞察

1. **GGUF 是事实标准**，但它的生态价值不必绑定 llama.cpp 运行时（ds4 已证明）
2. **agent 工作流让磁盘 KV cache 从"优化"变成"必备"**——但只有 ds4 做了
3. **MoE 的稀疏激活特性让异构 CPU+GPU 跑超大模型成为可能**——但只有 ktransformers 做了
4. **Rust 的工程纪律 + 跨平台能力**让"单二进制覆盖所有场景"成为可能

---

## 3. 目标用户

完整 personas 见 [`01-product-profile.md`](01-product-profile.md)。本节只列 Top 3 + 各自核心痛点：

| Persona | 一句话画像 | 核心痛点 | rsLLM 提供 |
|---|---|---|---|
| **老王**（自托管开发者） | 家里 4090 + 64GB，给 IDE 插件用 | Python 栈难分发、ollama 黑盒 | 单 Rust 二进制 + OpenAI API |
| **Dr. 张**（异构工作站） | 双路 Xeon + 4090 + 512GB，跑 DeepSeek-V3 | ktransformers Python 难维护 | 异构调度 + Rust 错误链 |
| **设计师阿康**（Mac 创作者） | M3 Ultra 512GB，跑大模型做创意 | LM Studio 黑盒、ds4 锁单模型 | 多模型 Metal 后端 + Anthropic API |

---

## 4. 产品能力（用户视角）

### 4.1 模型能力

**v1.0 必须覆盖的开源模型家族**：

| 家族 | 代表型号 | 体验承诺 |
|---|---|---|
| Llama | Llama 3.1 8B / 70B, Llama 3.2 1B / 3B | 流畅本地推理，等价 ollama 的易用 |
| Qwen | Qwen 2.5 7B / 14B / 32B / 72B | 同上 |
| Mistral | Mistral 7B v0.3 | 含 sliding window 支持 |
| Phi | Phi-3.5-mini, Phi-3.5-MoE | MoE 初体验 |
| Gemma | Gemma 2 2B / 9B / 27B | — |
| Mixtral | Mixtral 8x7B / 8x22B | 需异构 CPU+GPU |
| Qwen-MoE | Qwen2-57B-A14B | — |
| DeepSeek | DeepSeek-V2 / V2.5 / V3 / R1 | 含 MLA + MoE 完整支持 |
| GLM | GLM-4 9B | — |
| Yi | Yi 1.5 9B / 34B | — |

**用户拿到任意主流 GGUF 文件 → 能立刻加载使用**，无需转格式、无需配套工具链。

### 4.2 部署形态

用户可以三种方式使用 rsLLM：

1. **单二进制 CLI**——`rsllm chat -m model.gguf`，下载一个文件即可使用
2. **本地 HTTP 服务**——`rsllm serve` 启动后是 OpenAI API 兼容服务器
3. **Rust 库嵌入**——作为 crate 集成进桌面应用 / 服务端

三种形态共享同一套底层引擎，**用户在不同形态间切换无认知成本**。

### 4.3 API 兼容

| API | 用户能直接接的客户端 |
|---|---|
| **OpenAI Chat Completions** | OpenAI SDK / LangChain / Open WebUI / IDE 插件（Continue / Cursor）/ 大量第三方工具 |
| **OpenAI Completions** | 旧式 completions 接口客户端 |
| **Anthropic Messages** | **Claude Code / opencode 等 agent 工具直接可用** |
| **SSE 流式** | 所有上述客户端的流式输出原生支持 |

### 4.4 性能体验（用户感知）

我们不用 t/s 数字做承诺，用**用户能感知的体验**：

| 场景 | 体验承诺 |
|---|---|
| 4090 + Llama 3 8B 量化 | "**比阅读速度快得多，几乎是 OpenAI API 的感觉**" |
| 4090 + Llama 70B 量化 | "**比阅读速度快**，长答案不需要等" |
| 4090 + 64GB + DeepSeek-V3 | "**慢但可用**，长 agent 工作流可接受" |
| M3 Max + 70B 量化 | "**比阅读速度快**，Mac 上跑大模型不再卡顿" |
| 8×H100 + 50 并发 | "**长尾不抖动**，企业级稳定性" |
| Agent 25k 系统提示重复使用 | "**第二次请求秒回**，磁盘 KV 命中" |

技术性能基线（具体 t/s 数字）见 [`03-HLD.md`](03-HLD.md) §附录 B 与 [`FEATURE_LIST.md`](FEATURE_LIST.md) 各 feature 验收标准。

### 4.5 跨平台体验

#### v1.0 终点（不变）

| 平台 | 体验承诺 |
|---|---|
| Linux x86_64 | **Tier 1**：CPU + CUDA 全功能，CI 全绿，主战场 |
| macOS arm64 | **Tier 1**：CPU + Metal，体验与 Linux 等价 |
| Linux aarch64 | Tier 2：CPU + NEON，best-effort |
| Windows x86_64 | Tier 2：CPU + CUDA，无 AMX 加速 |
| macOS x86_64 | Tier 2：仅 CPU |
| FreeBSD / 其他 | Tier 3：社区维护 |

**用户在 Tier 1 平台上的体验是商业级的**，Tier 2 是"能用、好用"，Tier 3 是"社区能跑得起来"。

#### v0.1.0 首发硬件（重定位后）

v0.1.0 阶段不追求覆盖所有 Tier 1 平台，而是**聚焦两台特定机器**（这是 ds4 复刻的实施载体）：

| 平台 | v0.1.0 体验承诺 | 后端 |
|---|---|---|
| **Mac M3+（Apple Silicon）** | DS V4 Flash decode ≥15 tok/s，Metal 主路径 | F025 Metal + NEON CPU 验证 |
| **AMD Ryzen AI Max+ 395 (Strix Halo) Linux** | DS V4 Flash decode ≥3 tok/s（CPU only），128GB + 2TB SSD 装下 140GB 量化模型 | F004 AVX-512 CPU + VNNI |

AMD AI Max+ 在 v0.1.1 加 iGPU(Vulkan) 后提升到 ≥10 tok/s，v0.1.2 调优后 ≥15 tok/s。NVIDIA CUDA、Windows、Linux aarch64 等覆盖延后到 v0.1.6+。

### 4.6 Agent 工作流

rsLLM 是少数几个**把 agent 工作流作为一等公民**的开源推理引擎：

| 能力 | 用户价值 |
|---|---|
| **磁盘 KV cache** | 25k token 系统提示一次 prefill，后续会话直接复用——`Claude Code` 用户最大痛点 |
| **Radix prefix cache** | 多 session 共享前缀的 KV，agent 集群场景节省 60%+ prefill 计算 |
| **工具调用双协议** | OpenAI tools / Anthropic tool_use 都原生支持 |
| **多模型并存** | 控制面 API 动态 load/unload，按 query 复杂度路由不同模型 |
| **Anthropic Messages API** | Claude Code / opencode 切换 `ANTHROPIC_BASE_URL` 即可用 rsLLM 做本地后端 |

---

## 5. 范围

### 5.1 v1.0 In Scope

- 文本生成（chat / completion）
- ≥10 个主流开源模型家族
- CPU / CUDA / Metal 三个 Tier 1 后端
- OpenAI + Anthropic API 兼容
- 异构 CPU+GPU MoE 卸载
- 磁盘 KV cache + Prefix cache
- 单二进制 CLI + HTTP server + Rust crate 三种形态

### 5.2 Out of Scope（v1.0 不做）

- 训练 / 微调 / RLHF（推理引擎专注推理）
- 多模态 vision / audio（v1.1 之后再开）
- 自研模型架构（只复用已发布权重）
- 分布式训练 / FSDP / DeepSpeed
- 跨节点分布式推理（单节点多 GPU 内）
- Embedding / Reranker 模型
- 浏览器 / WASM 部署
- **链接外部推理引擎** llama.cpp / vLLM / TensorRT-LLM 等作为运行时依赖（见 [ADR-0001](adr/0001-engine-architecture.md)）

---

## 6. 用户故事

### 6.1 老王的一天（自托管开发者）

> 老王下班回家，想本地化一些代码助手任务，避免敏感代码上云。
>
> ```
> $ rsllm serve -m ~/models/qwen-2.5-32b-instruct-q4_k_m.gguf --port 8080
> ```
>
> 启动 5 秒。打开 VS Code Continue 插件，把 base URL 改成 `http://localhost:8080/v1`。
> 立即可用，逐字输出比 GPT-4 还快，因为没有网络往返。
> 24 小时后回家继续用，进程没崩、内存稳定。

### 6.2 Dr. 张的工作站（异构超大模型）

> Dr. 张实验室刚买的工作站：双路 Xeon Gold + 4090 + 512GB DRAM。
>
> ```
> $ rsllm serve -m deepseek-v3-iq2.gguf \
>     --backend hybrid \
>     --cpu-experts auto \
>     --kv-disk-dir /tmp/ds-kv
> ```
>
> 加载 5 分钟（mmap 388GB）。第一个请求 prefill 慢，但第二个相同 system prompt 的请求**首 token 100ms 内**——磁盘 KV cache 命中了。
> 学生们说"比上次 ktransformers 部署稳定多了，不报莫名其妙的 Python 错"。

### 6.3 阿康的 Mac Studio（创作场景）

> 阿康用 M3 Ultra 512GB Mac Studio 写小说。
>
> ```
> $ rsllm serve -m llama-3.1-70b-q4_k_m.gguf
> ```
>
> Mac 风扇没怎么转，70B 模型 25 t/s 输出。配合 Claude Code 客户端，`ANTHROPIC_BASE_URL=http://localhost:8080` 一切就绪。
> "本地 Claude，离线可用，不收钱。"

### 6.4 小李的客户部署（嵌入式分发）

> 小李做桌面应用，要给 Windows 客户机集成 LLM 能力。客户机器 16GB iGPU 笔记本。
>
> 小李在打包时附带 `rsllm.exe`（45MB），应用启动时 spawn 这个进程。
> 客户机器 CPU 自动用 AVX2 路径，无需配置。
> 客户没遇到"找不到 cuda12.dll"这种事——`rsllm` 是单文件。

### 6.5 老赵的小集群（生产服务）

> 老赵创业团队 4×A100 80GB，给团队 50 人提供内部 API。
>
> ```
> $ rsllm serve -m llama-3.1-70b-bf16 --tp 4 --batch continuous
> ```
>
> 50 人 PR review / 文档生成 / 代码补全请求混合，P99 延迟稳定。
> Prometheus 看板显示 GPU 占用 85%，KV cache 利用率 70%。
> 老赵没在 Slack 收到任何 OOM 告警。

---

## 7. 路线图（用户价值视角，2026-05-14 重定位后）

每个版本告诉用户**这个版本之后能多用到什么**。详细 feature 列表见 [`FEATURE_LIST.md`](FEATURE_LIST.md)。

### 7.1 v0.1.0-v0.1.3：ds4 复刻路径

| 版本 | 用户能多用到什么 |
|---|---|
| **v0.1.0**（M0：ds4 复刻第一阶段） | 第一次能用 rsLLM：**在 Mac M3+ 或 AMD Ryzen AI Max+ 395 Linux 上单二进制跑通 DeepSeek V4 Flash chat**（Mac 经 Metal 加速 ≥15 tok/s，AMD CPU only 时 ≥3 tok/s） |
| **v0.1.1** | AMD AI Max+ 上 iGPU(Vulkan) 加速接入，decode 提升到 ≥10 tok/s，进入"日常可用" |
| **v0.1.2** | 双平台 kernel 性能调优 + 全 kernel 数值回归，Mac ≥25 tok/s / AMD ≥15 tok/s |
| **v0.1.3** | **磁盘 KV cache** 命中：25k token 系统提示重复使用秒回——agent 工作流痛点的核心解药 |

### 7.2 v0.1.4-v0.1.9：通用化扩展

| 版本 | 用户能多用到什么 |
|---|---|
| **v0.1.4** | **HTTP server 双协议**（OpenAI + Anthropic + tool calling）：Open WebUI / Claude Code 等客户端切换 base URL 直接接入 rsLLM |
| **v0.1.5** | **logprob 回归测试体系** 上线，rsLLM 与 DeepSeek 官方 API token bytes 级一致，"可证明正确" |
| **v0.1.6** | NVIDIA CUDA 后端，4090/H100 用户开始受益 |
| **v0.1.7-v0.1.8** | 第二、第三个模型家族（Qwen 3.6 / GLM-5.1 / Kimi K2.6 / Gemma 4 中选 2 个），开始通用引擎拼图 |
| **v0.1.9** | 多 GPU TP：双 4090 跑 DS V4 Flash 或同规模模型 ≥12 tok/s |

### 7.3 v0.2.0+：超越 ds4

| 版本 | 用户能多用到什么 |
|---|---|
| **v0.2.0** | **超越 ds4**：异构 CPU+GPU MoE 卸载 + AMX/AVX-512 算子统一 + 推测解码 + 历史 DS V2/V3 模型 + Metal 完整版 + KVC 完整版（continued + tool replay） |
| **v0.2.x** | 长尾覆盖：Continuous Batching + Prefix cache + wgpu 兜底 + JSON mode + 多模型控制面 + ROCm 独立 dGPU |
| **v1.0.0** | 商业级稳定性 + 全平台 CI + 性能达标（vs ds4 同硬件 ≥ 85%） |

> 路线图相比 2026-05-11 初稿做了**重大重定位**：v0.1.0 不再是"CPU 跑 Llama 7B"通用引擎 MVP，而是"双硬件 ds4 复刻"。详见 [`features/v0.1.0.md#修订记录`](features/v0.1.0.md#修订记录)。

---

## 8. 成功标准（产品视角）

### 一年内（v1.0 时）

- 🎯 GitHub Stars **5000+**
- 🎯 真实生产部署案例 **≥ 2**
- 🎯 外部贡献者 **≥ 10**
- 🎯 三个主流模型家族（Llama / Qwen / DeepSeek）跑通且性能 ≥ llama.cpp
- 🎯 在 4090 + 64GB DRAM 上 DeepSeek-V3 IQ2 跑通
- 🎯 Mac 用户能用 Metal 跑 70B 模型

### 三年内

- 🎯 成为 Rust 推理引擎事实标准
- 🎯 模型家族覆盖 **≥ 10**
- 🎯 跨 **5** 个硬件后端（CUDA / Metal / Vulkan / CPU x86 / CPU ARM）
- 🎯 性能在 ktransformers 0.9× 以上、llama.cpp 1.3× 以上
- 🎯 **≥ 3** 家公司贡献核心代码

---

## 9. 风险与未决（产品视角）

### 产品风险

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| Mac 用户在 v0.1.6 前体验差，转向 LM Studio | 中 | 中 | v0.1.6 Metal 基础版必须达标 |
| Agent 用户在 v0.1.8 前没磁盘 KV 转向 ds4 | 低 | 中 | v0.1.8 简化版 KVC 可用即可 |
| 性能始终上不去，用户流向 llama.cpp | 中 | 高 | 大量借鉴 llama.cpp 内核 + 性能基线每月对照 |
| Rust 同类项目（mistral.rs）领先并赢得心智 | 中 | 中 | 差异化锚定在"异构 + 磁盘 KV + Agent" |
| 出现一个杀手级硬件（Hopper 替代品） | 低 | 高 | 借鉴战略保护对硬件演进的跟进能力 |

### 产品未决项

| 项 | 状态 |
|---|---|
| 是否在 v1.0 引入控制面 Admin UI？ | 倾向否，CLI + API 够 |
| 是否提供模型仓库（hub.rsllm.io）？ | 不做，复用 HuggingFace |
| 商业化策略？ | 暂无，纯开源项目 |
| 是否对企业用户提供 SLA / 付费支持？ | 看一年后社区情况 |

技术性未决项（cargo feature 切分、CUDA 绑定选型等）见 [`03-HLD.md`](03-HLD.md)。

---

## 10. 与本文件的关系

| 这份文档不写什么 | 在哪里写 |
|---|---|
| 具体技术选型（cudarc / axum / objc2 …） | [`03-HLD.md`](03-HLD.md) |
| Cargo workspace 切分 | [`03-HLD.md`](03-HLD.md) §2 |
| 关键 trait 定义（Backend / KvCache …） | [`03-HLD.md`](03-HLD.md) §3 |
| 详细 feature 编号与状态 | [`FEATURE_LIST.md`](FEATURE_LIST.md) |
| 每个 feature 的 6 节设计 | [`features/v*.md`](features/) |
| 引擎架构决策（不链接 llama.cpp 等） | [`adr/0001-engine-architecture.md`](adr/0001-engine-architecture.md) |
| 性能 t/s 基线 | [`03-HLD.md`](03-HLD.md) §附录 B + FEATURE_LIST 各 feature 验收 |
| 编码规范 | [`03-HLD.md`](03-HLD.md) §12 |
