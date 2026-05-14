# rsLLM 总览与愿景

> 本文是 rsLLM 项目的纲领性文档：浓缩调研结论，确立产品愿景，并给出后续四份文档（产品画像 / PRD / HLD）的导航。

---

## 0. 一句话定位

**rsLLM 是一个 Rust 原生、跨平台、面向异构算力的开源 LLM 推理引擎；在 16GB 笔记本上跑得动，在 8×H100 集群上跑得快，并尽可能覆盖主流开源模型生态。**

## 1. 项目缘起

近一年开源 LLM 推理领域出现了三类典型路线：

| 路线 | 代表 | 特点 | 局限 |
|---|---|---|---|
| 通用兼容派 | **llama.cpp**, ollama | 跨平台、GGUF 生态、广覆盖 | 单卡为主，调度简单 |
| 异构计算派 | **ktransformers**, MoonCake | CPU+GPU MoE 卸载，AMX/AVX512 | Python 耦合，Linux only，门槛高 |
| 单模型极致派 | **ds4**, MLX-LM | 紧 IR、Metal 内核、零拷贝 mmap | 锁死单模型/单平台 |

rsLLM 的判断是：**三者之间没有不可调和的矛盾**，只是没有一个项目把工程纪律、跨平台、多后端、多模型全部做齐。Rust 的内存安全 + FFI 能力 + 生态成熟度（cudarc / wgpu / candle 等）让这个事现在可行。

## 2. 调研核心收获

### 来自 ktransformers
- ✅ **`cudaLaunchHostFunc` 把 CPU 工作挂上 CUDA stream**：CPU/GPU 时序由 stream 单一定义
- ✅ **NUMA 感知线程池**：双路 Xeon 上 cross-NUMA 内存访问会腰斩带宽
- ✅ **每层动态 GPU expert mask**：根据 activation frequency EMA 重路由
- ✅ **CPU 变体启动期探测**：AMX > AVX512+BF16 > … > AVX2 自动选最优内核
- ❌ **不要 PyTorch monkey-patching**：rsLLM 用静态图变换
- ❌ **不要 SGLang fork 服务器**：rsLLM 自己拥有 HTTP server

### 来自 ds4 ⭐ 整个工程姿势的根
- ✅ **「格式兼容但代码独立」的工程姿势**——ds4 README 的原话 *"does not link against GGML, but exists thanks to the path opened by llama.cpp"* 是 rsLLM 的根本姿势（详见 [ADR-0001](adr/0001-engine-architecture.md)）
- ✅ **mmap 零拷贝**：80GB 模型不复制到 RAM
- ✅ **磁盘 KV cache 一等公民**：SHA1(token IDs) 作为缓存键，agent 模式下省下重复 prefill
- ✅ **非对称量化**：只压 MoE routed experts 到 2-bit，其他权重保持精度
- ✅ **FP8 (E4M3FN) KV 量化**：64-element block round-trip
- ✅ **Engine / Session 分离**：immutable engine 可共享，mutable session 单线程
- ✅ **官方 logprob 回归测试**：捕获官方 API 输出作为 ground truth
- ❌ **不要 Metal-only**：必须多后端
- ❌ **不要单会话 server**：至少要带前缀复用的请求队列
- ❌ **不要硬编码模型 shape**：从 GGUF metadata 加载 + const-generic 单态化

### 来自现实
- **GGUF 是事实标准**：必须一等公民支持
- **OpenAI / Anthropic API 是事实生态接口**：必须兼容
- **Linux + CUDA + 4090** 是最大单点市场，**MacOS + Metal** 是单机长尾，**Windows + DirectML/CUDA** 是不可忽视的桌面市场

## 3. 愿景

### 3.1 三年愿景
让任何想运行开源 LLM 的人，都能在自己的硬件上得到「最大化的性能 / 质量比」：
- 手里只有 8GB iGPU 笔记本 → 跑 Qwen 7B INT4，得到 15+ t/s
- 一张 4090 → 跑 Llama 70B INT4 或 Mixtral 8x7B，得到 30+ t/s
- 双路 Xeon + 4090 → 跑 DeepSeek-V3 671B，得到 15+ t/s
- 8×H100 → 跑 DeepSeek-V3 FP8，得到 200+ t/s
- M3 Ultra 512GB → 跑 DeepSeek-V3 4-bit，得到 25+ t/s

### 3.2 北极星指标
不是「跑赢任何对手」，而是 **「同一硬件 / 同一模型 / 同一质量下，性能在 llama.cpp 的 1.2× 到 ktransformers 的 0.9× 之间」**，且：
- 单二进制，零 Python 运行时依赖
- 跨 Linux / Windows / macOS，跨 CUDA / Metal / CPU / Vulkan
- 支持 ≥10 个主流开源模型家族
- 内存安全（Rust 保证）

## 4. 非目标（明确不做）

为了避免范围蔓延，下列工作明确**不在**初始 12-18 个月路线图内：
- 训练 / 微调 / RLHF（推理引擎专注推理）
- 多模态 vision/audio（v1 之后再开）
- 自研模型架构（只复用已发布权重）
- 分布式训练 / FSDP / DeepSpeed
- Embedding / Reranker 模型（独立赛道，复杂度不匹配）
- 浏览器 / WASM 部署（远期再说）
- **链接外部推理引擎（llama.cpp / ggml / vLLM / TensorRT-LLM / MLX 等）**——见 [ADR-0001](adr/0001-engine-architecture.md)，仅做**代码层借鉴 + 致谢**，不引入运行时依赖

## 5. 战略支柱（产品哲学）

### 支柱一：**Rust 是「胶水 + 主控」，热路径在底层语言**
- Rust：模型加载、调度、KV 管理、HTTP 服务、CLI、配置
- C/C++/汇编：CPU SIMD 内核（AVX2/AVX512/AMX/NEON/SVE）
- CUDA / Metal / WGSL：GPU 内核
- 优先复用 llama.cpp / ggml 的成熟 INT4 内核（FFI），逐步替换为自有实现

### 支柱二：**GGUF 格式兼容 + 自研内核 + 借鉴开源**（ds4 风格，见 [ADR-0001](adr/0001-engine-architecture.md)）
- **不链接 llama.cpp / ggml**——rsLLM 是独立的纯 Rust + CUDA + Metal 内核栈
- **GGUF 格式兼容**：自己实现 GGUF 解析（几百到一千行 Rust），用户任意 llama.cpp GGUF 可直接加载
- **借鉴而非从零**：CUDA / Metal kernels 显式借鉴 llama.cpp / ds4 / candle / FlashAttention / MLX，按 MIT/Apache 规则致谢
- safetensors 作为「友好导入」（来自 HF），用工具链转 GGUF
- 不发明新格式

### 支柱三：**Engine / Session 分离 + 显式资源**
- `Engine: Send + Sync`：加载完成的不可变模型，可被多 session 共享
- `Session: !Send`：每个推理时间线独占，绑定到一个执行流
- 显式资源所有权（GPU 内存、pinned host buffer、磁盘 KV 文件）

### 支柱四：**异构计算抽象层（CAL, Compute Abstraction Layer）**
- 统一 trait：`Backend`、`Buffer`、`Kernel`、`Stream`
- 按硬件能力分层选择：编译期 feature + 启动期 capability check
- 同一模型，不同后端：cpu-rust / cpu-llama / cpu-amx / cuda / metal / wgpu

### 支柱五：**调度可插拔**
- Single-session：ds4 风格，低算力 / 单用户
- Continuous-batching：vLLM 风格，多用户并发
- Hybrid CPU-GPU layer split：ktransformers 风格，超大模型
- MoE expert offload：cold experts 走 CPU

### 支柱六：**KV cache 一等公民**
- 内存：paged attention（vLLM 风格 block 管理）
- 持久化：磁盘 KV cache（ds4 风格 KVC 文件格式）
- 复用：radix tree prefix cache（多请求共享前缀）
- 压缩：FP8 / INT8 KV 量化、MLA 吸收（DeepSeek 系）

### 支柱七：**API 兼容性是用户体验**
- OpenAI Chat Completions（必备）
- OpenAI Completions（v1.0）
- Anthropic Messages（v1.1）
- SSE 流式 + 工具调用 + JSON 模式

## 6. 路线图概览

形态 G（自研内核 + 借鉴致谢）下，v1.0 时间窗口 **12-18 个月**。

**2026-05-11 review 后调整**：Metal 基础版从 M5 提前到 M2（v0.1.6）；磁盘 KV 简化版从 M4 提前到 M2（v0.1.8）；F014 MoE 验收目标改为 Phi-3.5-MoE；F015/F016 顺序对调。

| 里程碑 | 版本 | 时间 | 关键产出 |
|---|---|---|---|
| **M0：骨架** | v0.1.0 | 第 1-3 月 | Cargo workspace、GGUF loader（自实现）、CPU 后端跑 Llama 7B Q4_K_M、CLI `rsllm chat` |
| **M1：CUDA + 服务化** | v0.1.1-v0.1.2 | 第 4-6 月 | CUDA 后端（借鉴 ggml-cuda）、Paged KV、axum HTTP server、OpenAI Chat API |
| **M2：多模型 + MoE + 早期 Metal + 磁盘 KV** | v0.1.3-v0.1.9 | 第 7-12 月 | Qwen/Mistral/Phi、Phi-3.5-MoE、连续批处理、Metal 基础内核、Radix prefix cache、磁盘 KV 简化版、TP、metrics |
| **M3-M4：异构 + DeepSeek + Agent 完整** | v0.2.0 | 第 13-15 月 | DeepSeek-V2/V3（MLA + MoE）、AMX 内核、cudaLaunchHostFunc 异构（借鉴 ktransformers）、Anthropic API、Tool calling、Metal 完整版、磁盘 KV 完整版、推测解码 |
| **M5：跨平台 GPU + 完善** | v0.2.x | 第 16-18 月 | wgpu、ROCm、JSON mode、多模型控制面 |

详细分解见 [`02-PRD.md`](02-PRD.md) 和 [`features/v0.1.x-roadmap.md`](features/v0.1.x-roadmap.md)。

## 7. 关键设计决策清单

下面是 HLD 阶段需要拍板的决策。✅ 表示已决（见对应 ADR），🟡 表示倾向已明确但未正式决议。

1. ✅ **引擎架构 = ds4 风格（不链接 llama.cpp，但 GGUF 兼容 + 借鉴致谢）**
   - 见 [ADR-0001](adr/0001-engine-architecture.md)
2. ✅ **不引入 candle / tch-rs 作为基础**
   - candle 抽象太高、tch-rs 依赖 libtorch。rsLLM 自己掌控底层
   - 折中：可以借鉴 candle 的 GGUF 解析代码（按 Apache 2.0 致谢）
3. 🟡 **CUDA 绑定**：`cudarc`（轻量、纯 Rust、运行时加载、活跃维护）
4. 🟡 **HTTP server 框架**：`axum`（tokio 原生，SSE 良好支持）
5. ✅ **GGUF 加载**：自实现（参考 ds4 `ds4.c:217-298` 量化解码、candle `candle-core/src/quantized/gguf_file.rs`，按 MIT/Apache 致谢）
6. 🟡 **WGSL / wgpu**：M5 之前不做。先把 CUDA + Metal + CPU 做扎实
7. 🟡 **Windows AMX**：不做。AMX tile permission 需要 Linux `arch_prctl`，Windows 上没有等价接口
8. 🟡 **磁盘 KV 格式**：参考 ds4 KVC，加版本字段、模型指纹、增量更新支持

## 8. 风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| GGUF 生态格式漂移 | 中 | 高 | 跟踪 llama.cpp upstream，引入 `gguf-rs` 测试套件 |
| CUDA 版本兼容性矩阵 | 高 | 中 | 通过 `cudarc` 的运行时加载策略，不静态链接特定 toolkit |
| AMX 内核维护成本 | 高 | 中 | 早期只做 INT8，INT4 留待社区贡献 |
| 模型架构碎片化（MoE / MLA / GQA） | 高 | 中 | 抽象 attention/MLP trait，逐模型扩展 |
| 性能不达 ktransformers | 中 | 高 | 早期阶段以「跑得通 + 跑得对」为先，性能进 M3 后再卷 |
| 跨平台测试矩阵爆炸 | 高 | 中 | CI 覆盖 Linux x86_64 / macOS arm64 主轴，其他平台靠社区 |
| Rust 团队学习曲线 | 中 | 中 | 严格代码规范（见 HLD），引入 PR 审查 |

## 9. 文档导航

| 文档 | 用途 | 适合读者 |
|---|---|---|
| `00-overview.md`（本文） | 愿景 + 战略支柱 | 所有人 |
| `01-product-profile.md` | 产品画像、目标用户、场景 | PM、决策人 |
| `02-PRD.md` | 详细功能需求与验收标准 | 工程团队、PM |
| `03-HLD.md` | 高层架构、模块切分、技术选型 | 工程团队 |
| `adr/0001-engine-architecture.md` | **核心架构决策**：ds4 风格 + 借鉴致谢 | 所有人 |
| `research/ktransformers-analysis.md` | ktransformers 调研原文 | 工程团队 |
| `research/ds4-analysis.md` | ds4 调研原文 | 工程团队 |
| `../README.md` | 项目首页 + 致谢 | 所有人 |
| `../NOTICE.md` | 完整依赖与致谢声明 | 法务、贡献者 |
