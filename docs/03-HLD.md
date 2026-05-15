# rsLLM HLD（高层设计）

> 本文聚焦「**怎么做**」：架构图、模块切分、关键 trait、技术选型、数据流、并发模型。
> 不写一行 Rust 代码（只用伪码示意 trait 形态）。具体实现到 LLD / 编码阶段再讨论。

---

## 1. 整体架构（C4 - Level 1）

```
                       ┌─────────────────────────────────────┐
                       │            外部生态                  │
                       │  (OpenAI SDK, Claude Code, Open WebUI,│
                       │   IDE plugins, Anthropic Tools)     │
                       └──────────────┬──────────────────────┘
                                      │ HTTP/SSE
                                      ▼
   ┌──────────────────────────────────────────────────────────┐
   │                       rsLLM 进程                          │
   │                                                          │
   │   ┌────────────┐    ┌────────────┐    ┌────────────┐    │
   │   │   CLI      │    │ HTTP Server│    │ Rust Lib   │    │
   │   │ (rsllm)    │    │  (axum)    │    │ (rsllm-*)  │    │
   │   └─────┬──────┘    └─────┬──────┘    └─────┬──────┘    │
   │         └─────────────────┼─────────────────┘            │
   │                           ▼                              │
   │   ┌──────────────────────────────────────────────────┐   │
   │   │          Scheduler & Session Manager             │   │
   │   │  - SingleSession / ContinuousBatch / Hybrid      │   │
   │   │  - Priority queue + KV cache lifecycle           │   │
   │   └──────────────┬───────────────────────────────────┘   │
   │                  ▼                                       │
   │   ┌──────────────────────────────────────────────────┐   │
   │   │              Inference Runtime                   │   │
   │   │  Model Graph → Pipeline → Kernel Dispatch        │   │
   │   └──────────────┬───────────────────────────────────┘   │
   │                  ▼                                       │
   │   ┌──────────────────────────────────────────────────┐   │
   │   │       Compute Abstraction Layer (CAL)            │   │
   │   │  Backend trait + Buffer trait + Stream trait     │   │
   │   └─────┬───────────┬──────────┬──────────┬─────────┘   │
   │         ▼           ▼          ▼          ▼              │
   │   ┌────────┐  ┌──────────┐ ┌──────┐ ┌─────────┐        │
   │   │  CPU   │  │  CUDA    │ │ Metal│ │  wgpu   │        │
   │   │backend │  │ backend  │ │backend│ │ backend │        │
   │   └────────┘  └──────────┘ └──────┘ └─────────┘        │
   │                                                          │
   │   ┌──────────────────────────────────────────────────┐   │
   │   │     I/O 层：GGUF loader, mmap, KV disk cache     │   │
   │   └──────────────────────────────────────────────────┘   │
   └──────────────────────────────────────────────────────────┘
```

## 2. Cargo Workspace 切分

```
rsllm/
├── Cargo.toml                # workspace root
├── crates/
│   ├── rsllm-core/           # Engine / Session / SamplingParams + 公共 trait
│   ├── rsllm-gguf/           # GGUF 解析与量化解码（无 GPU 依赖）
│   ├── rsllm-tokenizer/      # JoyAI 状态机分词器（v0.1.0）→ 多家族 PreTokenizer trait（v0.1.7+）
│   ├── rsllm-cal/            # Compute Abstraction Layer trait
│   ├── rsllm-backend-cpu/    # CPU 后端：NEON dotprod (Mac) + AVX-512 VNNI (AMD Strix Halo+)
│   ├── rsllm-backend-metal/  # Metal 后端（objc2 + .metal kernels）—— v0.1.0 主 Mac 加速
│   ├── rsllm-backend-vulkan/ # Vulkan compute 后端 —— v0.1.1 主 AMD iGPU 加速（Strix Halo Radeon 8060S）
│   ├── rsllm-backend-cuda/   # CUDA 后端（cudarc）—— v0.1.6 接入
│   ├── rsllm-backend-cpu-amx/# 可选 AMX 内核（cfg feature, Intel SPR+ only）—— v0.2.0
│   ├── rsllm-backend-wgpu/   # 可选 wgpu 后端（跨平台兜底）—— v0.2.1
│   ├── rsllm-models/         # 模型架构（v0.1.0: deepseek_v4_flash；v0.1.7+: qwen/glm/kimi/gemma）
│   ├── rsllm-kvcache/        # 三级 KV（v0.1.0 DS V4）+ Paged KV（v0.1.6 CUDA 路径）+ 磁盘 KV (KVC, v0.1.3)
│   ├── rsllm-scheduler/      # 调度器（v0.1.4 单 session 服务化；v0.2.x continuous batching）
│   ├── rsllm-server/         # axum HTTP server + OpenAI/Anthropic API 适配（v0.1.4）
│   ├── rsllm-cli/            # 命令行入口（linenoise REPL 风格，编译为 `rsllm` 二进制）
│   └── rsllm-bench/          # 性能基准
├── tests/                    # 跨 crate 集成测试
├── benches/                  # criterion 基准
├── examples/                 # 嵌入示例
├── docs/                     # 本设计文档树
├── xtask/                    # 自定义构建任务（download model, run e2e）
└── rust-toolchain.toml
```

**v0.1.0 实际包含的 crate**：`core / gguf / tokenizer / cal / backend-cpu / backend-metal / kvcache / models / cli`（9 个）。其余 backend / scheduler / server / bench 是 v0.1.x+ 增量。

**每个 backend 都是独立 crate**，通过 `rsllm-core` 的 `Backend` trait 集成。`rsllm-cli` 默认 feature 集成 CPU + 平台对应 GPU 后端（macOS → metal，Linux x86_64 → vulkan/cuda，其他 → cpu only），可裁剪。

**所有后端都是 rsLLM 自研的纯 Rust + CUDA/Metal/Vulkan/SIMD 内核**——没有 `rsllm-backend-llama` 这样的 FFI crate。详见 [`adr/0001-engine-architecture.md`](adr/0001-engine-architecture.md)。

### 2.1 关于 AMD AI Max+ 统一内存（v0.1.0 + v0.1.1 关键）

Strix Halo 是 **统一内存架构**（CPU 与 iGPU 共享 LPDDR5X-8000，256-bit，~256 GB/s）。这跟 Apple Silicon UMA 同类型，导致一个工程上的好处：
- `rsllm-gguf` mmap 的权重，CPU 和 iGPU 都能零拷贝访问
- v0.1.0 的 CPU 路径与 v0.1.1 的 Vulkan iGPU 路径**共享同一份权重 buffer**，切换不需要 upload/download
- 128GB 内存 + 2TB SSD 配合下，DS V4 Flash 的 140-160GB 量化权重可走 mmap 流式访问冷专家（每 token 只激活 6/256 ≈ 2.3% expert）

这跟传统 x86+离散 CUDA GPU 完全不同。`rsllm-backend-vulkan` 在 Strix Halo 上要利用这个特性，buffer 分配走 `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT` 组合。

## 3. 核心抽象（伪码）

### 3.1 Engine / Session

```text
// rsllm-core
struct Engine {
    model: ModelGraph,            // 模型架构 + 权重指针（mmap）
    backend: BackendHandle,       // 选定的 backend
    tokenizer: Arc<Tokenizer>,
    config: EngineConfig,
}
// Send + Sync —— 可被多 session 共享

struct Session {
    engine: Arc<Engine>,
    kv_cache: KvCache,            // session 私有 KV 状态
    stream: BackendStream,        // 绑定到一个 CUDA stream / CPU thread group
    history_hash: TokenHash,      // 用于 disk KV 复用
    sampler: Sampler,
}
// !Send —— 推理时间线单线程独占
```

### 3.2 Backend trait（CAL）

```text
trait Backend {
    type Buffer: Buffer;
    type Stream: Stream;
    type Event;

    fn capability() -> BackendCapability;     // 编译期标识

    fn alloc(&self, size: usize, dtype: DType) -> Self::Buffer;
    fn upload(&self, host: &[u8], dst: &mut Self::Buffer, stream: &Self::Stream);
    fn download(&self, src: &Self::Buffer, host: &mut [u8], stream: &Self::Stream);

    fn stream(&self) -> Self::Stream;

    fn dispatch(&self, op: KernelOp, args: KernelArgs, stream: &Self::Stream);

    fn sync(&self, stream: &Self::Stream);
}

trait Buffer {
    fn len(&self) -> usize;
    fn as_device_ptr(&self) -> DevicePtr;     // CUDA: CUdeviceptr; Metal: MTLBuffer
}

trait Stream { /* opaque queue */ }
```

`KernelOp` 是高层算子枚举：`MatMul`, `RmsNorm`, `RoPE`, `FlashAttention`, `MoEDispatch`, `Sample`, …。每个 backend 在 `dispatch` 里 fan-out 到自己的内核实现。

### 3.3 ModelGraph trait

```text
trait Model {
    fn arch(&self) -> &str;
    fn config(&self) -> &ModelConfig;
    fn forward(
        &self,
        backend: &dyn Backend,
        stream: &dyn Stream,
        input: ForwardInput<'_>,
        kv: &mut KvCache,
    ) -> ForwardOutput;
}
```

具体模型在 `rsllm-models/` 下：`llama.rs`、`qwen.rs`、`mixtral.rs`、`deepseek_v3.rs` 等，各自实现 `forward`。共享算子（`rms_norm`、`rope`、`swiglu`、`moe`）放在 `rsllm-models/common/`。

### 3.4 KvCache

```text
struct KvCache {
    layout: KvLayout,        // dense / paged / mla / sliding-window
    blocks: Vec<KvBlock>,
    block_table: BlockTable, // 类 vLLM
    dtype: DType,            // f16 / bf16 / fp8 / int8
    backend: BackendHandle,
}
```

支持的 layout：
- **Dense**：连续，简单 session 用
- **Paged**：块大小 16/32，vLLM 风格
- **MLA**：压缩 KV（DeepSeek）
- **Sliding Window**：限定窗口（Mistral）

## 4. 关键数据流

### 4.1 Decode 一步（单 session 在 CUDA 上）

```
Token i → embedding lookup
       → for each layer:
            rms_norm
            QKV proj (matmul)
            apply RoPE
            update KV cache block
            flash attention
            output proj (matmul)
            rms_norm
            MLP (gate/up/down matmul + swiglu)
       → final rms_norm
       → lm_head (matmul)
       → sample(logits) → token i+1
```

所有算子在同一 CUDA stream 上排队。Sample 完成后 `cuMemcpyDtoH` token id 回主机，触发下一步。

### 4.2 异构 MoE 解码（M3+）

```
Token i → ... → router selects top-K experts
       → gpu_mask = expert ∈ hot_set
       → for hot experts: dispatch on CUDA stream
       → for cold experts:
            cuLaunchHostFunc(stream, |args| {
                cpu_pool.submit(expert_fn, args);
                cpu_pool.wait();
            })
       → accumulate hot + cold outputs
       → next layer
```

`cuLaunchHostFunc` 让 CPU 工作挂到 CUDA stream 上，保留时序保证。CPU 池 NUMA 绑定。

### 4.3 连续批处理（M2+）

```
请求 r1 (prefilling), r2 (decoding), r3 (decoding), …
            ↓
        Scheduler tick:
            合并 r1 chunked prefill + r2/r3 decode tokens
            → 一个 batched forward
            → split outputs back to per-request streams
```

参考 vLLM 设计：每 tick 从队列取若干请求，按 token budget 打包。Paged KV 让显存碎片管理可控。

### 4.4 磁盘 KV 命中

```
新请求 R 进入：
  1. hash = sha1(token_ids[:N])
  2. lookup disk_kv_index[hash]
  3. hit?
       是 → mmap KVC 文件 → 上传到 GPU paged blocks → 跳过 prefill
       否 → 正常 prefill → 离开 session 时 dump 到 KVC
```

KVC 文件格式分两版：

**v0.1.3 简化版（F022，直接复刻 ds4 KVC v1）**：

```
Header (48 bytes):
  magic "KVC"        3
  version u8         1   (=1)
  quant_bits u32     4   防止换量化方案误用
  reason u32         4   COLD / CONTINUED / EVICT / SHUTDOWN
  token_count u64    8
  ctx_size u64       8
  created_at u64     8   unix timestamp
  payload_bytes u64  8   后续 payload 长度
Body:
  rendered_text: 人类可读的渲染文本（调试用）
  engine_payload: 由 ds4_session_save_payload 等价的 Rust 函数写入
    - magic "DSV4" u32 = 0x34565344
    - version u32
    - vocab_size u32
    - n_layers u32
    - head_dim u32
    - token_ids [u32; token_count]
    - per_layer KV tensor bytes（三级 KV：raw_kv + compressed_kv + indexer_kv）
  [可选] tool_id_map: magic "KTM" + entries (v0.1.4 server 用)
```

文件名为 `SHA1(token_ids).kvc`（不是 SHA1(text)，避免 BPE 重新分词产生不同 hash）。

**v0.2.0 完整版（F031）**：在简化版基础上加 continued checkpoint（会话进行中阶段性保存）+ tool call replay map（DSML 块的 byte-exact 回放索引），格式向后兼容。

## 5. 关键技术选型

| 维度 | 选型 | 理由 |
|---|---|---|
| Async runtime | `tokio` | 生态最强，axum 原生 |
| HTTP server | `axum` | tower 生态，SSE 一等支持 |
| CUDA 绑定 | `cudarc` | 纯 Rust，运行时加载，无 build.rs CUDA 依赖 |
| Metal 绑定 | `objc2` + `metal` crate | objc2 是 metal-rs 的现代替代 |
| Tokenizer | `tokenizers`（HF） | Rust 实现，覆盖广 |
| Mmap | `memmap2` | 事实标准 |
| 序列化 | `serde` + 自定义二进制（KVC、内部 IPC） | 标准 |
| Logging | `tracing` + `tracing-subscriber` | 结构化日志 |
| Metrics | `prometheus` crate | 标准 |
| CLI | `clap` v4 | 标准 |
| 配置 | `figment` | 多源合并（CLI / env / file） |
| 测试 | `cargo test` + `criterion` | 标准 |
| 模糊测试 | `cargo fuzz`（tokenizer / GGUF parser） | 防护边界 |

### 5.1 关于 llama.cpp FFI（决策：不引入）

**决策**：**不**引入 llama.cpp / ggml 作为编译期或运行期依赖。详见 [`adr/0001-engine-architecture.md`](adr/0001-engine-architecture.md)。

简要理由：
- rsLLM 走 ds4 风格：**格式兼容但代码独立**——读 GGUF，不链接 ggml
- 自研内核才能完全自控 KV cache、调度、量化、磁盘 KV——这是 rsLLM 差异化的根
- llama.cpp / ggml 的 CUDA / Metal kernels 通过**借鉴 + MIT 致谢**复用，不需要 FFI

代码层借鉴策略：
- CUDA 后端借鉴 `ggml-cuda.cu` 的算子（matmul / attention / norm / rope / quant）
- Metal 后端借鉴 `ggml-metal.m` + MLX
- CPU 后端借鉴 ggml CPU SIMD 内核与 ds4 ARM NEON 路径
- 复用代码片段的源文件 header 双重署名 `The rsLLM authors` + `The ggml authors` / `The ds4 authors`

### 5.2 关于 candle / tch-rs / mistral.rs（决策：不依赖）

**决策**：不依赖任何外部推理框架。

理由：
- candle 抽象太高，不易接入异构调度
- tch-rs 依赖 libtorch 二进制
- mistral.rs 与我们定位重叠，是同行不是依赖
- rsLLM 需要对底层有完整控制（KV layout、stream、内核选择）
- 可以**参考代码**：candle 的 GGUF 解析、Llama 实现（Apache 2.0 致谢）

## 6. 并发模型

### 6.1 进程级
- 一个 `Engine` 共享给多个 `Session`（`Arc<Engine>`）
- 多 `Session` 通过 `Scheduler` 协调
- 单一 tokio runtime，I/O 与请求处理共用

### 6.2 推理执行
- **CPU 后端**：内部使用 `rayon` 或自管 thread pool，每个 NUMA 节点一个 pool
- **CUDA 后端**：每个 `Session` 持有一个 `CUstream`，scheduler 决定 stream 复用策略
  - SingleSession 模式：一 session 一 stream
  - ContinuousBatch 模式：所有 session 共享 1-2 个 stream（batched forward）
  - Hybrid CPU+GPU：CPU 工作通过 `cuLaunchHostFunc` 挂入 stream

### 6.3 Async / Sync 边界
- `async fn generate(...) -> impl Stream<Item = Token>`：tokio 异步接口
- 底层 forward 是同步的（GPU stream 隐式异步，但 API 边界是同步 `submit + sync`）
- 用 `spawn_blocking` 把 forward 调用从 tokio runtime 撇出（避免阻塞 reactor）

## 7. 错误处理

- 顶层错误类型：`enum RsllmError { Io, Cuda, Backend, Model, Tokenizer, ... }`
- 公开 API 用 `Result<T, RsllmError>`
- 内部用 `anyhow::Result` 简化，到 API 边界转换
- HTTP server 把 `RsllmError` 映射到 OpenAI 标准错误响应

### 7.1 GPU OOM 处理
- CUDA backend 检测 `CUDA_ERROR_OUT_OF_MEMORY`
- 触发 scheduler 优雅降级：
  - 拒绝新请求（返回 429）
  - 可选 evict 低优先级 session
  - 不允许进程崩溃

## 8. 关键算法 / 数据结构

### 8.1 Paged KV Cache（vLLM 风格）

- Block size：默认 16 token，可配 8/16/32
- 每个 block 是一段连续设备内存
- `BlockTable[session_id] = Vec<BlockId>`
- 分配：free list；回收：session 结束或 evict
- 跨 session 共享：相同 prefix 的 block 引用计数

### 8.2 Radix Tree Prefix Cache

- 节点 = 一段 token 序列 + 对应 KV block IDs
- 插入：插入新 token 序列，分裂或创建子节点
- 查询：最长公共前缀，返回可复用 block 序列
- LRU 淘汰

### 8.3 Disk KV Index

- 主索引：B-tree on `sha1(token_ids[:N])` → `KvcFileId`
- 磁盘空间：LRU，超过 `--kv-disk-space-mb` 淘汰最久未用
- 文件级 mmap，零拷贝上传到 GPU

### 8.4 非对称量化 MoE 加载（ds4 启发）

- GGUF metadata 标注每个 tensor 的量化级
- 加载策略：
  - `^blk\.\d+\.ffn_(gate|up|down)_exps\.weight$` → routed expert，可低位（IQ2/Q2_K）
  - `^blk\.\d+\.ffn_shared\..*` → shared expert，至少 Q4_K
  - 其余（attn / norm / embed / lm_head）→ 高精度（Q8_0 / F16）
- 配置可覆盖

## 9. 安全设计

### 9.1 输入边界
- HTTP 请求 → JSON schema 验证 → 限流 → 鉴权
- Prompt 内容 → 仅经 tokenizer，不直接拼接进 Jinja 模板（防注入）
- 模型路径 → 白名单或限定目录

### 9.2 资源限制
- 单请求最大 token 数
- 全局并发请求数上限
- 每请求时间预算（超时取消）
- KV 内存预算

### 9.3 鉴权
- 可选 API key（环境变量或 config）
- v1.1+ 支持 OIDC / JWT

## 10. 可观测性

### 10.1 指标（Prometheus）
- `rsllm_request_total{method, model, status}`
- `rsllm_request_duration_seconds{method, model}`
- `rsllm_decode_tokens_per_second{model}`
- `rsllm_kv_cache_blocks{state}`（free/used）
- `rsllm_gpu_memory_bytes{device}`
- `rsllm_queue_depth`

### 10.2 Traces（OpenTelemetry）
- root span = HTTP request
- child spans = tokenize / prefill / decode / sample / network

### 10.3 日志
- `tracing` 结构化日志
- 日志级别可按模块调（`RSLLM_LOG=rsllm_server=info,rsllm_cuda=debug`）

## 11. CI / 发布

### 11.1 CI Matrix
- Linux x86_64：CPU + CUDA（自托管 GPU runner）
- macOS arm64：CPU + Metal
- Windows x86_64：CPU only（GPU 等社区）
- Lints：`fmt`, `clippy`, `cargo audit`, `cargo deny`
- 测试：单元 + 集成 + tokenizer 一致性 + 模型 logprob 回归

### 11.2 发布
- semver
- 每个 milestone 发 `0.M.0`
- v1.0 = M0-M5 全部 P0 完成 + 所有 NFR 达标
- GitHub Releases + crates.io 同步

### 11.3 性能回归
- 每周自动跑 benchmark suite
- 与上周比对，回退 > 5% 报警
- 与 llama.cpp / ktransformers 月度对比

## 12. 文件大小 / 编码规范

参考用户全局规则：
- 单文件 ≤ 800 行（典型 200-400）
- 函数 ≤ 50 行
- 嵌套 ≤ 4 层
- 优先不可变模式（Rust 默认 immutable，符合）
- 严格 `clippy -- -D warnings`
- 公开 API 必须 rustdoc + 例子

## 13. 关键里程碑的技术决策时间线（2026-05-14 重定位后）

| 里程碑 | 决策点 |
|---|---|
| **M0 (v0.1.0) ds4 复刻** | **Metal kernel 风格**（直接 port ds4_metal.m vs 借鉴 MLX）、**JoyAI tokenizer 状态机 Rust port 边界**（Unicode 字符分类用 `unicode_categories` crate vs 自查表）、**MLA+HC+MoE 实现策略**（一份代码同时跑 CPU 和 Metal vs 各自特化）、**AVX-512 VNNI 与 NEON dotprod 抽象层**（`is_x86_feature_detected!` runtime 分发 vs cfg 编译期分发） |
| M1 (v0.1.1) AMD Vulkan | `ash` vs `vulkano`、subgroup operations 兼容性矩阵（RDNA 3.5 / RDNA 4 / NVIDIA Ampere+）、shared memory 布局 |
| M2 (v0.1.4) HTTP server | axum tower middleware 链、SSE backpressure、DSML ↔ OpenAI ↔ Anthropic 三向 schema 翻译边界 |
| M3 (v0.1.6) NVIDIA CUDA | cudarc API 边界、Paged KV block size、SSE 流式协议细节 |
| M4 (v0.1.9) TP | TP 实现（按 head 切 vs 按 hidden 切）、NCCL FFI 引入 |
| M5 (v0.2.0) 超越 ds4 | AMX 内核实现策略（内嵌 C 还是纯 Rust intrinsics）、NUMA 抽象、推测解码算法选型、KVC continued checkpoint 触发策略 |

## 14. 与现有引擎的协同 / 互操作

### 14.1 模型权重互通
- 任何 llama.cpp 可读 GGUF rsLLM 应可读（最大兼容）
- 不发明新权重格式

### 14.2 API 兼容
- OpenAI / Anthropic 兼容是双向：rsLLM 既可作为后端，也可作为客户端调用其他服务（v1.1+ 路由器）

### 14.3 Tokenizer 兼容
- 用 HF `tokenizers` crate 加载 `tokenizer.json`
- GGUF 内嵌 tokenizer 元数据可重建

## 15. 未来空间（v1.0 之后）

| 方向 | 概要 |
|---|---|
| 多模态 | Llava / Qwen-VL / InternVL，新加 vision encoder backend |
| 推理时 LoRA | 加载多 LoRA adapter，请求级切换 |
| Distributed inference | 跨节点 TP/PP，gRPC + RDMA |
| Embedding / Reranker | 独立子项目，复用 KV cache 与 backend |
| WASM | 浏览器内推理（小模型） |
| 服务网格 | k8s operator、HPA 基于 KV 压力 |
| 微调集成 | 调用外部 trainer（不内建训练），但提供格式互通 |

## 16. 决策记录（ADR 索引，待初始化）

每个重大决策一份 `docs/adr/NNNN-title.md`：
- ✅ [ADR-0001](adr/0001-engine-architecture.md)：引擎架构 = ds4 风格（自研内核 + GGUF 兼容 + 借鉴致谢，**不**链接 llama.cpp）
- 🟡 ADR-0002：选 cudarc 作为 CUDA 绑定（待写）
- 🟡 ADR-0003：Engine/Session 分离原则（待写）
- 🟡 ADR-0004：KV cache 走 paged + 磁盘持久化（待写）
- 🟡 ADR-0005：异构调度走 cuLaunchHostFunc（待写）
- 🟡 ADR-0006：Tokenizer 复用 HF crate（待写）
- 🟡 ADR-0007：HTTP server 用 axum（待写）
- 🟡 ADR-0008：Metal 绑定用 objc2（待定）

ADR 模板：背景 / 备选方案 / 决策 / 后果 / 重新评审条件。

---

## 附录 A：模块依赖图

```
                         rsllm-cli
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        rsllm-server    rsllm-bench    (Rust lib user)
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                       rsllm-scheduler
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
        rsllm-models   rsllm-kvcache  rsllm-tokenizer
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                         rsllm-core
                             │
                       rsllm-cal (trait only)
                             │
        ┌────────────┬───────┴────┬────────────┐
        ▼            ▼            ▼            ▼
   backend-cpu   backend-cuda  backend-metal  backend-wgpu
                             ▲
                             │
                         rsllm-gguf
```

## 附录 B：性能预期推导（M1 4090 + Llama 3 8B Q4_K_M 为例）

- 模型权重：~4.7 GB
- 单 token 解码 FLOPS：~9.7 GFLOP（8B × 2 × 2 ops）
- 4090 FP16 算力：~165 TFLOPS
- 算力上限解码：~17,000 t/s（理论 roof）
- 实际瓶颈：内存带宽（~1 TB/s），权重读一遍 ≈ 4.7ms → 213 t/s 理论 mem-bound
- llama.cpp 当前：~95 t/s（55% efficiency）
- rsLLM 目标：110-150 t/s（55-70% efficiency，至少持平 llama.cpp 然后逐步往 vLLM 看齐）

## 附录 C：编码规范摘要

- 严格 `cargo fmt` + `clippy -- -D warnings`
- 公开 API rustdoc + doctest
- `unsafe` 块必须注释「不变量」
- FFI 边界单独 crate（`rsllm-backend-cuda` / `rsllm-backend-metal`）
- 每个 backend 必须实现一组共享的 conformance tests（在 `rsllm-cal/tests/conformance.rs`）
- benchmark 用 `criterion`，结果纳入 CI

## 附录 D：开放问题列表

1. 是否实现自有 `arch_prctl` AMX 启用，还是依赖外部辅助？
2. Tokenizer 跨语言一致性测试集如何长期维护？
3. 模型架构碎片化加剧时，是否引入「模型描述 DSL」（YAML/RON）让新模型不写 Rust 也能跑？
4. 跨节点分布式是否需要，何时需要？
5. 是否提供 ONNX / GGML 互转工具？
