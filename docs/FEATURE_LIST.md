# Feature List

> Last Updated: 2026-05-14
> 版本策略：v1.0.0 之前最多走到 0.2.x，每个 patch 版本聚焦小迭代，每个 minor 版本是重大分水岭。
>
> **2026-05-14 重大重定位**：v0.1.0-v0.1.3 改为「**严格复刻 ds4**」路径，v0.1.4+ 开始通用化扩展。
> 首发硬件目标：**Mac (Apple Silicon M3+)** + **AMD Ryzen AI Max+ 395 (Strix Halo) Linux**。
> 首发模型：**DeepSeek V4 Flash** 单模型。
> 详见 [`features/v0.1.0.md#修订记录`](features/v0.1.0.md#修订记录)。
>
> 2026-05-11 早期 review 调整记录（B 类）已被 2026-05-14 重定位覆盖，详见各 feature 的 Status 列。

## 版本路线图（2026-05-14 重定位后）

| 版本 | 对应里程碑 | 主线内容 | 状态 |
|---|---|---|---|
| **v0.1.0** | M0：ds4 复刻第一阶段 | Mac Metal + AMD AVX-512 双硬件跑通 DeepSeek V4 Flash chat | InProgress |
| **v0.1.1** | AMD iGPU 加速 | F033 AMD Vulkan compute backend（Radeon 8060S） | Planned |
| **v0.1.2** | 性能调优 + 数值回归 | 双平台性能基线对齐 ds4 + 全 kernel 数值回归 | Planned |
| **v0.1.3** | 磁盘 KV 简化版 | F022 KVC v2 格式（SHA1(token_ids) + 4 reason）| Planned |
| **v0.1.4** | HTTP server 双协议 | F011 OpenAI + F012 axum + F023 Anthropic + F024 Tool/DSML + F018 metrics | Planned |
| **v0.1.5** | logprob 回归测试 | F034 官方 API 黄金数据 + token bytes 级比对 | Planned |
| **v0.1.6** | NVIDIA CUDA | F009 CUDA backend + F010 Paged KV cache（CUDA 路径用） | Planned |
| **v0.1.7** | 第二个模型家族 | F013 Qwen 3.6 / GLM-5.1 二选一 | Planned |
| **v0.1.8** | 第三个模型家族 | F013 续：Kimi K2.6 / Gemma 4 二选一 | Planned |
| **v0.1.9** | 多 GPU + 收尾 | F015 Tensor Parallel | Planned |
| **v0.2.0** | M3-M4 分水岭 | F019 异构 + F020 AMX/AVX512 算子统一 + F021 历史 DS V2/V3 + F026 推测解码 + F031 KVC 完整版 + F032 Metal 完整版 | Planned |
| **v0.2.x** | 长尾覆盖 | F016 Continuous Batching + F017 Prefix cache + F027 wgpu + F028 JSON mode + F029 多模型控制面 + F030 ROCm | Planned |
| **v1.0.0** | 验收 | 全部 P0 通过、性能达标、CI 矩阵全绿 | Planned |

## Index

| ID | Title | Category | Priority | Version | Status | Design |
|----|-------|----------|----------|---------|--------|--------|
| 001 | Cargo workspace 骨架 + 构建基线 | Internal | Critical | v0.1.0 | Completed | [v0.1.0.md#001](features/v0.1.0.md#feature_001-cargo-workspace-骨架--构建基线) |
| 002 | GGUF 文件解析器（12 种 dequant + FP8 元素级转换） | New | Critical | v0.1.0 | **Phase 4 + F002.1 ✅ Completed** | [v0.1.0.md#002](features/v0.1.0.md#feature_002-gguf-文件解析器) |
| 003 | **JoyAI 状态机分词器（DS V4 vocab，复刻 ds4）** | New | Critical | v0.1.0 | **✅ Completed** | [v0.1.0.md#003](features/v0.1.0.md#feature_003-joyai-状态机分词器ds4-复刻) |
| 004 | **CPU 算子（DS V4 特化 + NEON + AVX-512 双 SIMD）** | New | Critical | v0.1.0 | **✅ Completed**（Phases A-E；VNNI 优化 deferred） | [v0.1.0.md#004](features/v0.1.0.md#feature_004-cpu-算子ds-v4-flash-特化--双-simd) |
| 005 | **DeepSeek V4 Flash 模型架构（MLA + HC + MoE）** | New | Critical | v0.1.0 | Planned | [v0.1.0.md#005](features/v0.1.0.md#feature_005-deepseek-v4-flash-模型架构) |
| 006 | **三级 KV cache（SWA ring + compressed pool + indexer）** | New | Critical | v0.1.0 | Planned | [v0.1.0.md#006](features/v0.1.0.md#feature_006-三级-kv-cache) |
| 007 | 采样器（greedy / temp / top-k / top-p / **min-p** / **think_mode**） | New | High | v0.1.0 | Planned | [v0.1.0.md#007](features/v0.1.0.md#feature_007-采样器) |
| 008 | **CLI（linenoise REPL + 斜杠命令，复刻 ds4）** | New | Critical | v0.1.0 | Planned | [v0.1.0.md#008](features/v0.1.0.md#feature_008-clilinenoise-repl--斜杠命令) |
| 009 | CUDA 后端基础 | New | Critical | **v0.1.6** | Planned | _待 v0.1.6 设计_ |
| 010 | Paged KV cache（vLLM 风格 block 管理） | New | High | **v0.1.6** | Planned | _待 v0.1.6 设计；DS V4 用三级 KV，paged 是 CUDA 路径补充_ |
| 011 | OpenAI Chat Completions API + SSE 流式 | New | Critical | **v0.1.4** | Planned | _待 v0.1.4 设计_ |
| 012 | HTTP server（axum） | New | Critical | **v0.1.4** | Planned | _待 v0.1.4 设计_ |
| 013 | 第二/第三个模型家族（Qwen 3.6 / GLM-5.1 / Kimi K2.6 / Gemma 4） | New | High | **v0.1.7-v0.1.8** | Planned | _待 v0.1.7/8 设计_ |
| 014 | ~~MoE 模型支持（Phi-3.5-MoE）~~ | — | — | — | **Obsoleted** | v0.1.0 已通过 DS V4 Flash 实现 MoE，无需 Phi-3.5-MoE 作为单独 feature |
| 015 | Tensor Parallel（单机多 GPU） | New | High | v0.1.9 | Planned | _待 v0.1.9 设计_ |
| 016 | 连续批处理（Continuous Batching） | New | Critical | **v0.2.x** | Planned | _DS V4 单用户对话不需要 batching，留 v0.2.x_ |
| 017 | Radix tree prefix cache | New | High | **v0.2.x** | Planned | _DS V4 chat 的前缀复用由三级 KV+disk KV 覆盖_ |
| 018 | Prometheus metrics + tracing | New | Medium | **v0.1.4** | Planned | _与 server 一起做_ |
| 019 | cudaLaunchHostFunc 异构调度（CPU+GPU） | New | Critical | v0.2.0 | Planned | [v0.2.0.md#019](features/v0.2.0.md#feature_019-cudalaunchhostfunc-异构调度) |
| 020 | AMX / AVX512 CPU 内核（v0.2.0 统一通用算子） | New | High | v0.2.0 | Planned | [v0.2.0.md#020](features/v0.2.0.md#feature_020-amx--avx512-cpu-内核) |
| 021 | DeepSeek-V2 / V3 历史模型支持 | New | Medium | v0.2.0 | Planned | [v0.2.0.md#021](features/v0.2.0.md#feature_021-deepseek-v2v3-支持) |
| 022 | 磁盘 KV cache 简化版（KVC v2 cold checkpoint） | New | Critical | **v0.1.3** | Planned | _待 v0.1.3 设计_ |
| 023 | Anthropic Messages API | New | High | **v0.1.4** | Planned | _合并到 v0.1.4 server_ |
| 024 | Tool calling（DSML + OpenAI tools + Anthropic tool_use） | New | High | **v0.1.4** | Planned | _合并到 v0.1.4 server_ |
| 025 | **Metal 基础内核（matmul/attn/rope/rmsnorm/Sinkhorn/FP8-KV）** | New | Critical | **v0.1.0** | Planned | [v0.1.0.md#025](features/v0.1.0.md#feature_025-metal-基础内核v010-提前进入mac-主加速路径) |
| 026 | 推测解码（MTP / Eagle / Medusa 之一） | New | Medium | v0.2.0 | Planned | [v0.2.0.md#026](features/v0.2.0.md#feature_026-推测解码) |
| 027 | wgpu 后端（跨平台 GPU 兜底） | New | Low | v0.2.1 | Planned | _待 v0.2.1 设计_ |
| 028 | JSON mode / 结构化输出 | Enhancement | Medium | v0.2.2 | Planned | _待 v0.2.2 设计_ |
| 029 | 多模型 load/unload 控制面 API | New | Medium | v0.2.3 | Planned | _待 v0.2.3 设计_ |
| 030 | ROCm 后端（AMD GPU，best-effort） | New | Low | v0.2.4 | Planned | _AMD AI Max+ iGPU 走 F033 Vulkan，ROCm 留给独立 Radeon dGPU_ |
| 031 | 磁盘 KV cache 完整版（continued + tool call replay） | New | High | v0.2.0 | Planned | _待 v0.2.0 设计_ |
| 032 | Metal 完整版（flash_attn + MoE matvec 优化） | New | High | v0.2.0 | Planned | _待 v0.2.0 设计_ |
| **033** | **AMD Vulkan compute backend（Radeon 8060S iGPU）** | New | Critical | **v0.1.1** | Planned | _待 v0.1.1 设计_ |
| **034** | **logprob 回归测试体系（DeepSeek 官方 API 黄金数据 + token bytes 比对）** | New | Critical | **v0.1.5** | Planned | _待 v0.1.5 设计_ |

## Details

### FEATURE_001: Cargo workspace 骨架 + 构建基线

- **Category**: Internal
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Completed
- **Started**: 2026-05-11
- **Completed**: 2026-05-11
- **Description**: 建立 Rust workspace 项目结构，8 个核心 crate（rsllm-core/gguf/tokenizer/cal/backend-cpu/kvcache/models/cli），rust-toolchain.toml，能通过 `cargo build` 和 `cargo test`
- **Design**: [v0.1.0.md#feature_001](features/v0.1.0.md#feature_001-cargo-workspace-骨架--构建基线)
- **2026-05-14 重定位影响**：v0.1.0 中将**新增 `rsllm-backend-metal` crate**（cfg `target_os = "macos"`），workspace 扩展至 9 crate
- **实现备注**：
  - `cargo build --workspace --release` 通过（30s 冷构建）
  - `cargo test --workspace` 全部通过
  - `cargo clippy --workspace --all-targets -- -D warnings` 零 warning
  - `cargo fmt --check` 通过
  - `rsllm`、`rsllm info`、`rsllm --version` 三个 CLI 调用 smoke-tested 通过
  - 使用 edition = "2024"，rust-version = "1.87"，rust-toolchain = stable

### FEATURE_002: GGUF 文件解析器

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: ✅ **Phase 4 + F002.1 Completed**（2026-05-14）
- **Description**: 自实现 GGUF 解析器，借鉴 ggml/candle/ds4，按 MIT/Apache 致谢
- **Design**: [v0.1.0.md#feature_002](features/v0.1.0.md#feature_002-gguf-文件解析器)
- **已实现**：
  - Phase 4（9 种通用 dequant）：F32 / F16 / BF16 / Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / Q8_0
  - F002.1（DS V4 Flash 必需的 3 种 + FP8）：Q2_K（MoE down）/ IQ2_XXS（MoE gate-up，含 ds4 port 320 行查找表）/ Q8_K（临时激活）/ FP8 E4M3 元素级转换器（KV cache 用）
  - **127 单元 + 2 集成测试全绿，clippy + fmt clean**

### FEATURE_003: JoyAI 状态机分词器（DS V4 vocab，复刻 ds4）

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: 复刻 ds4 的手写 UTF-8 状态机预分词 + BPE 合并 + DS V4 特殊 token（｜begin▁of▁sentence｜等）+ 3 种 think 模式 chat 拼装。**不依赖** HuggingFace tokenizers crate（JoyAI 规则含 negative lookahead，纯 Rust regex 不支持）
- **Design**: [v0.1.0.md#feature_003](features/v0.1.0.md#feature_003-joyai-状态机分词器ds4-复刻)

### FEATURE_004: CPU 算子（DS V4 Flash 特化 + 双 SIMD）

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: 为 DS V4 Flash 推理实现必需 CPU 算子集，**不做通用 GEMM**。SIMD 双路径：Apple Silicon NEON dotprod + AMD Zen 5 AVX-512 VNNI。核心算子包括 Q8_0 batched matmul、Q4_K/Q2_K/IQ2_XXS matmul、RMSNorm、RoPE-YaRN、SwiGLU、Sinkhorn 20 轮迭代
- **Design**: [v0.1.0.md#feature_004](features/v0.1.0.md#feature_004-cpu-算子ds-v4-flash-特化--双-simd)

### FEATURE_005: DeepSeek V4 Flash 模型架构

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: ✅ Completed（attention dot-product + KV cache 留待 F006）
- **Description**: 实现 DS V4 Flash 完整 forward：MLA（Q/KV LoRA + 64 头 × 512 维）+ HC（4 路残差流 + Sinkhorn 20 轮）+ MoE（256 routed + 1 shared，前 3 层 hash 路由，其余层 top-6）+ SwiGLU + RMSNorm + RoPE-YaRN
- **Design**: [v0.1.0.md#feature_005](features/v0.1.0.md#feature_005-deepseek-v4-flash-模型架构)

### FEATURE_006: 三级 KV cache

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: 复刻 ds4 三级 KV：Raw SWA ring（128 token）+ compressed pool（ratio-2/4 池化）+ ratio-4 sparse indexer（top-512）。CPU 路径用 f32，Metal 路径用 FP8 E4M3
- **Design**: [v0.1.0.md#feature_006](features/v0.1.0.md#feature_006-三级-kv-cache)

### FEATURE_007: 采样器

- **Category**: New
- **Priority**: High
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: greedy / temperature / top-k / top-p / min-p（链式过滤器，端口 `ds4.c:14183-14386`）+ think_mode 触发的 prompt 前缀注入（实际拼装在 F003/F008）
- **Design**: [v0.1.0.md#feature_007](features/v0.1.0.md#feature_007-采样器)

### FEATURE_008: CLI（linenoise REPL + 斜杠命令）

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: 复刻 ds4 CLI：交互 REPL（rustyline）+ 斜杠命令（`/help /think /think-max /nothink /ctx /read /system /clear /quit`）+ `inspect` 子命令（模型摘要 + 指纹）+ `info` + 诊断 flag（`--dump-tokens / --dump-logprobs`，JSON schema 与 ds4 兼容）
- **Design**: [v0.1.0.md#feature_008](features/v0.1.0.md#feature_008-clilinenoise-repl--斜杠命令)

### FEATURE_025: Metal 基础内核（v0.1.0 提前进入）

- **Category**: New
- **Priority**: Critical
- **Version**: **v0.1.0**（原 v0.1.6，2026-05-14 重定位后提前）
- **Status**: Planned
- **Description**: Apple Silicon Metal 加速 kernel：matmul (F16/Q4_K/Q2_K/IQ2_XXS)、RMSNorm、RoPE-YaRN、SwiGLU、attention（朴素）、Sinkhorn、FP8 E4M3 KV 量化。新增 `rsllm-backend-metal` crate（cfg target_os = "macos"）。Flash Attention 与 MoE matvec 优化推到 v0.2.0 F032 完整版
- **Design**: [v0.1.0.md#feature_025](features/v0.1.0.md#feature_025-metal-基础内核v010-提前进入mac-主加速路径)

### FEATURE_033: AMD Vulkan compute backend ⭐ 新增

- **Category**: New
- **Priority**: Critical
- **Version**: **v0.1.1**
- **Status**: Planned
- **Description**: AMD Ryzen AI Max+ 395 的 iGPU（Radeon 8060S，40 CU RDNA 3.5）加速。用 **Vulkan compute shader**（不绑 ROCm，跨平台），新增 `rsllm-backend-vulkan` crate。目标：DS V4 Flash decode 从 v0.1.0 CPU 的 ~3 tok/s 提升到 ~10-20 tok/s
- **Design**: 待 v0.1.1 设计
- **依赖**：`ash`（Vulkan Rust binding）或 `vulkano`
- **借鉴**：llama.cpp `ggml-vulkan` kernel（MIT）

### FEATURE_034: logprob 回归测试体系 ⭐ 新增

- **Category**: New
- **Priority**: Critical
- **Version**: **v0.1.5**
- **Status**: Planned
- **Description**: 复刻 ds4 `tests/test-vectors/` 框架——从 DeepSeek 官方 API 拉取 `top_logprobs=20` 黄金数据，存为 fixture，rsllm 用 greedy decoding + `--dump-logprobs` 生成本地输出，token bytes 级（不仅 id）比对。覆盖 short（事实问答 / 代码补全 / 推理）+ long（11-12k token 代码审计）两类
- **Design**: 待 v0.1.5 设计
- **借鉴**：`ds4 tests/test-vectors/`、`fetch_official_vectors.py`、`ds4_test.c:--logprob-vectors`

## Summary

- **Total**: 33 features（+1 自 2026-05-14 重定位：F033, F034 新增；F014 obsoleted；总数 32+2-1=33）
- **InProgress**: 0
- **Phase 4 Completed**: 1（F002）
- **Completed**: 1（F001）
- **Planned**: 30
- **Obsoleted**: 1（F014）

**By Priority**:
- Critical: 17（+2 F033/F034，-1 F014 obsolete = 16+1）
- High: 8
- Medium: 5
- Low: 3

**Next Release (v0.1.0)**: 9 features（F001 ✅, F002 ✅, F003 ✅, F004 ✅, F005 ✅, F006-F008 Planned, F025 Planned）
**Next to Start**: **F006**（三级 KV cache：SWA ring + compressed pool + ratio-4 indexer）

## 2026-05-14 重定位调整记录

详见 [`docs/features/v0.1.0.md#修订记录`](features/v0.1.0.md#修订记录) 与 [`docs/features/v0.1.x-roadmap.md`](features/v0.1.x-roadmap.md)。

| Feature | 原版本 | 新版本 | 备注 |
|---|---|---|---|
| F003 内容 | HF tokenizers crate | **复刻 JoyAI 状态机** | DS V4 需要，HF crate 不支持 lookahead |
| F004 内容 | 通用 CPU 算子 | **DS V4 特化 + NEON + AVX-512 双 SIMD** | 复刻 ds4 + AMD Strix Halo |
| F005 内容 | Llama 2/3 dense | **DeepSeek V4 Flash（MLA + HC + MoE）** | 复刻 ds4 |
| F006 内容 | dense KV | **三级 KV（SWA + compressed + indexer）** | 复刻 ds4 |
| F008 内容 | clap 子命令 | **linenoise REPL + 斜杠命令** | 复刻 ds4 |
| F025 Metal | v0.1.6 | **v0.1.0**（提前） | Mac 必需加速 |
| F009 CUDA | v0.1.1 | **v0.1.6** | 复刻完 ds4 再扩展到 CUDA |
| F010 Paged KV | v0.1.1 | **v0.1.6** | DS V4 用三级 KV，Paged 是 CUDA 路径补充 |
| F011/F012 OpenAI server | v0.1.2 | **v0.1.4** | 复刻 ds4-server，合并 Anthropic + Tool |
| F013 多模型 | v0.1.3 | **v0.1.7-v0.1.8** | v0.1.0-v0.1.6 都聚焦 DS V4 |
| F014 Phi-3.5-MoE | v0.1.4 | **Obsoleted** | DS V4 已实现 MoE，无需 Phi-3.5-MoE 作为单独 feature |
| F015 TP | v0.1.9 | v0.1.9（不变） | — |
| F016 Continuous Batching | v0.1.5 | **v0.2.x** | DS V4 单用户对话不需要 |
| F017 Prefix cache | v0.1.7 | **v0.2.x** | 三级 KV+disk KV 已覆盖 chat 场景 |
| F018 Prometheus | v0.1.9 | **v0.1.4** | 合并到 server |
| F022 磁盘 KV 简化版 | v0.1.8 | **v0.1.3** | ds4 核心特性，靠前 |
| F023 Anthropic API | v0.2.0 | **v0.1.4** | 合并到 server |
| F024 Tool/DSML | v0.2.0 | **v0.1.4** | 合并到 server |
| **F033 AMD Vulkan**（新） | — | **v0.1.1** | AMD AI Max+ iGPU 加速 |
| **F034 logprob 测试体系**（新） | — | **v0.1.5** | 复刻 ds4 测试框架 |
