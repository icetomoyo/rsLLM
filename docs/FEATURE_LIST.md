# Feature List

> Last Updated: 2026-05-14
> 版本策略：v1.0.0 之前最多走到 0.2.x，每个 patch 版本聚焦小迭代，每个 minor 版本是重大分水岭。
>
> **2026-05-11 review 调整**：F014 Mixtral→Phi-3.5-MoE；F015/F016 对调；Metal 基础版（F025）从 v0.2.0 提前到 v0.1.6；磁盘 KV 简化版（F022）从 v0.2.0 提前到 v0.1.8；F031/F032 新增为 v0.2.0 完整版占位。

## 版本路线图

| 版本 | 对应里程碑 | 主线内容 | 状态 |
|---|---|---|---|
| **v0.1.0** | M0 骨架 | CPU 后端 + GGUF + Tokenizer + Llama 模型 + CLI 基线 | InProgress |
| **v0.1.1** | CUDA 启动 | F009 CUDA 后端 + F010 Paged KV cache | Planned |
| **v0.1.2** | 服务化 | F011 OpenAI Chat API + F012 axum HTTP server | Planned |
| **v0.1.3** | 多模型 | F013 Qwen / Mistral / Phi 支持 | Planned |
| **v0.1.4** | MoE 试水 | F014 Phi-3.5-MoE（含 IQ2_XXS dequant 前置补齐） | Planned |
| **v0.1.5** | 连续批处理 | F016 Continuous Batching | Planned |
| **v0.1.6** | Mac 第一波 | F025 Metal 基础内核（matmul/attn/rope/rmsnorm） | Planned |
| **v0.1.7** | Prefix 复用 | F017 Radix tree prefix cache | Planned |
| **v0.1.8** | Agent 基础 | F022 磁盘 KV cache 简化版（cold checkpoint） | Planned |
| **v0.1.9** | 多 GPU | F015 Tensor Parallel + F018 Prometheus metrics | Planned |
| **v0.2.0** | M3-M4 分水岭 | F019 异构 + F020 AMX + F021 DeepSeek-V3 + F023 Anthropic + F024 Tool + F026 推测解码 + **F031 磁盘 KV 完整版** + **F032 Metal 完整版** | Planned |
| **v0.2.x** | M5 + 完善 | F027 wgpu + F028 JSON mode + F029 多模型控制面 + F030 ROCm | Planned |
| **v1.0.0** | 验收 | 全部 P0 通过、性能达标、CI 矩阵全绿 | Planned |

## Index

| ID | Title | Category | Priority | Version | Status | Design |
|----|-------|----------|----------|---------|--------|--------|
| 001 | Cargo workspace 骨架 + 构建基线 | Internal | Critical | v0.1.0 | Completed | [v0.1.0.md#001](features/v0.1.0.md#feature_001-cargo-workspace-骨架--构建基线) |
| 002 | GGUF 文件解析器（自实现，借鉴 ggml/candle） | New | Critical | v0.1.0 | Completed | [v0.1.0.md#002](features/v0.1.0.md#feature_002-gguf-文件解析器) |
| 003 | Tokenizer 集成（HuggingFace tokenizers + chat template） | New | Critical | v0.1.0 | Planned | [v0.1.0.md#003](features/v0.1.0.md#feature_003-tokenizer-集成) |
| 004 | CPU 后端基础算子（matmul/rmsnorm/rope/softmax） | New | Critical | v0.1.0 | Planned | [v0.1.0.md#004](features/v0.1.0.md#feature_004-cpu-后端基础算子) |
| 005 | Llama 模型架构（dense, GQA, RoPE） | New | Critical | v0.1.0 | Planned | [v0.1.0.md#005](features/v0.1.0.md#feature_005-llama-模型架构) |
| 006 | KV cache 基础（dense layout） | New | Critical | v0.1.0 | Planned | [v0.1.0.md#006](features/v0.1.0.md#feature_006-kv-cache-基础) |
| 007 | 采样器（greedy / temperature / top-k / top-p） | New | High | v0.1.0 | Planned | [v0.1.0.md#007](features/v0.1.0.md#feature_007-采样器) |
| 008 | CLI 入口（rsllm chat / rsllm run / rsllm info） | New | Critical | v0.1.0 | Planned | [v0.1.0.md#008](features/v0.1.0.md#feature_008-cli-入口) |
| 009 | CUDA 后端基础 | New | Critical | v0.1.1 | Planned | _待 v0.1.1 设计_ |
| 010 | Paged KV cache（vLLM 风格 block 管理） | New | Critical | v0.1.1 | Planned | _待 v0.1.1 设计_ |
| 011 | OpenAI Chat Completions API + SSE 流式 | New | Critical | v0.1.2 | Planned | _待 v0.1.2 设计_ |
| 012 | HTTP server（axum） | New | Critical | v0.1.2 | Planned | _待 v0.1.2 设计_ |
| 013 | Qwen / Mistral / Phi 模型支持 | New | High | v0.1.3 | Planned | _待 v0.1.3 设计_ |
| 014 | **MoE 模型支持（Phi-3.5-MoE，4090 可装下）** | New | High | v0.1.4 | Planned | _待 v0.1.4 设计；前置：F002 IQ2_XXS dequant 补齐_ |
| 015 | Tensor Parallel（单机多 GPU） | New | High | **v0.1.9** | Planned | _待 v0.1.9 设计_ |
| 016 | 连续批处理（Continuous Batching） | New | Critical | **v0.1.5** | Planned | _待 v0.1.5 设计_ |
| 017 | Radix tree prefix cache | New | High | v0.1.7 | Planned | _待 v0.1.7 设计_ |
| 018 | Prometheus metrics + tracing | New | Medium | **v0.1.9** | Planned | _待 v0.1.9 设计_ |
| 019 | cudaLaunchHostFunc 异构调度（CPU+GPU） | New | Critical | v0.2.0 | Planned | [v0.2.0.md#019](features/v0.2.0.md#feature_019-cudalaunchhostfunc-异构调度) |
| 020 | AMX / AVX512 CPU 内核 | New | High | v0.2.0 | Planned | [v0.2.0.md#020](features/v0.2.0.md#feature_020-amx--avx512-cpu-内核) |
| 021 | DeepSeek-V2/V3 支持（MLA + MoE） | New | Critical | v0.2.0 | Planned | [v0.2.0.md#021](features/v0.2.0.md#feature_021-deepseek-v2v3-支持) |
| 022 | **磁盘 KV cache 简化版（cold checkpoint only）** | New | Critical | **v0.1.8** | Planned | _待 v0.1.8 设计_ |
| 023 | Anthropic Messages API | New | High | v0.2.0 | Planned | [v0.2.0.md#023](features/v0.2.0.md#feature_023-anthropic-messages-api) |
| 024 | Tool calling（OpenAI + Anthropic 双协议） | New | High | v0.2.0 | Planned | [v0.2.0.md#024](features/v0.2.0.md#feature_024-tool-calling) |
| 025 | **Metal 基础内核（matmul/attn/rope/rmsnorm）** | New | Critical | **v0.1.6** | Planned | _待 v0.1.6 设计_ |
| 026 | 推测解码（MTP / Eagle / Medusa 之一） | New | Medium | v0.2.0 | Planned | [v0.2.0.md#026](features/v0.2.0.md#feature_026-推测解码) |
| 027 | wgpu 后端（跨平台 GPU 兜底） | New | Low | v0.2.1 | Planned | _待 v0.2.1 设计_ |
| 028 | JSON mode / 结构化输出 | Enhancement | Medium | v0.2.2 | Planned | _待 v0.2.2 设计_ |
| 029 | 多模型 load/unload 控制面 API | New | Medium | v0.2.3 | Planned | _待 v0.2.3 设计_ |
| 030 | ROCm 后端（AMD GPU，best-effort） | New | Low | v0.2.4 | Planned | _待 v0.2.4 设计_ |
| **031** | **磁盘 KV cache 完整版（continued + tool call replay）** | New | High | v0.2.0 | Planned | _待 v0.2.0 设计_ |
| **032** | **Metal 完整版（flash_attn + MoE matvec + FP8 KV）** | New | High | v0.2.0 | Planned | _待 v0.2.0 设计_ |

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
- **实现备注**：
  - `cargo build --workspace --release` 通过（30s 冷构建）
  - `cargo test --workspace` 全部通过（每 crate 1-2 个占位测试）
  - `cargo clippy --workspace --all-targets -- -D warnings` 零 warning
  - `cargo fmt --check` 通过
  - `rsllm`、`rsllm info`、`rsllm --version` 三个 CLI 调用 smoke-tested 通过
  - 使用 edition = "2024"，rust-version = "1.85"，rust-toolchain = stable

### FEATURE_002: GGUF 文件解析器

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: 自实现 GGUF 解析器，借鉴 ggml/candle，按 MIT/Apache 致谢
- **Design**: [v0.1.0.md#feature_002](features/v0.1.0.md#feature_002-gguf-文件解析器)

### FEATURE_003: Tokenizer 集成

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: 封装 HuggingFace tokenizers crate，加 chat template 渲染
- **Design**: [v0.1.0.md#feature_003](features/v0.1.0.md#feature_003-tokenizer-集成)

### FEATURE_004: CPU 后端基础算子

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: matmul、rmsnorm、rope、softmax、量化 dequant，AVX2/NEON SIMD 加速
- **Design**: [v0.1.0.md#feature_004](features/v0.1.0.md#feature_004-cpu-后端基础算子)

### FEATURE_005: Llama 模型架构

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: Llama 2/3 dense 架构实现，GQA、RoPE、SwiGLU
- **Design**: [v0.1.0.md#feature_005](features/v0.1.0.md#feature_005-llama-模型架构)

### FEATURE_006: KV cache 基础

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: Dense 连续 KV cache，单 session
- **Design**: [v0.1.0.md#feature_006](features/v0.1.0.md#feature_006-kv-cache-基础)

### FEATURE_007: 采样器

- **Category**: New
- **Priority**: High
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: greedy / temperature / top-k / top-p / repetition penalty
- **Design**: [v0.1.0.md#feature_007](features/v0.1.0.md#feature_007-采样器)

### FEATURE_008: CLI 入口

- **Category**: New
- **Priority**: Critical
- **Version**: v0.1.0
- **Status**: Planned
- **Description**: `rsllm chat -m model.gguf`, `rsllm run -p "..."`, `rsllm info` 系统能力探测
- **Design**: [v0.1.0.md#feature_008](features/v0.1.0.md#feature_008-cli-入口)

## Summary

- **Total**: 32 features（**+2** 自 2026-05-11 review）
- **InProgress**: 0
- **Planned**: 31
- **Completed**: 1

**By Priority**:
- Critical: 15
- High: 9
- Medium: 5
- Low: 3

**Next Release (v0.1.0)**: 8 features (1 Completed, 7 Planned)
**Next to Start**: F002 (GGUF 解析器)

## 2026-05-11 Review 调整记录

详见 [`docs/features/v0.1.0.md#修订记录`](features/v0.1.0.md#修订记录)。本次调整不改 feature ID，只调整版本归属：

| Feature | 原版本 | 新版本 | 备注 |
|---|---|---|---|
| F014 MoE | v0.1.4（Mixtral） | v0.1.4（**改 Phi-3.5-MoE**） | 4090 物理装不下 Mixtral 8x7B Q4_K_M (26GB) |
| F015 TP | v0.1.5 | **v0.1.9** | 没有连续批处理时 TP 收益有限 |
| F016 连续批处理 | v0.1.6 | **v0.1.5** | 单 GPU 服务化先获益 |
| F018 metrics | v0.1.8 | **v0.1.9** | 跟随 F015 |
| F022 磁盘 KV | v0.2.0 | **v0.1.8 简化版** + v0.2.0 完整版 | agent 工作流是核心差异化，不能拖到 v0.2.0 |
| F025 Metal | v0.2.0 | **v0.1.6 基础版** + v0.2.0 完整版 | Mac 用户痛点，CUDA 验证 CAL 后跟进 |
| F031（新增） | — | v0.2.0 | 磁盘 KV 完整版（continued + tool call replay） |
| F032（新增） | — | v0.2.0 | Metal 完整版（flash_attn + MoE matvec + FP8 KV） |
