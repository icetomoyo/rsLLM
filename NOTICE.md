# NOTICE

rsLLM
Copyright (c) 2026 The rsLLM Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not obtain a copy of this product except in compliance with the License.
A copy of the License is provided in the [`LICENSE`](LICENSE) file.

---

## 致谢与依赖声明

rsLLM 是一个独立的 Rust 推理引擎，**不**链接以下任何项目的运行时。但出于工程致谢和源码层面的代码借鉴，我们必须显式声明这些项目的贡献。

本文件遵循 ds4.c 项目所树立的范例：**借鉴 ≠ 链接，致谢 ≠ 依赖**。

### 一、源码级借鉴（按 MIT/Apache 规则保留双重署名）

下列项目的源代码片段、算法实现、数据表常量，在 rsLLM 源代码中按 MIT/Apache 规则进行了借鉴或改编。对应源文件 header 会同时署名 rsLLM authors 与原作者。

#### 🌟 llama.cpp / ggml

- **项目**：https://github.com/ggml-org/llama.cpp
- **License**：MIT
- **Copyright**：The ggml authors, 2023-present
- **借鉴内容**：
  - GGUF 文件格式（magic、metadata schema、tensor layout）
  - 量化 block 内存布局：Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ2_XXS, IQ3_XXS 等
  - 量化解码查找表常量
  - CUDA kernel 算法参考：`ggml-cuda.cu` 中的 matmul / attention / rope / norm / quant dequant 等
  - Metal kernel 算法参考：`ggml-metal.m`
  - CPU SIMD kernel 思路
  - 跨平台编译与硬件特化经验

#### 🌟 ds4.c

- **项目**：https://github.com/antirez/ds4
- **License**：MIT
- **Copyright**：The ds4.c authors (2026)
- **借鉴内容**：
  - 整体工程姿势：「格式兼容但代码独立」（"does not link against GGML"）
  - 磁盘 KV cache 一等公民设计与 KVC 文件格式
  - 非对称量化策略：只压 MoE routed experts，保留其他权重精度
  - FP8 (E4M3FN) KV cache 量化：64-element block round-trip
  - Engine / Session 分离的 API 边界
  - 官方 logprob 回归测试方法论
  - DSML 工具调用格式（DeepSeek 系适配时参考）
  - ARM NEON CPU kernel 思路

#### 🌟 ktransformers

- **项目**：https://github.com/kvcache-ai/ktransformers
- **License**：Apache 2.0
- **Copyright**：MADSys Lab, Tsinghua University 等
- **借鉴内容**：
  - `cudaLaunchHostFunc` 异构调度模式
  - NUMA 感知线程池设计
  - AMX / AVX512 内核结构（CRTP → Rust trait 单态化）
  - 动态 GPU expert mask（按 activation frequency EMA 重路由）
  - CPU 变体启动期探测策略

#### 🌟 candle

- **项目**：https://github.com/huggingface/candle
- **License**：Apache 2.0 / MIT
- **Copyright**：HuggingFace
- **借鉴内容**：
  - GGUF 解析的 Rust 表达（`candle-core/src/quantized/gguf_file.rs`）
  - Llama / Mistral 等模型的 Rust 实现思路

#### 🌟 FlashAttention

- **项目**：https://github.com/Dao-AILab/flash-attention
- **License**：BSD-3-Clause
- **Copyright**：Tri Dao 等
- **借鉴内容**：FlashAttention v1 / v2 / v3 算法

#### 🌟 MLX

- **项目**：https://github.com/ml-explore/mlx
- **License**：MIT
- **Copyright**：Apple
- **借鉴内容**：Apple Silicon Metal 优化模式（用于 rsLLM Metal 后端参考）

#### 🌟 vLLM

- **项目**：https://github.com/vllm-project/vllm
- **License**：Apache 2.0
- **借鉴内容**：Paged Attention 设计、Continuous Batching 调度模式

---

### 二、运行时依赖（Cargo crate 直接依赖）

> 完整列表会在 v1.0 发布前列出。早期规划阶段已确定使用的关键 crate：

| Crate | License | 用途 |
|---|---|---|
| [tokio](https://github.com/tokio-rs/tokio) | MIT | Async 运行时 |
| [axum](https://github.com/tokio-rs/axum) | MIT | HTTP server |
| [tokenizers](https://github.com/huggingface/tokenizers) | Apache 2.0 | Tokenizer 实现 |
| [cudarc](https://github.com/coreylowman/cudarc) | MIT/Apache | CUDA Rust 绑定 |
| [objc2](https://github.com/madsmtm/objc2) | MIT | Metal Rust 绑定 |
| [memmap2](https://github.com/RazrFalcon/memmap2-rs) | MIT/Apache | mmap 抽象 |
| [serde](https://github.com/serde-rs/serde) | MIT/Apache | 序列化 |
| [tracing](https://github.com/tokio-rs/tracing) | MIT | 结构化日志 |
| [clap](https://github.com/clap-rs/clap) | MIT/Apache | CLI 解析 |
| [prometheus](https://github.com/tikv/rust-prometheus) | Apache 2.0 | Metrics |

完整依赖与 license 在 `cargo deny` CI 中持续校验。

---

### 三、模型权重 license 声明

rsLLM 是推理引擎，**不**对任何模型权重的 license 负责。

不同模型有不同的使用条款：

| 模型 | License | 商业用条款 |
|---|---|---|
| Llama 3.x | Meta Llama Community License | 月活 > 7 亿用户需单独申请 |
| Llama 2 | Llama 2 Community License | 同上 |
| Qwen 2.5 | Apache 2.0（多数版本） | 自由 |
| Mistral 系 | Apache 2.0 / MRL（分版本） | 视版本 |
| Mixtral | Apache 2.0 | 自由 |
| DeepSeek 系 | DeepSeek License | 类 MIT + 道德条款 |
| Gemma 2 | Gemma Terms | 有使用限制 |
| Phi-3.5 | MIT | 自由 |
| Yi | Yi Series Model License | 商业要登记 |

**用户在使用 rsLLM 运行任何模型前，必须自行阅读并遵守该模型的 license。** rsLLM 不对模型权重的使用授权负任何责任。

---

### 四、致谢精神

引用 ds4.c 的话：

> *"This project would not exist without the path opened by llama.cpp."*

rsLLM 同样可以说：

> **rsLLM would not exist without ds4's worked example of "format-compatible, code-independent", without llama.cpp's GGUF ecosystem, without ktransformers' heterogeneous scheduling insight, and without every other contributor who pushed open-source LLM inference forward.**

向所有上游作者和贡献者致以最深的敬意。

---

### 五、变更条款

如果你认为 rsLLM 错误使用了你的项目的代码或声明，请通过 GitHub Issue 联系我们，我们会立即修正。
