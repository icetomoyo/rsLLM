# rsLLM

**Rust-native LLM 推理引擎，跨硬件等级、跨模型家族、跨操作系统。**

> 设计目标：在 16GB 笔记本上跑得动，在 8×H100 集群上跑得快——同一个二进制，同一套 API。

## 当前状态

🚧 **规划阶段**（pre-M0）：仓库已完成产品画像、PRD、HLD、ADR-0001 引擎架构决策。代码尚未开始，预计 v1.0 时间窗口 12-18 个月。

## 引擎架构（已决）

详见 [ADR-0001](docs/adr/0001-engine-architecture.md)。

- **自研 Rust + CUDA + Metal + SIMD 内核**，不链接 llama.cpp / ggml
- **GGUF 格式兼容**：用户任意 llama.cpp GGUF 文件可直接加载
- **借鉴而非从零**：CUDA / Metal kernels 显式借鉴 llama.cpp / ds4 / candle / FlashAttention / MLX 等开源实现，按 MIT/Apache 规则致谢
- **差异化**：磁盘 KV cache、连续批处理、异构 CPU+GPU MoE 卸载、Radix tree prefix cache、Anthropic + OpenAI API 双兼容

## 设计文档

完整设计文档树位于 [`docs/`](docs/)：

| 文档 | 内容 |
|---|---|
| [`docs/00-overview.md`](docs/00-overview.md) | 项目愿景、战略支柱、调研收获 |
| [`docs/01-product-profile.md`](docs/01-product-profile.md) | 产品画像、目标用户、使用场景、竞品对照 |
| [`docs/02-PRD.md`](docs/02-PRD.md) | 详细功能需求、非功能需求、验收标准 |
| [`docs/03-HLD.md`](docs/03-HLD.md) | 高层架构、Cargo workspace、关键 trait、技术选型 |
| [`docs/adr/`](docs/adr/) | 架构决策记录（ADR）|
| [`docs/research/`](docs/research/) | ktransformers / ds4 深度调研 |

## Acknowledgements 致谢

rsLLM 站在巨人的肩膀上。本项目从设计到内核实现都显式借鉴以下开源项目，向其作者和贡献者致以最深的敬意：

### 🌟 [llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml) — Georgi Gerganov 等

**没有 llama.cpp 就没有今天的开源 LLM 推理生态。** 我们借鉴：

- **GGUF 文件格式**——事实标准，我们自实现解析器但完整兼容
- **量化方案**：Q2_K / Q4_K / Q5_K / Q6_K / Q8_0 / IQ2_XXS 等 block 内存布局与查找表
- **CUDA kernels**：`ggml-cuda.cu` 的众多算子（matmul / attention / norm / rope / quant dequant）作为我们 CUDA 后端的参考实现
- **Metal kernels**：`ggml-metal.m` 同上，作为 Metal 后端的参考实现
- **跨平台编译思路**与无数硬件特化的工程经验

按 MIT 规则，复用代码片段的源文件 header 同时署名 `The ggml authors`。

### 🌟 [ds4.c](https://github.com/antirez/ds4) — Salvatore Sanfilippo (antirez)

**ds4 开辟了「格式兼容但代码独立」这条路。** rsLLM 的整体工程姿势直接继承自 ds4，并借鉴了多个具体设计：

- **不链接 llama.cpp、但借鉴 GGUF 格式**的工程纪律——`ds4.c` README 的原话 "does not link against GGML, but exists thanks to the path opened by llama.cpp" 也是 rsLLM 的姿势
- **磁盘 KV cache 作为一等公民**——KVC 文件格式启发我们的设计
- **非对称量化**：仅压缩 MoE routed experts，保留其他权重精度
- **FP8 (E4M3FN) KV cache**：64-element block round-trip
- **Engine / Session 分离**的干净 API 边界
- **官方 logprob 回归测试**作为质量门
- **DSML 工具调用格式**（DeepSeek 系适配时参考）

按 MIT 规则，复用代码片段的源文件 header 同时署名 `The ds4 authors`。

### 🌟 [ktransformers](https://github.com/kvcache-ai/ktransformers) — 清华 MADSys 实验室等

**ktransformers 证明了异构 CPU+GPU MoE 推理在消费级硬件上的可行性。** 我们借鉴：

- **`cudaLaunchHostFunc` 异构调度**：让 CPU 工作挂上 CUDA stream
- **NUMA 感知线程池**：双路 Xeon 必备
- **AMX / AVX512 内核结构**
- **动态 GPU expert mask**：根据 activation frequency 重路由
- **CPU 变体启动期探测**：AMX > AVX512+BF16 > … > AVX2

### 其他致谢

- **[candle](https://github.com/huggingface/candle)** (HuggingFace, Apache 2.0)：Rust 中 GGUF 解析与 Llama 等模型实现的优雅表达
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** (Tri Dao 等, BSD-3)：FlashAttention v1/v2/v3 算法
- **[MLX](https://github.com/ml-explore/mlx)** (Apple, MIT)：Apple Silicon Metal 优化模式
- **[vLLM](https://github.com/vllm-project/vllm)** (Apache 2.0)：Paged Attention、Continuous Batching 设计
- **[tokenizers](https://github.com/huggingface/tokenizers)** (HuggingFace, Apache 2.0)：直接复用 crate
- **[cudarc](https://github.com/coreylowman/cudarc)** (MIT/Apache)：CUDA Rust 绑定

完整依赖与致谢见 [`NOTICE.md`](NOTICE.md)。

## License

Apache-2.0

借鉴源码片段的部分按其原 license 同时保留版权署名，详见 [`NOTICE.md`](NOTICE.md) 和各源文件 header。
