# rsLLM 产品画像（Product Profile）

> 本文回答四个问题：**为谁做？做什么？在哪个场景？相对谁更值？**

---

## 1. 一句话产品定义

**rsLLM 是一个 Rust 原生的开源 LLM 推理引擎，目标是「跨硬件等级、跨模型家族、跨操作系统」三跨——让一个二进制能在 8GB 笔记本到 8×H100 集群上提供同一套 API 与体验。**

## 2. 目标用户群（Personas）

### Persona A：**单机自托管开发者「老王」**
- **画像**：30 岁后端工程师，家里 RTX 4090 + 64GB RAM，想跑 Qwen 32B / Llama 70B 给自己的 IDE 插件、笔记软件用
- **痛点**：
  - llama.cpp 性能够但调度落后，长上下文掉得快
  - ollama 易用但黑盒、性能不可调
  - vLLM 性能好但要 Python 环境 + 重，单机过头
- **rsLLM 给他**：
  - 单 Rust 二进制 + GGUF 模型，开机即用
  - OpenAI API 兼容，IDE 插件零修改
  - 性能在 llama.cpp 之上，资源占用比 vLLM 轻

### Persona B：**轻量边缘部署工程师「小李」**
- **画像**：嵌入式/桌面应用开发者，要在客户 Windows / macOS 设备上集成 LLM 能力，硬件可能只是 16GB iGPU 笔记本
- **痛点**：
  - llama.cpp Windows 构建链长、AVX 检测出过事故
  - 跨平台分发 Python 推理栈是地狱
  - 客户设备五花八门：Intel iGPU、AMD APU、Apple Silicon、老 NVIDIA
- **rsLLM 给他**：
  - 单二进制（< 50MB），按 feature 编译目标平台后端
  - 启动期 capability 探测，自动选最优后端
  - 量化模型 4-7B 上 15+ t/s 体验

### Persona C：**异构工作站 power user「Dr. 张」**
- **画像**：高校 / 中小厂 AI 研究员，单机配置「双路 Xeon AMX + 4090 + 512GB DRAM」，想跑 DeepSeek-V3 671B
- **痛点**：
  - ktransformers 是唯一可用方案，但 Python 栈难维护、Linux only、AMX 配置复杂
  - 一旦出问题难以 debug
- **rsLLM 给他**：
  - CPU/GPU 混合调度 + AMX/AVX512 内核
  - 完整 Rust 错误链 + tracing
  - 关键时刻可以读源码、可以打 patch

### Persona D：**小规模服务集群运维「老赵」**
- **画像**：创业团队 SRE，4-8 张 A100/H100，要给内部业务提供推理 API
- **痛点**：
  - vLLM/SGLang 跑得动但内存抖动、长尾延迟控制困难
  - 模型切换要重启服务，磁盘 KV 不持久
  - Python GIL + asyncio 在高并发下偶发 bug
- **rsLLM 给他**：
  - 多 GPU TP + 连续批处理
  - 磁盘 KV cache + prefix cache 提速 agent 场景
  - Rust 内存安全 + tokio 并发模型，长尾抖动小

### Persona E：**Mac 创作者「设计师阿康」**
- **画像**：Mac Studio M3 Ultra 512GB 用户，想跑大模型做创意写作 / 代码助手
- **痛点**：
  - llama.cpp Metal 后端可用但落后于 MLX / ds4
  - ds4 只支持一个模型
  - LM Studio / Ollama 是黑盒
- **rsLLM 给他**：
  - Metal 后端，跑 70B+ 模型流畅
  - 多模型切换，不锁定 DeepSeek
  - Anthropic Messages API 兼容（可接 Claude Code）

## 3. 不是给谁的（Anti-Personas）

| 群体 | 为什么不是 |
|---|---|
| 模型训练 / 微调研究员 | 推理引擎不做训练 |
| 浏览器 / WASM 部署需求 | 早期不优先支持 |
| 必须 Python 集成的 ML pipeline | 走 HTTP API，不提供 Python C 扩展 |
| 极致延迟敏感（< 5ms）实时业务 | LLM 单 token 延迟下限就在那 |
| Embedding / 检索 / 重排 | 不同赛道，复杂度不匹配 |
| Vision LLM 重度用户（v1 之前） | 初期单模态文本优先 |

## 4. 核心使用场景（Use Cases）

按重要性排序：

### 场景 1：**单机 OpenAI 替代** ⭐⭐⭐⭐⭐
> 「我家里 4090，跑个本地 LLM 给 IDE / 笔记软件用，要 OpenAI API 兼容」

- 关键流程：下载 GGUF → `rsllm serve model.gguf` → IDE 切换 base_url
- 关键指标：启动时间 < 5s，首 token < 200ms，30+ t/s

### 场景 2：**异构 CPU+GPU 跑超大模型** ⭐⭐⭐⭐
> 「我有双路 Xeon + 4090，想跑 DeepSeek-V3 671B」

- 关键流程：YAML 配置 → 启动时自动 NUMA 绑定 → cold experts on CPU
- 关键指标：decode 15+ t/s，单卡 VRAM 占用 < 24GB

### 场景 3：**Agent / Coding 工作流** ⭐⭐⭐⭐
> 「我是 Claude Code / opencode 用户，想本地化大部分查询」

- 关键流程：磁盘 KV cache 启用 → 25k token 系统提示一次 prefill → 后续请求复用
- 关键指标：磁盘 KV 命中时首 token < 100ms

### 场景 4：**多模型 / 模型切换服务** ⭐⭐⭐
> 「我团队需要同时跑 Qwen / Llama / DeepSeek 给不同业务」

- 关键流程：模型仓库目录 → REST 控制面 load/unload → 自动 LRU
- 关键指标：模型切换 < 30s，并行加载支持 ≥2 模型

### 场景 5：**Mac 创作者本地 LLM** ⭐⭐⭐
> 「Mac Studio 跑 70B 模型，用 Claude Code 接口」

- 关键流程：单二进制下载 → 启动 Metal 后端 → 接 Anthropic API
- 关键指标：70B Q4 上 20+ t/s

### 场景 6：**嵌入式 / 桌面应用集成** ⭐⭐⭐
> 「我做一个桌面 App，要内置一个小模型」

- 关键流程：作为 Rust crate `rsllm-core` 嵌入 → 直接 in-process 调用
- 关键指标：crate size < 5MB，启动 < 1s

### 场景 7：**小集群推理服务** ⭐⭐
> 「4 张 A100，给团队 50 人提供 API」

- 关键流程：TP=4 → 连续批处理 → Prometheus metrics
- 关键指标：QPS 在 vLLM 的 0.9× 以上

## 5. 竞品对照

| 维度 | llama.cpp | ollama | vLLM | ktransformers | ds4 | **rsLLM** |
|---|---|---|---|---|---|---|
| 语言 | C/C++ | Go+llama.cpp | Python+CUDA | Python+C++ | C | **Rust+C/C++/CUDA** |
| 内存安全 | ✗ | 部分 | ✗ | ✗ | ✗ | **✓** |
| 跨平台 | ✓ | ✓ | Linux 主 | Linux 主 | macOS only | **✓** |
| GGUF 支持 | ✓ | ✓ | 部分 | ✓ | 部分 | **✓** |
| CUDA | ✓ | ✓ | ✓ | ✓ | ✗ | **✓** |
| Metal | ✓ | ✓ | ✗ | ✗ | ✓ | **✓**（M5） |
| AMX/AVX512 | 部分 | 借 llama.cpp | ✗ | ✓ | ✗ | **✓**（M3） |
| 异构 CPU/GPU | 基础 layer split | 同左 | ✗ | **强** | ✗ | **✓**（M3） |
| MoE 卸载 | 简单 | 同左 | ✗ | **强** | 单模型 | **✓**（M3） |
| 连续批处理 | 部分 | ✗ | **强** | 部分 | ✗ | **✓**（M2） |
| Paged Attention | ✗ | ✗ | ✓ | ✗ | 等价 | **✓**（M1） |
| 磁盘 KV 持久化 | ✗ | ✗ | ✗ | ✗ | **✓** | **✓**（M4） |
| Prefix Cache | ✗ | ✗ | 部分 | ✗ | 隐式 | **✓**（M4） |
| OpenAI API | 第三方 | ✓ | ✓ | ✓ | ✓ | **✓**（M1） |
| Anthropic API | ✗ | ✗ | ✗ | ✗ | ✓ | **✓**（M4） |
| 工具调用 | ✗ | 部分 | ✓ | 部分 | ✓ | **✓**（M4） |
| 推测解码 | ✓ | ✗ | ✓ | ✗ | ✓ (MTP) | **✓**（M5） |
| 多模型并存 | ✗ | ✓ | 部分 | ✗ | ✗ | **✓**（M2） |
| 多 GPU TP | 基础 | 借 llama.cpp | **强** | ✓ | ✗ | **✓**（M2） |
| Python 依赖 | ✗ | ✗ | **重** | **重** | ✗ | **✗** |

**rsLLM 的差异化优势矩阵**：
1. **唯一同时具备**「跨平台 + 多后端 + 多模型 + 异构 CPU/GPU + 磁盘 KV + 内存安全」六项的开源引擎
2. **唯一 Rust 实现**，单二进制部署、跨平台编译矩阵清晰
3. **ds4 风格的工程姿势**：自研内核 + GGUF 兼容 + 显式借鉴致谢 llama.cpp/ds4/candle/FlashAttention/MLX，不引入任何外部推理引擎运行时依赖（见 [ADR-0001](adr/0001-engine-architecture.md)）

## 6. 价值主张（Value Proposition Canvas）

### 客户痛点 → rsLLM 价值
| 痛点 | rsLLM 解决方式 |
|---|---|
| Python 推理栈难分发 | 单 Rust 二进制 |
| 跨平台部署不一致 | CI 矩阵 + 启动期 capability 探测 |
| 大模型显存不够 | 异构 CPU/GPU + 异步 expert offload |
| Agent 系统提示反复 prefill | 磁盘 KV cache + prefix cache |
| 多模型切换重启 | 控制面 load/unload + 模型仓库 |
| 多种量化格式碎片化 | GGUF 一统 + 非对称量化 |
| Mac 用户被 ds4 锁死 | 多模型 Metal 后端 |
| 工业级稳定性 | Rust 类型系统 + tokio + tracing |

### 客户收益 → rsLLM 创造
1. **降低单 token 成本**：异构计算让旧硬件跑动新模型
2. **降低部署复杂度**：单二进制 + 配置文件
3. **降低集成成本**：API 兼容主流生态
4. **降低运维风险**：Rust 内存安全 + 完整可观测性
5. **降低锁定风险**：开源 + 开放格式 + 可读源码

## 7. 商业 / 治理模型

> 注：rsLLM 是一个**开源项目**，不直接产生收入；本节是治理模型。

- **License**：Apache 2.0（已固定）
- **治理**：
  - 早期：BDFL（仁慈独裁者）模式，决策清晰
  - 成熟期：core maintainers + RFC 流程（参考 Rust RFC）
- **依赖政策**：
  - 关键路径不引入 GPL/AGPL 依赖
  - llama.cpp / ggml 通过可选 feature 引入（MIT 兼容）
- **贡献门槛**：
  - 所有 PR 必须通过 CI（编译、clippy、test、benchmark 回归）
  - 性能敏感路径需附 benchmark 数据

## 8. 成功画像（What does success look like）

### 一年内
- ⭐ GitHub Stars 5k+
- ✅ 至少 3 个主流开源模型家族（Llama/Qwen/DeepSeek）能跑且性能 ≥ llama.cpp
- ✅ 在 4090 上 Llama 70B Q4 达到 35+ t/s
- ✅ 在 4090 + 64GB DRAM 上 DeepSeek-V3 IQ2 跑通
- ✅ 有 ≥10 个外部贡献者
- ✅ 至少 2 个真实生产项目使用

### 三年内
- ⭐ 成为 Rust 推理引擎事实标准
- ✅ 覆盖 ≥10 个模型家族
- ✅ 跨 5 个硬件后端（CUDA / Metal / Vulkan / CPU x86 / CPU ARM）
- ✅ 性能在 ktransformers 0.9× 以上、llama.cpp 1.3× 以上
- ✅ 有 ≥3 家公司贡献核心代码

## 9. 风险画像（What could kill this）

| 风险 | 致命度 | 缓解 |
|---|---|---|
| 性能始终上不去 | **致命** | 早期请性能工程师 review，第 6 个月起每月 benchmark vs 同类 |
| 跨平台编译矩阵失控 | 高 | 限制 supported triple 数量，明确 tier 1/2/3 |
| 模型生态变化太快（新架构） | 中 | Attention/FFN trait 设计要可扩展 |
| 维护者 burnout | 高 | 早期不要承诺 SLA，明确"best effort" |
| 出现一个 Rust 同类项目并赢得心智 | 中 | 紧盯 candle / mistral.rs 生态，差异化在「异构 + 多平台 + 磁盘 KV」 |

## 10. 给读者的 take-away

如果你只能记住三句话：

1. **rsLLM = ds4 的工程姿势 + ktransformers 的异构调度 + llama.cpp 的格式生态，用 Rust 重写并显式致敬。**
2. **同一二进制、同一 API，在 8GB 笔记本和 8×H100 上都能跑得动 / 跑得好。**
3. **借鉴而非依赖、致谢而非链接——见 [ADR-0001](adr/0001-engine-architecture.md) 与 [`NOTICE.md`](../NOTICE.md)。**
