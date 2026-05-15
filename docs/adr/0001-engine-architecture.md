# ADR-0001：引擎架构——自研内核 + GGUF 格式兼容（ds4 风格）

- **状态**：Accepted（2026-05-14 amended：v0.1.0 双首发硬件确认为 Mac + AMD Ryzen AI Max+ 395）
- **日期**：2026-05-11（原始）/ 2026-05-14（amendment）
- **决策者**：项目发起人

---

## 背景

rsLLM 是从零启动的 Rust LLM 推理引擎。在确定整体架构之前，需要决定一个最根本的问题：

> **「张量算子、量化解码、内核计算这一层，是复用 llama.cpp/ggml 这类成熟的 C/C++ 推理栈，还是自己用 Rust + CUDA/Metal/SIMD 重写？」**

这个决策影响 workspace 切分、路线图长度、跨平台编译复杂度、性能上限、社区定位、商业风险——是 v1.0 之前最重大的单一决策。

## 备选方案

在长达数轮的讨论中，考虑了四种形态：

### 形态 A：完全 llama.cpp wrapper
- 进程内嵌 llama.cpp，所有内核走它
- 类比：ollama
- 评价：维护负担最小，但和 ollama 撞型，差异化几乎只有"Rust 写的"

### 形态 B：llama.cpp FFI + 渐进替换
- llama.cpp 作为可选 feature 兜底，自有内核逐步替换
- 评价：野心大但回报周期长，"自己跟自己卷"

### 形态 C：llama.cpp FFI 永久共存
- llama.cpp 是永久内核后端，rsLLM 在它**周围**做差异化
- 评价：务实，6-9 个月可见产品价值；但 KV/调度受 llama.cpp 设计限制，差异化天花板低

### 形态 G：自研内核 + GGUF 格式兼容（ds4 风格）⭐ 选定
- **不链接** llama.cpp/ggml
- **复用** GGUF 文件格式定义、量化 block 内存布局、量化解码查找表常量（按 MIT 致谢复用）
- 张量算子、量化解码、KV cache、调度全部自己实现
- 类比：ds4.c（antirez）的工程姿势

## 决策

**选定形态 G。**

具体规约：

1. **不链接 llama.cpp 或 ggml** 作为编译期或运行期依赖
2. **GGUF 兼容**：自己实现 GGUF 解析器（几百到一千行 Rust），用户任意 llama.cpp GGUF 文件可直接加载
3. **算法借鉴 + 法律致谢**：必要的量化 block 布局、查找表常量等可以借鉴并复用，源码顶部按 MIT 规则双重署名（`The rsLLM authors` + `The ggml authors`）
4. **致谢 ds4**：ds4 启发了形态 G 这条路线本身（"格式兼容但代码不依赖"的工程姿势）以及多个具体设计（磁盘 KV cache、非对称量化、FP8 KV、Engine/Session 分离），在 `README.md` / `NOTICE.md` / 关键文件 header 中明确致谢
5. **后端列表**：Tier 1 = CPU（NEON dotprod + AVX-512 VNNI）、Metal、Vulkan compute、CUDA；Tier 2 = AMX、ROCm；Tier 3 = wgpu。**llama.cpp 不在后端列表中**
6. **内核实现策略：借鉴而非从零**
   - rsLLM 的 CUDA / Metal / CPU SIMD 内核**显式借鉴**以下开源参考实现：
     - **llama.cpp / ggml**（MIT）：CUDA kernels、Metal kernels、量化解码、GGUF 格式
     - **ds4**（MIT）：磁盘 KV cache、非对称量化、FP8 KV、MLA 优化、Engine/Session 分离、单文件工程纪律
     - **candle**（Apache 2.0）：GGUF 解析、Llama / Mistral 等模型实现的 Rust 表达
     - **FlashAttention**（BSD-3）：FlashAttention v1/v2/v3 内核思路
     - **MLX**（MIT）：Apple Silicon Metal 优化思路（用于 Metal 后端参考）
     - **ktransformers**（Apache 2.0）：cudaLaunchHostFunc 异构调度、AMX 内核结构
   - **借鉴的法律规则**：
     - 直接复用源码片段（如量化查找表、内核算法）→ 源文件 header 双重署名（`The rsLLM authors` + 原作者）
     - 算法思路借鉴 + 自己重写 → 模块 doc-comment 中引用上游源文件路径
     - 所有借鉴源在 `NOTICE.md` 中汇总声明
   - **目标**：v1.0 阶段 rsLLM 性能达到 llama.cpp 同硬件的 80%+，**不是从零调优一个新内核生态，而是把已有最佳实践用 Rust 重新组装**
   - **协作模式（人 + AI）**：AI 负责读源码、移植代码、写测试、写调度；人负责提供 GPU 实测硬件、跑 benchmark、profile 找瓶颈、拍板架构

## 决策的关键依据

### ds4 证明了这条路可行
- `ds4.c` README 第 36-38 行明确："`ds4.c` **does not link against GGML**"
- ds4 用 4000+ 行 C 跑通了 284B MoE 模型
- 仅靠 MIT 双重署名复用 GGUF 格式定义和量化查找表

### 形态 G 是唯一能让 rsLLM 差异化兑现的方案
- 磁盘 KV cache、连续批处理、异构 CPU+GPU MoE、Radix tree prefix cache 等差异化能力**必须**完全自控 KV 和调度
- llama.cpp 内部 KV 模型固定，FFI 兜底无法兑现这些差异化

### 形态 G 的商业 / license 路径最干净
- 只依赖自己的代码 + 系统库 + 少数 Rust crate
- 没有 C++ 工具链耦合、没有 cmake 子项目、没有 ABI 风险
- 可以发布纯 MIT/Apache 双重 license 的纯 Rust 二进制

### 形态 G 的代价是工程量
- v1.0 时间窗口 12-18 个月（vs 形态 C 的 6-9 个月）
- 需要 CUDA / Metal 内核能力（团队需要或培养）
- 早期性能可能不如 llama.cpp，要明确预期

## 后果

### 工程后果
- Workspace 不再有 `rsllm-backend-llama` crate
- 所有后端（CPU/CUDA/Metal/wgpu）都是 rsLLM 自实现的纯 Rust + 原生计算后端
- 新增 `rsllm-gguf` crate 自实现 GGUF 解析
- `build.rs` 不调 cmake，构建链路纯 cargo + 各平台原生工具链
- 二进制体积可压到 30-80MB
- 跨平台编译复杂度大幅下降

### 产品后果
- 路线图重新校准为 12-18 个月到 v1.0
- 性能基准：M0/M1 阶段目标"可用"，M3 之后争取追平/超越 llama.cpp
- 产品定位更独立，差异化"Rust 原生 + 跨平台 + 自有内核 + 异构调度 + 磁盘 KV"完整保留

### 法律后果
- 引擎代码使用 Apache 2.0 license
- 复用自 ggml 的代码片段（GGUF 格式定义 + 量化查找表 + 必要时的算法）按 MIT 规则保留 ggml authors 版权署名
- 项目根目录提供 `NOTICE.md` 列明所有上游致谢和复用源
- 用户运行的模型权重 license 由用户自行负责（rsLLM 不对模型授权负任何责任）

### 致谢后果
- README.md "Acknowledgements" 章节同时致敬：
  - **llama.cpp / ggml**：GGUF 生态、量化格式、kernels 设计参考
  - **ds4 (antirez)**：形态 G 工程姿势 + 磁盘 KV cache + 非对称量化 + FP8 KV + Engine/Session 分离 等具体设计
  - **ktransformers**：异构 CPU+GPU MoE 卸载 + cudaLaunchHostFunc 调度模式
- 关键源文件 header 加入双重版权（rsLLM + 借鉴源）
- 推荐做法仿照 ds4：诚实说明"借鉴 ≠ 链接，致谢 ≠ 依赖"

## 不做的事（避免混淆）

- ❌ 不引入 llama.cpp / ggml 作为编译期或运行期依赖
- ❌ 不引入 candle / tch-rs 作为基础（同样原因：抽象层级不对，要掌控底层）
- ❌ 不引入 mistral.rs / mlx-rs 等其他推理栈
- ❌ 不在 v1.0 之前考虑"如果自有内核打不过 llama.cpp 怎么办"——接受短期性能差距，长期靠工程兑现

## 重新评审条件

如果以下事件之一发生，本 ADR 应重新评审：

- **18 个月后自有 CUDA 内核性能仍距 llama.cpp > 30%**：考虑借鉴更多上游内核或临时引入 FFI 兜底 feature
- **核心团队失去 CUDA / Metal 内核工程能力**：考虑形态 C 退路
- **GGUF 生态被显著替代**（例如 HF 推出新事实标准）：调整格式兼容策略
- **某主流模型架构 6 个月内无人在 rsLLM 实现**：考虑临时借助外部引擎

---

## 2026-05-14 修订：v0.1.0 双首发硬件确认

本次修订**不改变 ADR-0001 的本体决策**（形态 G），只把 v0.1.0 的硬件落地目标具体化：

### v0.1.0 双首发硬件

| 平台 | 主加速路径 | 验证路径 | 选择理由 |
|---|---|---|---|
| **Mac (Apple Silicon M3+)** | Metal kernel（F025） | NEON CPU reference | ds4 已经在此平台跑通 DS V4 Flash，移植阻力最小 |
| **AMD Ryzen AI Max+ 395 (Strix Halo) Linux** | AVX-512 + VNNI CPU（F004）→ v0.1.1 加 Vulkan iGPU（F033） | 标量 fallback | 用户实际持有硬件；统一内存架构（与 Apple Silicon 同 idiom）；128GB + 2TB SSD 可装 DS V4 Flash 量化 |

### 为什么不在 v0.1.0 做 NVIDIA CUDA

- ds4 没有 CUDA 实现，没有"工程姿势"参考
- 已选定的两台首发硬件覆盖了 Apple Silicon UMA + AMD Strix Halo UMA 两个"统一内存"案例，已足以验证 idiom
- NVIDIA CUDA 留到 v0.1.6——届时 v0.1.0-v0.1.5 已验证 CAL trait 在 4 个 backend（NEON CPU / AVX-512 CPU / Metal / Vulkan）上的可移植性

### AMD AI Max+ 选 Vulkan compute 而非 ROCm

- **Vulkan 跨平台**：一份 compute shader 在 Linux/Windows 都能跑，NVIDIA dGPU 和 Intel Arc iGPU 也能复用
- **ROCm 在 Strix Halo 上不稳**：HSA runtime 对 Strix Halo 的支持还在演进
- **借鉴 llama.cpp `ggml-vulkan` 已有 kernel**（MIT 致谢）
- ROCm 留给独立 Radeon dGPU 用户（v0.2.x F030 best-effort）

### 致谢更新

NOTICE.md / README.md 致谢部分**新增**：
- **AMD Strix Halo 文档**：用于统一内存特性理解
- **llama.cpp ggml-vulkan**（MIT）：v0.1.1 Vulkan compute shader 参考

## 参考资料

- 调研报告：[`../research/ktransformers-analysis.md`](../research/ktransformers-analysis.md)
- 调研报告：[`../research/ds4-analysis.md`](../research/ds4-analysis.md)
- ds4 README 致谢章节（`C:\Works\PubGItProj\ds4\README.md`，第 34-42 行）
- ds4 LICENSE 双重版权写法（`C:\Works\PubGItProj\ds4\LICENSE`，第 3-4 行）
