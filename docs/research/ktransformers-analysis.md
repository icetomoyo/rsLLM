# KTransformers 深度调研

> 调研对象：`C:\Works\PubGItProj\ktransformers`（v0.6.2.post2）
> 目的：作为 rsLLM 的设计参考，理解其架构、创新与边界。

---

## 1. 项目定位

- **维护方**：清华 MADSys 实验室 + Approaching.AI + 9#AISoft
- **使命**：让前沿规模的 MoE 模型（DeepSeek-V2/V3/R1，671B 量级）跑在「消费级/工作站级 GPU + 大 DRAM」上
- **核心洞察**：MoE 推理中每个 token 只激活 8/256 专家 → 大部分专家是「冷」的，可以放到 CPU 上
- **代表数据**：DeepSeek-R1-0528 FP8 在 8×L20 + Xeon Gold 6454S：**227.85 t/s 总吞吐 / 87.58 t/s 输出**

## 2. 架构两层结构

### Archive 层（Python/PyTorch 集成框架）
- `archive/ktransformers/`：GGUF 加载、YAML 规则注入、PyTorch 算子替换、CLI、OpenAI API
- 通过 `BaseInjectedModule` 透明替换 HuggingFace 的 `nn.Module`

### kt-kernel 层（C++/CUDA 内核库）⭐ 主力开发线
- 独立 C++ 库，pybind11 暴露 Python API
- 只负责 **expert FFN 计算**，attention/routing/embedding 留给调用方
- 关键类：`KTMoEWrapper`

## 3. 核心技术创新

### 3.1 YAML 驱动的模块注入
```yaml
- match:
    name: "^model\\.layers\\..*\\.mlp\\.experts$"
  replace:
    class: ktransformers.operators.experts.KTransformersExperts
    kwargs:
      generate_device: "cpu"
      prefill_device: "cuda"
```
- 首匹配胜出，递归遍历 `nn.Module` 树
- `BaseInjectedModule` 用 `__getattr__/__setattr__` 代理原模块属性，保持上层透明

### 3.2 CPUInfer：CUDA Stream 集成的 CPU 工作池
- `kt-kernel/cpu_backend/cpuinfer.h`
- **关键创新**：用 `cudaLaunchHostFunc()` 而不是 mutex/condvar 来调度 CPU 任务
- CUDA stream 成为 CPU/GPU 事件序的单一真相源
- NUMA 感知：`CPUInfer(thread_num, numa_id)` 构造函数绑定到指定 NUMA 节点，用 `hwloc + libnuma`

### 3.3 AMX 内核架构
- CRTP 模板层次：`AMX_MOE_BASE<T, Derived>` → `AMX_MOE_TP<T>`
- 64 字节对齐的 weight/accumulator buffers
- 自定义 `.kt` 权重格式（`_quant_` + `_scale_` 二进制文件）
- AMX 指令：`_tile_loadd / _tile_dpbssd / _tile_stored`，~2 TOPS

### 3.4 CPU 变体自动检测
- `_cpu_detect.py` 启动时探测 CPUID
- 选择最优共享库：AMX > AVX512+BF16 > AVX512+VBMI > AVX512+VNNI > AVX512 Base > AVX2 (LLAMAFILE 兜底)
- 通过 `KT_KERNEL_CPU_VARIANT` 环境变量覆盖

### 3.5 MLA 吸收（DeepSeek 专用）
- `KDeepseekV2Attention.get_absorbed()`
- 把 `kv_b_proj` 权重预先吸收进 `q_absorb / out_absorb`
- KV cache 保持压缩形态，省 decompression 带宽

### 3.6 动态专家放置
- `generate_gpu_experts_masks()` 根据 `activation_freq` 张量逐层 topk
- 允许运行时根据任务类型重新路由

### 3.7 双缓冲 pinned memory
- `KExpertsCPUBuffer` depth-2 ping-pong
- 隐藏 PCIe 传输延迟

## 4. 支持矩阵

### 模型
| 家族 | 状态 |
|---|---|
| DeepSeek-V2/V3/R1 | 主要目标，全部优化生效 |
| Mixtral | MoE 卸载适用 |
| Qwen2-MoE | 支持 |
| Llama/Mistral 稠密 | 主要走 GPU |
| Kimi-VL（多模态） | 近期加入 |

### 量化
- AMXINT4 / AMXINT8 / RAWINT4
- FP8 / FP8_PERCHANNEL
- BF16
- GPTQ_INT4（Marlin 后端）
- MXFP4
- LLAMAFILE / MOE_INT4 / MOE_INT8（GGUF 兼容，AVX2 后端）

### 硬件
| 维度 | 支持 |
|---|---|
| GPU CUDA | SM 80+（A100/3090/4090/L20/H100），不支持 Turing/Volta |
| GPU 其他 | ROCm（AMD）、MUSA（摩尔线程） |
| CPU x86 | AMX / AVX512 全套 / AVX2 兜底 |
| CPU ARM | 编译标记包含 `armv8.2-a+sve+bf16`，进行中 |
| 系统 | Linux 完整支持；Windows 走 fallback；macOS 不支持 |

### 显存/内存门槛
- DeepSeek-V3 Q4_K_M：**382 GB DRAM + 14 GB VRAM**（单卡 INT4）
- 推荐：双路 Xeon + 768GB DRAM + RTX 4090

## 5. 性能数据

| 配置 | 吞吐 |
|---|---|
| DeepSeek-R1-0528 FP8，8×L20 | 87.58 t/s 输出 / 227.85 t/s 总 |
| DeepSeek-V3 Q4_K_M，4090 + Xeon | ~16.8 t/s decode |
| Prefill（AMX + 6 GPU experts） | 255–286 t/s |
| vs llama.cpp decode | 最高 3.03× |
| vs llama.cpp prefill | 最高 9.44× |
| 128K 长文本 vs llama.cpp | 7.1× |
| 1M 上下文（稀疏 attn） | ~16 t/s on 24GB GPU |

## 6. 用户界面

- **CLI**：`local_chat.py`（fire 驱动）
- **API Server**：基于 SGLang fork，OpenAI Chat Completions
- **Python lib**：`KTMoEWrapper.submit_forward() / sync_forward()`
- **SFT**：基于 AMX 反向算子，集成 LLaMA-Factory

## 7. 构建与依赖

- CMake + setuptools
- 关键 CMake 选项：`KTRANSFORMERS_USE_ROCM / MUSA / CPU_MOE_AMD / CUDA_STATIC_RUNTIME`
- 运行时依赖：CUDA、PyTorch、FlashInfer、Flash-Attention 2、Triton、hwloc、libnuma
- 服务器依赖 SGLang fork（维护负担大）

## 8. 局限与给 rsLLM 的启示

### 应该「保留」的设计
1. **`cudaLaunchHostFunc` 异构调度原语**——通过 `cudarc` FFI 在 Rust 里照搬
2. **NUMA 感知线程池**——Rust `hwloc2` crate 或 `libnuma` FFI
3. **CRTP 内核特化 → Rust trait 单态化**：每个量化方法是一个 `MoEKernel` trait 实现
4. **启动期变体探测 → 编译期单态化或函数指针分发**
5. **双缓冲 pinned memory**——`Arc<Mutex<[Buffer; 2]>>` 配合 `cudaMallocHost`
6. **submit/sync 异步 API**——Rust 里返回 `Future` 或 `JoinHandle<()>`
7. **每层 GPU expert mask 动态调整**：使用滚动 EMA 跟踪 activation frequency

### 应该「丢弃」的设计
1. **PyTorch/Python 深度耦合**：rsLLM 用静态图变换替代运行时 monkey-patching
2. **llama.cpp 作为 AVX2 兜底**：rsLLM 可用 `std::arch` 原生实现或薄绑定
3. **量化流水线缺失**：rsLLM 应内建或明确委托量化
4. **SGLang fork 服务器**：rsLLM 自己用 Axum + SSE
5. **Windows 上 AMX 失效**：rsLLM 早期可以接受同样限制，但要明确文档

### 性能陷阱
- **专家负载不均**：需要 work-stealing
- **PCIe 带宽瓶颈**：~16 GB/s 双向，双缓冲只能部分隐藏
- **AMX tile state 切换**：`_tile_loadconfig` ~100ns，按层批量摊销
- **Prefill 内存压力**：`chunked_prefill_size` 必须是显式调参旋钮
