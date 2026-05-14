# ds4.c 深度调研

> 调研对象：`C:\Works\PubGItProj\ds4`
> 目的：作为 rsLLM 的设计参考，吸收其低算力/单文件工程精神。

---

## 1. 项目定位

- **作者**：antirez（Salvatore Sanfilippo，Redis 创始人）
- **本质**：为 DeepSeek V4 Flash（284B MoE）量身打造的单模型 C 推理引擎
- **平台**：macOS / Apple Silicon / Metal-only
- **目标硬件**：MacBook Pro M3 Max 128GB / Mac Studio M3 Ultra 512GB
- **代码量**：核心 `ds4.c` ~4000+ 行 C99
- **License**：MIT
- **状态**：alpha，AI 辅助编写，作者亲自把关设计

**一句话定位**：手写的、为单一模型极致优化的 Mac Metal 推理引擎，靠非对称量化和磁盘 KV cache 让 284B MoE 跑在 128GB 笔记本上。

## 2. 核心能力

- 只做推理，不做训练/微调
- 只支持一个模型：DeepSeek V4 Flash
- 模型参数硬编码在 `enum`（`ds4.c:82-105`）：43 layers, 4096 dim, 129280 vocab, 64 attn heads, 1 KV head, 256 routed experts (6/token), 1 shared expert, 4 HC streams
- **独到之处**：
  - vs llama.cpp：放弃通用性，换更紧的内核融合
  - vs vLLM：单会话、单 Mac、Metal-only
  - vs ktransformers：苹果芯片专属，靠 unified memory 而非 PCIe 分摊

## 3. 架构

### 顶层结构
| 文件 | 角色 |
|---|---|
| `ds4.c` | 引擎核心：加载、tokenizer、CPU 参考内核、Metal 调度、session、磁盘 KV |
| `ds4.h` | 公开 API：`ds4_engine` / `ds4_session` 两个不透明类型 |
| `ds4_cli.c` | CLI 前端 + linenoise REPL |
| `ds4_server.c` | HTTP server：OpenAI + Anthropic Messages 端点 |
| `ds4_metal.m` | Objective-C Metal runtime |
| `metal/*.metal` | 20 个 MSL 计算内核 |
| `linenoise.c/h`, `rax.c/h` | 内嵌单文件库（REPL 和 radix tree） |

### 语言矩阵
- C99：~95% 代码
- Objective-C：只在 Metal runtime 必需处
- Metal Shading Language：GPU 内核
- **零 C++**（AGENT.md 明确禁止）、**零 Python**（除测试脚本）

### Engine / Session 分离 ⭐
- `ds4_engine`：加载完成的模型（不可变，可共享）
- `ds4_session`：可变 KV 状态、推理时间线（不可 Send）
- 这是非常干净的边界，适合 Rust 直译

## 4. 「低算力」故事

### 量化策略（核心）
- **routed MoE 专家 gate/up**：`IQ2_XXS`（~2-bit 码本查找）
- **routed MoE 专家 down**：`Q2_K`（2-bit + scale blocks）
- **Q4 变体**：`Q4_K`
- **其余权重**：F16 / Q8_0 / F32 不动
- **核心思想**：MoE 专家占字节数大头但每 token 只激活 6/256 → 只压它们就能在保质量前提下大幅省内存
- README 原话："only the routed MoE experts are quantized… others left untouched to guarantee quality"

### KV cache 多重压缩
1. **FP8 (E4M3FN)**：non-RoPE 部分按 64-element block FP8 round-trip（`ds4.c:1444-1507`, `dsv4_kv.metal`）
2. **MLA 压缩**：DeepSeek 多潜变量注意力本身把 KV 压缩到很小
3. **逐层压缩比异构**（`ds4_layer_compress_ratio()`, `ds4.c:407-411`）：
   - 层 0-1：dense raw KV
   - 偶数层 ≥2：ratio=4
   - 奇数层 ≥2：ratio=128
4. **Indexer top-K**：每 token 选 512 个 compressed rows
5. **结果**：1M 上下文 KV 只占 ~26GB（其中 indexer 22GB）

### mmap 零拷贝
- 模型 mmap 一次，Metal 把切片包装为零拷贝 `MTLBuffer`
- 不在 RAM 里再复制 80-153GB 模型

### 磁盘 KV cache ⭐
- KVC 文件格式（48-byte header + token IDs + logits + per-layer KV rows）
- SHA1(token IDs) 作为缓存键
- 单一活跃会话 + 冷会话 evict 到磁盘
- 关键场景：agent 模式下 25k token 系统提示无需重新 prefill
- 启用：`--kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`

### MTP 推测解码
- 独立 ~3.5GB GGUF 作为 draft model
- `--mtp gguf/... --mtp-draft 2 --mtp-margin F`
- 当前仅小幅提速

## 5. 性能数据

| 配置 | 吞吐 |
|---|---|
| MBP M3 Max 128GB, q2, 短提示 | 26.68 t/s gen / 58.52 t/s prefill |
| MBP M3 Max 128GB, q2, 11.7k token | 21.47 t/s gen / 250.11 t/s prefill |
| MS M3 Ultra 512GB, q4, 短提示 | 35.50 t/s gen / 78.95 t/s prefill |
| MS M3 Ultra 512GB, q4, 12k token | 26.62 t/s gen / 448.82 t/s prefill |

## 6. 模型 / 模态 / 特性

- 模型架构：**仅一个**——DeepSeek V4 Flash
- 长文本：**1M tokens**（YaRN RoPE，`DS4_ROPE_SCALE_FACTOR = 16.0`）
- 多模态：无
- 推测解码：MTP
- 批处理：**无**（单会话串行，server 也是单 worker 排队）
- 流式：SSE on `/v1/chat/completions` 和 `/v1/messages`
- 工具调用：渲染为 DeepSeek DSML 格式，回写时映射回 OpenAI/Anthropic shape
- Thinking 模式：`THINK_NONE / HIGH / MAX`（MAX 需 ≥384K 上下文）

## 7. 用户界面

### CLI
```sh
./ds4 -p "Explain Redis streams"
./ds4                              # 交互 REPL
./ds4 --dump-tokens -p "..."
./ds4 --dump-logprobs out.json ... # greedy + logprob 调试
```
交互模式命令：`/help /think /think-max /nothink /ctx N /read FILE /quit`

### HTTP Server
```sh
./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```
端点：`GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`, `POST /v1/messages`

## 8. 构建与依赖

- **GNU Make**（单 `make` 调用产出 `ds4` + `ds4-server`）
- `-O3 -ffast-math -mcpu=native -Wall -Wextra -std=c99`
- 依赖：仅 `Foundation`, `Metal`, `libm`, `pthread`
- 第三方库全部内嵌（linenoise.c + rax.c）

### 已知问题
- **macOS CPU backend 有内核 panic 风险**：大 mmap + 大量虚拟内存可能触发 Darwin `cpt_mapcnt_inc` panic，**生产环境必须用 Metal**
- 无 Windows、Linux GPU 路径
- 模型只认作者预制的 `antirez/deepseek-v4-gguf`

## 9. 代码质量

### 优点
- 干净、直白、单线程心智模型清晰
- 注释解释「为什么」（如 `ds4.c:1196-1210` 解释 mmap VM panic）
- 严格的官方 logprob 回归测试（短 + 12k+ 长文本）
- AGENT.md 立下明确规矩（窄 API、解释 cache 策略、禁 C++）

### 风险点
1. **单会话服务器**：所有请求阻塞排队
2. **macOS CPU 不安全**
3. **模型锁定**：换 V5 要大改
4. **单一平台**（Apple Silicon）
5. **AI 辅助编写**：边角逻辑可能有未被 logprob 测试覆盖的微妙错误

## 10. 给 rsLLM 的启示

### 强烈建议借鉴
1. ⭐ **mmap 零拷贝模型加载**：Rust `memmap2 + Arc<Mmap>`，分片传给 GPU buffer
2. ⭐ **磁盘 KV cache 一等公民**：KVC 文件格式直接复用或扩展
3. ⭐ **非对称量化**：只压 MoE routed experts，保留其他权重的精度
4. ⭐ **FP8 (E4M3FN) KV 量化**：64-element block round-trip
5. ⭐ **Engine / Session 分离**：`struct Engine: Send + Sync` + `struct Session: !Send`
6. ⭐ **官方 logprob 验证**：作为 `cargo test` 一等公民
7. **HC（Hyper-Connection）残差流**：DeepSeek 系特化，需要时按需实现

### 不要照搬
1. **Metal-only**：rsLLM 应 CUDA + Metal + wgpu 多后端
2. **单会话 server**：rsLLM 至少要支持带前缀复用的请求队列
3. **模型 shape 硬编码**：rsLLM 应从 GGUF metadata 加载并 const-generic 单态化热路径
4. **macOS-only CPU backend**：rsLLM CPU 后端必须在 Linux 上稳定
5. **GNU Make 单文件**：rsLLM 用 Cargo workspace 分层组织

### 必须一等公民的特性
1. mmap + GPU 零拷贝绑定
2. 磁盘 KV cache + 会话淘汰策略
3. 异构 KV 压缩（dense + ratio-4 + ratio-128 + FP8）
4. IQ2_XXS / Q2_K / Q4_K GGUF 解码（用于加载 ds4 兼容权重）
5. DSML 工具调用格式（如果支持 DeepSeek 系）
6. MTP 推测解码（可选模块）

## 11. 关键文件清单

- `/C:/Works/PubGItProj/ds4/ds4.c` — 引擎核心
- `/C:/Works/PubGItProj/ds4/ds4.h` — 公开 API 边界
- `/C:/Works/PubGItProj/ds4/ds4_metal.h` — Metal 调度 API
- `/C:/Works/PubGItProj/ds4/metal/dsv4_kv.metal` — FP8 E4M3FN KV
- `/C:/Works/PubGItProj/ds4/metal/moe.metal` — IQ2_XXS + Q2_K MoE matvec
- `/C:/Works/PubGItProj/ds4/metal/flash_attn.metal` — Flash attention
- `/C:/Works/PubGItProj/ds4/metal/dsv4_hc.metal` — Hyper-Connection
- `/C:/Works/PubGItProj/ds4/ds4_server.c` — HTTP server + KV 策略
- `/C:/Works/PubGItProj/ds4/README.md` — KVC 文件格式规范（行 373-460）
