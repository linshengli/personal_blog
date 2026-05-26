# 大模型推理服务 SLA 保障与延迟优化技术 — 深度调研报告

> **调研日期**：2026-05-26
> **所属领域**：大模型框架
> **关键词**：LLM Serving, SLA, TTFT, TPOT, Prefill-Decode Disaggregation, Continuous Batching, Speculative Decoding, PagedAttention, Quantization

---

## 目录

1. [第一部分：概念剖析](#第一部分概念剖析)
2. [第二部分：行业情报](#第二部分行业情报)
3. [第三部分：方案对比](#第三部分方案对比)
4. [第四部分：精华整合](#第四部分精华整合)

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

大模型推理服务 SLA（Service Level Agreement）保障与延迟优化，是指在大语言模型（LLM）在线推理场景中，通过系统架构设计、调度策略优化和底层算子加速，确保服务质量协议中约定的关键性能指标（如首 Token 延迟 TTFT、Token 间延迟 TPOT、有效吞吐 Goodput）得到满足的一系列技术与工程实践。

### 常见误解

1. **"延迟优化 = 模型加速"**：模型加速（如 FlashAttention、算子融合）只是其中一层。更大比例的延迟收益来自系统级调度——连续批处理、请求优先级调度、Prefill-Decode 分离等架构决策对延迟的影响往往远大于纯算子优化。
2. **"吞吐越高，延迟越低"**：在大模型推理中，吞吐和延迟通常是**相互冲突**的目标。最大化吞吐（如塞满 GPU 做最大批处理）会显著增加每个请求的排队延迟。现代推理系统的核心挑战是在 SLO 约束下最大化 Goodput（有效吞吐），而非原始吞吐。
3. **"推理延迟瓶颈在计算量"**：解码阶段（Decode）的核心瓶颈是**内存带宽**而非计算量。生成每个 Token 只需要一次 Transformer 前向传播，权重和 KV Cache 的读取时间远大于计算时间。因此量化（减少权重读取量）对解码加速效果显著。

### 边界辨析

| 易混淆概念 | 核心区别 |
|-----------|---------|
| **推理引擎 vs 推理框架** | 引擎指底层执行单元（如 TensorRT-LLM），框架指上层调度系统（如 vLLM、SGLang）。框架可切换底层引擎 |
| **量化 vs 稀疏化** | 量化降低数值精度（FP16→INT4），稀疏化裁剪或跳过不重要参数。量化是当前主流，稀疏化在推理端尚未大规模落地 |
| **Prefill vs Decode** | Prefill 处理输入 Prompt（计算密集型，需一次处理所有输入 Token），Decode 逐 Token 生成输出（内存带宽密集型）。二者瓶颈性质完全不同，分离部署成为主流选择 |

## 1.2 核心架构

### 系统级架构

```
                          ┌─────────────────────────────────────────┐
                          │          LLM 推理服务系统架构              │
                          └─────────────────────────────────────────┘
                                     │
  ┌───────┐    ┌──────────┐    ┌────▼─────┐    ┌──────────┐    ┌──────┐
  │客户端  │───▶│  网关层   │───▶│  调度器   │───▶│  推理引擎  │───▶│ 输出  │
  │请求    │    │ 路由/限流 │    │ 排队/批处理│    │ GPU执行   │    │ 流式  │
  └───────┘    └──────────┘    └────┬──────┘    └────┬─────┘    └──────┘
                                     │               │
                            ┌────────▼─────┐  ┌──────▼──────┐
                            │  KV Cache     │  │ 量化/算子   │
                            │  管理器       │  │ 加速层      │
                            └──────────────┘  └──────────────┘
```

**各组件职责**：

| 组件 | 一句话职责 |
|------|-----------|
| **网关层** | 路由请求到最合适的模型实例，执行速率限制（Token Bucket）、负载均衡和熔断降级 |
| **调度器** | 维护请求队列，基于 SLO 约束（TTFT/TPOT 目标值）执行连续批处理，混合 Prefill 和 Decode 请求 |
| **推理引擎** | 执行 Transformer 前向传播，管理 GPU 内存（Paged KV Cache），调用底层加速算子 |
| **KV Cache 管理器** | 以分页方式管理 Attention Key/Value 缓存，支持前缀复用（Prefix Caching）和内存换入换出（Swapping） |
| **量化/算子加速层** | 通过权重量化（AWQ/FP8）减少内存读取，通过 FlashAttention、CUDA Graph 减少计算和 Kernel Launch 开销 |

### Prefill-Decode 分离架构

```
          ┌───────────┐                    ┌───────────┐
          │ Prefill   │ ─── KV Cache ───▶ │ Decode    │
          │ GPU 集群  │    (高速网络传输)   │ GPU 集群  │
          │ (计算优化) │                    │ (带宽优化) │
          └───────────┘                    └───────────┘
                │                               │
                │  计算密集型                     │  内存带宽密集型
                │  200-400 ops/byte              │  60-80 ops/byte
                │  GPU利用率 90-95%               │  GPU利用率 20-40%
```

**核心洞察**：Prefill 和 Decode 对 GPU 资源的需求性质完全不同。物理分离后，Pre-fill实例专注高计算密度，Decode 实例专注高内存带宽利用，避免相互干扰。分离架构可降低基础设施总成本 **15-40%**（InfoQ, 2026）。

## 1.3 数学形式化

### 公式 1：Time-to-First-Token (TTFT)

$$
TTFT = T_{queue} + T_{prefill} + T_{network}
$$

- $T_{queue}$：请求在调度队列中的等待时间，取决于当前负载和调度策略
- $T_{prefill}$：输入 Prompt 的预填充时间，正比于 Prompt 长度 $L_{prompt} \times (层数 \times 隐藏维度)$
- $T_{network}$：网络传输时间（在分离架构中包括 KV Cache 传输）

### 公式 2：Time-Per-Output-Token (TPOT)

$$
TPOT = \max\left( \frac{D_{weights} + D_{kv}}{BW_{memory}},\; T_{compute} \right)
$$

- $D_{weights}$：每 Token 需读取的模型权重数据量，$= 2 \times N_{params} \times Precision_{bytes}$
- $D_{kv}$：每 Token 需读取的 KV Cache 数据量
- $BW_{memory}$：GPU 内存带宽（H100 约 3.35 TB/s，H200 约 4.8 TB/s）
- $T_{compute}$：纯计算时间（FlashAttention 等加速后通常远小于读取时间）

**工程意义**：解码阶段通常受内存带宽限制，因此量化（减小 $D_{weights}$）对 TPOT 的优化效果最为直接。

### 公式 3：Goodput（有效吞吐）

$$
Goodput = \frac{N_{slo\_compliant}}{T_{window}} \times \overline{L_{output}}
$$

- $N_{slo\_compliant}$：在时间窗口 $T_{window}$ 内满足 SLO 约束（TTFT < 阈值 && TPOT < 阈值）的请求数
- $\overline{L_{output}}$：平均输出长度

**工程意义**：Goodput 是 2025-2026 年业界公认的核心指标，它要求系统在满足延迟约束的前提下最大化产出，而非简单追求原始吞吐量。

### 公式 4：连续批处理的效率模型

$$
Util_{GPU} = 1 - \frac{T_{bubble}}{T_{batch}} = 1 - \frac{\sum (T_{wait} \cdot N_{idle\_SM})}{T_{batch} \cdot N_{total\_SM}}
$$

- $T_{bubble}$：批处理中 GPU 空闲等待时间
- $N_{idle\_SM}$：空闲的 Streaming Multiprocessor 数量

**工程意义**：连续批处理通过实时插入新请求到运行中的批次来最小化 $T_{bubble}$，GPU 利用率从传统批处理的 ~40-58% 提升至 ~82-92%。

### 公式 5：KV Cache 内存占用模型

$$
M_{kv} = 2 \times N_{layers} \times N_{kv\_heads} \times L_{seq} \times H_{head} \times P_{bytes}
$$

- $N_{layers}$：Transformer 层数
- $N_{kv\_heads}$：KV Head 数量
- $L_{seq}$：序列长度（含 Prompt + 已生成 Token）
- $H_{head}$：每个 Head 的维度
- $P_{bytes}$：精度字节数（FP16=2, FP8=1, INT4=0.5）

**工程意义**：此公式揭示 KV Cache 随序列长度线性增长。对于 128K 长上下文场景，KV Cache 可占用远超模型权重的显存，推动 KV Cache 量化、Prefix Caching 和稀疏 Attention 成为刚需。

## 1.4 实现逻辑（Python 伪代码）

```python
class LLMServingSystem:
    """大模型推理服务系统的核心抽象"""

    def __init__(self, engine, scheduler, quantizer):
        self.engine = engine          # 推理引擎（vLLM/SGLang/TensorRT-LLM）
        self.scheduler = scheduler    # SLO感知调度器
        self.quantizer = quantizer    # 量化器
        self.kv_cache_manager = KVCacheManager()
        self.request_queue = PriorityQueue()

    def serve(self, request_stream):
        """
        核心服务循环：持续接收请求、调度、批处理、推理
        """
        active_batch = []

        for request in request_stream:
            self.request_queue.push(request, priority=self._compute_slo_priority(request))

            # 连续批处理：在解码步边界插入新请求
            if self.scheduler.should_admit(self.request_queue, active_batch):
                batch = self.scheduler.form_batch(
                    running=active_batch,
                    pending=self.request_queue,
                    slo_constraints={
                        'ttft_max': 200,   # ms
                        'tpot_max': 50,     # ms
                        'gpu_mem_limit': self.engine.available_memory()
                    }
                )
                # 混合 Prefill 和 Decode 请求
                result = self.engine.execute_mixed_batch(batch)
                for r in result:
                    stream_output(r)

    def _compute_slo_priority(self, request):
        """基于 SLO 约束计算请求优先级（EDF: Earliest Deadline First）"""
        deadline = min(request.ttft_deadline, request.tpot_deadline)
        return deadline  # 越早截止的请求优先级越高


class KVCacheManager:
    """KV Cache 管理器——类虚拟内存的分页管理"""

    def __init__(self, block_size=16):
        self.block_size = block_size      # 每块 16 个 Token
        self.page_table = {}              # 逻辑页 → 物理块映射
        self.free_blocks = []             # 空闲块列表
        self.prefix_cache = LRUCache()    # Prompt 前缀缓存

    def allocate(self, sequence, prefix_hash=None):
        """分配 KV Cache，优先复用已缓存的前缀"""
        if prefix_hash and prefix_hash in self.prefix_cache:
            # Copy-on-Write 复用已缓存块
            return self.prefix_cache.get(prefix_hash)
        blocks = []
        for i in range(0, len(sequence), self.block_size):
            block = self.free_blocks.pop()
            self.page_table[i // self.block_size] = block
            blocks.append(block)
        return blocks

    def prefetch_prefix(self, prompt_tokens):
        """预取公共前缀的 KV Cache（用于 Prefix Caching）"""
        hash_val = hash(tuple(prompt_tokens[:self.block_size * 4]))
        if hash_val in self.prefix_cache:
            return self.prefix_cache[hash_val]
        return None


class MixedBatchExecutor:
    """混合 Prefill/Decode 批处理执行器"""

    def execute(self, batch):
        """
        关键调度决策：
        1. 将 Prefill 请求分块（Chunked Prefill）避免长 Prompt 阻塞
        2. 将 Decode 请求按照 KV Cache 长度对齐分组
        3. 在解码步边界插入新请求
        """
        # 分离 Prefill 和 Decode
        prefill_reqs = [r for r in batch if r.phase == 'prefill']
        decode_reqs = [r for r in batch if r.phase == 'decode']

        # 按 KV Cache 长度分组（AlignedServe 的核心思想）
        decode_groups = self._group_by_cache_length(decode_reqs, max_diff=256)

        # Prefill 分块执行
        for chunk in self._chunk_prefills(prefill_reqs, max_tokens=4096):
            self.engine.run_prefill(chunk)

        # Decode 分组执行
        for group in decode_groups:
            self.engine.run_decode(group)

        return batch
```

**关键设计思想**：以上伪代码体现了三个核心抽象——(1) **SLO 感知调度**将服务质量约束直接编码为优先级；(2) **KV Cache 分页管理**解决显存碎片化问题；(3) **混合批处理**通过分离 Prefill/Decode 和对齐 KV Cache 长度最大化 GPU 利用率。

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **TTFT (Time-to-First-Token)** | P50 < 200ms, P99 < 500ms | 端到端计时，至少 1000 请求（IETF 标准草案） | 直接影响用户首次感知体验 |
| **TPOT (Time-Per-Output-Token)** | < 30-50ms (等价 > 20-33 tok/s) | 流式输出每 Token 间隔计时 | 决定输出流畅度，受内存带宽主导 |
| **ITL (Inter-Token Latency)** | < 10ms (理想) | 相邻 Token 间延迟（与 TPOT 类似，侧重流式间隔） | 流式用户体验的核心指标 |
| **Goodput (有效吞吐)** | 最大化（场景依赖） | 满足 SLO 约束的请求总产出 Tokens/秒 | 2025-2026 业界公认核心指标，替代原始吞吐量 |
| **TTFT SLO 达成率** | > 99% | TTFT 小于阈值的请求占比 | 评估 SLA 合规程度 |
| **GPU 利用率** | > 80% | SM Active 时间占比 | 连续批处理效果的关键度量 |
| **KV Cache 命中率** | > 70% (有 Prefix Caching) | 前缀缓存命中的请求占比 | 直接影响 Prefill 延迟和吞吐 |

## 1.6 扩展性与安全性

### 水平扩展

- **模型分片（Tensor Parallelism）**：将单个模型切分到多张 GPU，每张 GPU 负责部分 Transformer 层的部分权重。Nitsum（2026）实现了自适应 TP 级别，支持亚秒级切换，SLO 合规吞吐提升最高 5.3 倍。
- **流水线并行（Pipeline Parallelism）**：将模型层切分到不同 GPU，适合单 GPU 装不下整个模型的场景。Cloudflare Infire 引擎使用流水线并行实现负载均衡。
- **多实例负载均衡**：CascadeInfer（2026）将集群分为长度专用的流水线阶段，请求随解码推进在阶段间传递，延迟降低 67%，吞吐提升 2.89 倍。
- **分离扩缩容**：Prefill 集群和 Decode 集群可独立扩缩容，Arrow（2025）实现动态调整 Prefill/Decode 实例比例，服务速率提升 2.55 倍。

### 垂直扩展

- **单 GPU 瓶颈上限**：受显存容量和内存带宽双重约束。H100 80GB 以 3.35TB/s 带宽运行 Llama-70B（FP16）时，单 GPU 仅能容纳 ~1 个并发序列。量化至 INT4 可扩展到 ~4-6 个并发序列。
- **算子优化**：FlashAttention-3 通过 warp specialization 重叠 matmul 和 softmax，在 H100 上实现 1.5-2 倍 Attention 加速。
- **CUDA Graph**：捕获稳定解码循环为 CUDA Graph，消除每 Token 的 Kernel Launch 开销（约节省数十微秒/Token）。

### 安全考量

- **请求隔离**：多租户场景下需防止恶意长 Prompt 挤占其他租户的显存和计算资源——通过 Request 级别的显存配额和 SLO 隔离机制。
- **SLA 攻击**：高并发请求可导致系统过载、TTFT 飙升。需配置令牌桶限流、队列长度限制和优雅降级策略。
- **Prompt 注入导致的资源滥用**：恶意构造的极长 Prompt 可耗尽 Prefill 资源。需要 Prompt 长度上限检测和分块 Prefill 保护。
- **数据安全**：KV Cache 中隐含了对话内容，在 Prefill-Decode 分离架构中跨节点传输时需考虑加密。Prefix Cache 跨用户共享时需注意信息泄露风险。

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | ~75k | PagedAttention 推理引擎，生产级通用推理框架 | Python, CUDA, C++ | 2026-05 | https://github.com/vllm-project/vllm |
| **SGLang** | ~26k | RadixAttention 推理引擎，结构化输出加速 | Python, CUDA, C++ | 2026-05 | https://github.com/sgl-project/sglang |
| **TensorRT-LLM** | ~10k | NVIDIA 官方编译式推理引擎，极致 GPU 性能 | C++, CUDA, Python | 2026-05 | https://github.com/NVIDIA/TensorRT-LLM |
| **llama.cpp** | ~75k | 纯 C/C++ 推理引擎，全平台支持（含 CPU/边缘设备） | C/C++, GGUF | 2026-05 | https://github.com/ggml-org/llama.cpp |
| **Ollama** | ~120k | 极简 LLM 本地部署工具（llama.cpp 包装器） | Go, C++ | 2026-05 | https://github.com/ollama/ollama |
| **TGI** | ~10k | HuggingFace 推理引擎（已进入维护模式） | Rust, Python | 2026-04 | https://github.com/huggingface/text-generation-inference |
| **vLLM (Ray Serve)** | 集成 | 分布式部署 + 自动扩缩容 | Python, Ray | 2026-05 | https://github.com/ray-project/ray |
| **guidellm** | ~2k | SLO 感知基准测试平台 | Python | 2026-04 | https://github.com/neuralmagic/guidellm |
| **inference-perf** | ~1k | LLM 推理性能基准（wg-serving 标准） | Python | 2026-04 | https://pypi.org/project/inference-perf/ |
| **AIPerf (NVIDIA)** | ~3k | 生产级推理基准工具，支持 OpenTelemetry | Python, CUDA | 2026-04 | https://github.com/ai-dynamo/aiperf |
| **FlashAttention** | ~15k | 高效 Attention 算法实现（v2/v3） | CUDA, C++ | 2026-03 | https://github.com/Dao-AILab/flash-attention |
| **xgrammar** | ~2k | 结构化输出 Grammar 约束推理 | C++, Python | 2026-05 | https://github.com/xlang-ai/xgrammar |
| **llumnix** | ~1k | 多实例 LLM 调度框架 | Python | 2025-12 | https://github.com/llumnix/llumnix |

## 2.2 关键论文（12 篇）

### 经典高影响力（奠基性工作 ~40%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 |
|------|----------|------|----------|---------|-----------|
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** | Kwon et al. (UC Berkeley) | 2023 | SOSP 2023 | 提出 PagedAttention，将 KV Cache 视为虚拟内存分页管理，消除显存碎片，提升 3-5 倍吞吐 | 2500+ 引用，vLLM 核心基础 |
| **Orca: A Distributed Serving System for Transformer-Based Generative Models** | Yu et al. (Microsoft) | 2023 | OSDI 2023 | 首次提出连续批处理（Continuous Batching）和选择性批处理，奠定现代推理系统基础 | 1000+ 引用 |
| **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving** | Zhong et al. (PKU) | 2024 | OSDI 2024 | 首次系统化 Prefill/Decode 物理分离，请求容量提升 7.4 倍，SLO 严格度提升 12.6 倍 | 500+ 引用 |
| **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Dao et al. (Stanford) | 2022 | NeurIPS 2022 | 提出了 IO 感知的精确 Attention 算法，显著减少 HBM 访问，加速 2-4 倍 | 4000+ 引用 |
| **MLP-MoE-like: Mixture of Experts for LLM Serving** | Hwang et al. (Stanford) | 2023 | MLSys 2024 | MoE 模型的推理优化，提出 Expert 动态加载和负载均衡 | 300+ 引用 |

### 最新 SOTA（前沿进展 ~60%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 链接 |
|------|----------|------|----------|---------|------|
| **CascadeInfer: Length-Aware Scheduling of LLM Serving** | - | 2026 | arXiv 2512.19179 | 提出长度感知的多实例流水线调度，延迟降低 67%，吞吐提升 2.89 倍 | [arXiv](https://arxiv.org/html/2512.19179v3) |
| **Nitsum: Serving Tiered LLM Requests with Adaptive Tensor Parallelism** | - | 2026 | arXiv 2605.05467 | 动态调整 TP 度作为运行时控制变量，SLO 合规吞吐提升 5.3 倍 | [arXiv](https://arxiv.org/abs/2605.05467) |
| **AlignedServe: Prefix-aware Batching for LLM Serving** | Bai et al. (SYSU) | 2026 | arXiv 2605.23389 | 按 KV Cache 长度对齐分组的批处理策略，吞吐提升 1.98 倍，延迟降低 7.4 倍 | [arXiv](https://arxiv.org/abs/2605.23389) |
| **PROTEUS: SLA-Aware Routing via Lagrangian RL** | - | 2026 | EuroMLSys 2026 | 拉格朗日对偶强化学习的多 LLM 路由，成本节省达 89.8% | [ACM](https://dl.acm.org/doi/10.1145/3805621.3807607) |
| **Stream2LLM: Overlap Context Streaming and Prefill** | Bachkaniwala et al. | 2026 | arXiv 2604.16395 | 上下文流式传输与 Prefill 重叠，TTFT 改善达 11 倍 | [arXiv](https://arxiv.org/abs/2604.16395) |
| **JITServe: SLO-aware LLM Serving with Imprecise Request Info** | - | 2025 | arXiv 2504.20068 | 利用不精确请求信息优化调度，Goodput 提升 1.4-6.3 倍 | [arXiv](https://arxiv.org/abs/2504.20068) |
| **Taming Request Imbalance: SLO-Aware Scheduling for Disaggregated LLM Inference (Kairos)** | - | 2026 | arXiv 2605.02329 | 紧迫度优先级调度 + 松弛引导自适应批处理，TTFT SLO 达成率提升 23.9% | [arXiv](https://arxiv.org/abs/2605.02329) |

## 2.3 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|---------|----------|------|------|---------|------|
| **Five Techniques to Reach the Efficient Frontier of LLM Inference** | Google Cloud Blog | EN | 架构实践 | 语义路由、PD分离、量化、上下文感知路由、推测解码五大技术；Vertex AI + GKE 生产案例 | 2026-03 |
| **The Great LLM Inference Engine Showdown (2026 Edition)** | Ultradune / Dev.to | EN | 深度评测 | vLLM/TGI/TensorRT-LLM/SGLang/llama.cpp/Ollama 六大引擎全面对比 | 2026-04 |
| **Disaggregation in Large Language Models: the Next Evolution** | InfoQ | EN | 架构综述 | Prefill-Decode 分离架构的行业实践，GPU 利用率提升 40-60%，成本降低 15-40% | 2026-02 |
| **vLLM vs TensorRT-LLM vs SGLang: H100 Benchmarks (2026)** | Spheron Blog | EN | 基准评测 | 三大引擎在 H100 上的吞吐/延迟详细基准，含 Llama-70B 数据 | 2026-03 |
| **破局高可用：大模型服务 SLA 保障与多模型无缝切换架构实践** | 七牛云 | CN | 工程实践 | 统一网关、令牌桶、智能路由、TTFT/TPOT 监控等生产级实践 | 2026-01 |
| **LLM推理引擎：主流框架深度解析与选型指南** | 中国东大 | CN | 选型指南 | 六大推理引擎架构对比与适用场景分析 | 2026-04 |
| **Best LLM Inference Engines in 2026 (Updated April 2026)** | GigaGPU | EN | 评测推荐 | 2026 年推理引擎排行榜，含 RTX 5090 实测数据 | 2026-04 |
| **LLM量化技术全景对比：AWQ、GPTQ、GGUF与FP8/INT8/INT4的抉择指南** | CSDN | CN | 技术教程 | 量化方案的全景对比，含 MMLU 质量测试数据 | 2026-04 |
| **大模型推理框架全景解析：从原理到选型** | 微信公众号 | CN | 架构解析 | 推理框架的技术原理与生产选型决策树 | 2026-05 |
| **Decoding Speculative Decoding** | NAACL 2025 | EN | 论文解读 | 推测解码核心洞察：草稿模型延迟比语言能力更重要 | 2025-04 |

## 2.4 技术演进时间线

| 时间 | 事件 | 发起方 | 影响 |
|------|------|-------|------|
| **2022.06** | FlashAttention 发布 | Stanford | Attention 计算加速 2-4 倍，奠定现代推理基础 |
| **2023.06** | PagedAttention / vLLM 发布 | UC Berkeley | KV Cache 分页管理，消除显存碎片，推理吞吐提升 3-5 倍 |
| **2023.07** | 连续批处理（Orca） | Microsoft | 替代传统静态批处理，GPU 利用率从 ~40% 提升至 ~80%+ |
| **2024.07** | DistServe: Prefill-Decode 分离 | PKU | 物理分离 Prefill 和 Decode，SLO 严格度提升 12.6 倍 |
| **2024.09** | SGLang RadixAttention | Stanford | 前缀树 KV Cache 管理，共享前缀场景命中率 75-95% |
| **2025.02** | DuoServe-MoE | 学术 | MoE 模型双阶段专家预取，TTFT 提升 5.34 倍 |
| **2025.05** | Arrow: 自适应 PD 分离调度 | 学术 | 动态调整 Prefill/Decode 实例比例，服务容量提升 2.55 倍 |
| **2025.12** | CascadeInfer: 长度感知调度 | 学术 | 管道式长度专用调度，延迟降 67%，吞吐升 2.89 倍 |
| **2026.01** | IETF LLM 基准标准化草案 | IETF | 首个 LLM 推理基准标准定义 TTFT/TPOT/ITL 测试方法 |
| **2026.03** | Google Cloud 五大推理技术 | Google | 语义路由+PD分离+量化+上下文路由+推测解码 |
| **2026.04** | Nitsum: 自适应张量并行 | 学术 | TP 作为运行时变量，SLO 合规吞吐提升 5.3 倍 |
| **2026.05** | AlignedServe: 前缀感知批处理 | SYSU | 按 KV Cache 长度对齐批处理，延迟降 7.4 倍 |

---

# 第三部分：方案对比

## 3.1 发展时间线

```
2022 ─┬─ FlashAttention (IO感知Attention) → Attention加速2-4倍，降低HBM瓶颈
2023 ─┬─ PagedAttention (vLLM) → 显存碎片消除，吞吐3-5倍提升
      └─ Continuous Batching (Orca) → GPU利用率从40%→80%+
2024 ─┬─ DistServe (Prefill-Decode分离) → SLO严格度12.6倍改善
      └─ RadixAttention (SGLang) → 前缀共享缓存命中率75-95%
2025 ─┬─ MoE专用优化 (DuoServe-MoE) → 专家预取+缓存，延迟降7.5倍
      └─ Arrow/自适应调度 → 动态实例比例调整
2026 ─┬─ Nitsum (自适应TP) → TP运行时动态切换
      ├─ CascadeInfer (长度感知调度) → 多实例流水线
      ├─ AlignedServe (前缀感知批处理) → KV Cache长度对齐
      └─ PROTEUS/MESS+ (SLA路由) → 强化学习驱动的多模型路由

当前状态：推理服务从"静态配置"全面转向"动态自适应"，SLA保障从经验规则演进为
可数学优化的控制问题，Goodput取代吞吐量成为核心指标。
```

## 3.2 6 种方案横向对比

### 方案 A：PagedAttention + 连续批处理（vLLM 核心方案）

| 维度 | 内容 |
|------|------|
| **原理** | 将 KV Cache 分页管理（类虚拟内存），在解码步边界动态插入新请求的连续批处理机制 |
| **优点** | (1) 显存利用率从 ~40% 提升至 ~90%，消除碎片化；(2) 生态最成熟，社区最大（75k stars）；(3) 硬件支持最广（NVIDIA/AMD/Intel/TPU/Trainium）；(4) 支持 Prefix Caching、Chunked Prefill、CUDA Graph 等多重优化 |
| **缺点** | (1) 原始吞吐低于 TensorRT-LLM 10-30%；(2) 长上下文场景显存效率仍有优化空间；(3) 大规模分布式复杂性较高 |
| **适用场景** | 通用生产部署，需要可靠默认选项的团队 |
| **成本量级** | 中等（开源免费，GPU 需求根据模型规模） |

### 方案 B：Prefill-Decode 分离（Disaggregated Serving）

| 维度 | 内容 |
|------|------|
| **原理** | 将 Prefill（计算密集型）和 Decode（内存带宽密集型）物理分离到不同 GPU 集群，之间通过高速网络传输压缩 KV Cache 状态 |
| **优点** | (1) 两阶段独立优化，GPU 利用率分别达 90-95% 和 20-40%；(2) 基础设施总成本降低 15-40%；(3) 灵活的独立扩缩容；(4) Microsoft Splitwise 在同等成本下吞吐提升 2.35 倍 |
| **缺点** | (1) 增加网络传输延迟和 KV Cache 传输开销；(2) 架构复杂度大幅上升；(3) 需要额外的高速互联网络（InfiniBand/NVLink）；(4) 非 NVIDIA 生态支持有限 |
| **适用场景** | 大规模生产环境，对 SLO 和成本有严格要求的企业 |
| **成本量级** | 高（初始基础设施投入大，但长期可显著降本） |

### 方案 C：推测解码（Speculative Decoding）

| 维度 | 内容 |
|------|------|
| **原理** | 小模型（Draft Model）快速生成 N 个候选 Token，大模型（Target Model）通过一次并行前向传播验证并接受，突破内存带宽限制的逐 Token 生成瓶颈 |
| **优点** | (1) 解码速度 2-3 倍提升，理论可达 5 倍；(2) 与量化/PagedAttention 正交叠加；(3) 无需修改模型架构；(4) Apple Mirror-SD 在异构加速器上实现 2.8-5.8 倍加速 |
| **缺点** | (1) 实现复杂度高，需要维护双模型版本；(2) 草稿模型延迟比语言能力更关键（NAACL 2025 发现）；(3) 高负载场景加速效果衰减；(4) 部分采样策略兼容性有限 |
| **适用场景** | 低并发、单用户场景下的解码加速；对首 Token 延迟敏感的应用 |
| **成本量级** | 中等（需要额外 GPU 运行草稿模型，但减少总体 Token 计算量） |

### 方案 D：权重量化（AWQ/GPTQ/FP8/INT4）

| 维度 | 内容 |
|------|------|
| **原理** | 降低模型权重的数值精度（如 FP16→INT4），减少 GPU 内存读取带宽需求，间接加速解码（解码瓶颈 = 内存带宽） |
| **优点** | (1) 几乎通用，几乎所有模型都能量化；(2) FP8 质量损失 < 0.5%，INT4 AWQ-Marlin 损失 < 1.5%；(3) AWQ-Marlin 在 RTX 4090 上 135 tok/s vs FP16 的 52 tok/s（2.6 倍）；(4) 减少显存需求，支持更大批次 |
| **缺点** | (1) INT4 在数学推理（GSM8K: -4%）和代码（HumanEval: -3.5%）任务上质量损失明显；(2) 需要 GPU 算子的量化内核支持；(3) 不同 GPU 架构的量化方案不通用；(4) 量化校准需要额外计算开销 |
| **适用场景** | 所有推理部署的通用优化步骤，FP8 是高精度场景首选，INT4 是成本优先场景首选 |
| **成本量级** | 低（仅需一次校准，部署后零额外成本） |

### 方案 E：SLO 感知路由（PROTEUS / MESS+ / RAG-based Routing）

| 维度 | 内容 |
|------|------|
| **原理** | 在多个大小不等的模型间进行智能路由——简单请求路由到小模型（低成本、低延迟），复杂请求路由到大模型（高成本、高精度），通过强化学习或在线优化保障 SLA |
| **优点** | (1) PROTEUS 可节省 89.8% 成本；(2) MESS+ 比现有路由技术平均节省 2 倍成本；(3) 可与推理引擎优化正交叠加；(4) 灵活适配多模型生产环境 |
| **缺点** | (1) 需要部署和维护多个模型版本；(2) 路由决策的延迟增加了额外的开销；(3) 冷启动问题——新请求类型的路由策略需要学习；(4) 精度评估的准确性影响路由质量 |
| **适用场景** | 多模型服务（Model Zoo）场景，追求成本效益最大化的企业 |
| **成本量级** | 中-低（需要多模型部署，但显著降低对昂贵大模型的依赖） |

### 方案 F：RadixAttention + 前缀感知批处理（SGLang / AlignedServe）

| 维度 | 内容 |
|------|------|
| **原理** | 使用基数树（Radix Tree）存储所有请求的 KV Cache，实现任意长度的前缀共享；批处理时按 KV Cache 长度对齐分组，消除迭代级气泡 |
| **优点** | (1) 多轮对话前缀缓存命中率 75-95% vs vLLM 10-25%；(2) 结构化输出（JSON/XML）加速 3-10 倍；(3) AlignedServe 解码吞吐提升 1.98 倍，延迟降低 7.4 倍；(4) xAI Grok 3、LMSYS Arena 等大规模使用 |
| **缺点** | (1) 社区规模小于 vLLM，生产 battle-tested 程度较低；(2) 前缀树的内存开销在极端多样化的请求场景下较高；(3) 非前缀共享场景（如随机查询）收益有限 |
| **适用场景** | RAG 流水线、多轮对话、结构化输出、DeepSeek 生态 |
| **成本量级** | 中等（开源，对前缀共享场景成本效益突出） |

## 3.3 技术细节对比

| 维度 | A: PagedAttention | B: PD分离 | C: 推测解码 | D: 量化 | E: SLA路由 | F: RadixAttention |
|------|:-:|:-:|:-:|:-:|:-:|:-:|
| **性能加速** | 3-5×吞吐 | 2.35×吞吐/等成本 | 2-3×解码 | 2-3×吞吐 | 89.8%成本节省 | 2×吞吐/7.4×延迟 |
| **部署难度** | 低 | 高 | 中-高 | 低-中 | 中 | 中 |
| **生态成熟度** | ★★★★★ | ★★★☆ | ★★★☆ | ★★★★ | ★★☆ | ★★★★ |
| **社区活跃度** | ★★★★★ | ★★★ | ★★★ | ★★★★ | ★★ | ★★★★ |
| **硬件依赖** | 通用 | 高端NVIDIA | 通用 | 通用 | 通用 | 通用 |
| **学习曲线** | 低 | 高 | 中 | 低 | 中 | 中-高 |
| **与其它方案叠加** | 可叠加所有 | 可叠加A/C/D/F | 可叠加A/D/F | 可叠加A/C/F | 可叠加所有 | 可叠加A/C/D/E |

## 3.4 选型建议

| 场景 | 推荐方案组合 | 核心理由 | 预估月成本 |
|------|------------|---------|-----------|
| **小型项目/原型验证** | vLLM (FP16) + 单 GPU | 开箱即用，社区支持最好，无需复杂配置 | $500-2,000（1-2×A100/H100） |
| **中型生产环境（API服务）** | vLLM/SGLang + AWQ-INT4 + Prefix Caching | 量化降低 GPU 需求 4 倍，Prefix Caching 缓解高重复请求的 Prefill 开销 | $3,000-10,000（4-8×A100/H100） |
| **大型分布式系统（高SLO要求）** | SGLang/vLLM + PD分离 + FP8 + CascadeInfer调度 | PD分离实现独立扩缩容，FP8保持精度近乎无损，长度感知调度优化延迟 | $20,000-80,000（16-64×H100，含网络） |
| **延迟极度敏感（毫秒级TTFT）** | TensorRT-LLM + FP8 + Speculative Decoding + PD分离 | 编译式引擎裸性能最高，推测解码降低每Token延迟，PD分离避免Prefill干扰 | $30,000-100,000（专属H100集群 + InfiniBand） |
| **成本优先/边缘部署** | llama.cpp + GGUF Q4_K_M + CPU/单GPU | 零GPU成本或消费级GPU即可运行，GGUF格式生态最全 | $100-1,000（CPU/RTX 4090/消费级卡） |
| **多模型路由（Model Zoo）** | PROTEUS/MESS+ + vLLM多实例 + 语义路由 | 拉格朗日对偶RL自动路由，简单请求用小模型节约 >80% 成本 | $5,000-15,000（混合GPU集群，可显著降本） |
| **多轮对话/RAG/结构化输出** | SGLang + RadixAttention + Prefix Caching + AWQ-INT4 | 前缀树缓存命中率遥遥领先，结构化输出加速 3-10 倍 | $3,000-8,000（4-8×A100） |

### 选型决策树

```
开始
├─ 我只有 1 张GPU 或 没有GPU？
│   └─→ llama.cpp / Ollama (GGUF)
├─ 我在小规模原型验证？
│   └─→ vLLM (入门最稳妥)
├─ 我是生产环境，追求成本效益？
│   ├─ 请求重复/共享前缀多？ ──→ SGLang + Prefix Caching
│   └─ 请求随机多样化？ ──→ vLLM + AWQ-INT4
├─ 我需要极高精度 + 低延迟？
│   └─→ vLLM + FP8 + PD分离
├─ 我的延迟要求极其严苛？
│   └─→ TensorRT-LLM + FP8 + 推测解码
├─ 我管理多个模型？
│   └─→ PROTEUS/MESS+ 多模型路由
└─ 我不确定选哪个？
    └─→ vLLM (稳妥默认) → 遇到瓶颈再升级到SGLang/TRT-LLM
```

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{LLM推理SLA保障} = \underbrace{\text{连续批处理 + PagedAttention}}_{\text{最大化GPU利用率}} + \underbrace{\text{Prefill-Decode分离}}_{\text{消除阶段干扰}} - \underbrace{\text{排队等待 + KV Cache传输开销}}_{\text{不可消除的系统损耗}}
$$

**解读**：大模型推理的 SLA 保障本质上是一个"在 GPU 资源和延迟约束之间做最优权衡"的控制问题。连续批处理和 PagedAttention 负责填满 GPU 算力（解决"利用率不足"），PD 分离负责消除 Prefill 和 Decode 之间的互相干扰（解决"阶段冲突"），而排队等待和 KV Cache 传输是系统固有的损耗，需要通过调度优化最小化。

## 4.2 一句话解释

> **大模型推理服务 SLA 保障就像管理一个餐厅后厨**——洗菜备菜（Prefill）和炒菜出餐（Decode）需要分配不同的厨师和设备，但都要在顾客可接受的时间里上菜；排队太长会丢客（SLA 违例），厨房空转会亏钱（GPU 利用率低），好的流程设计就是在这两者之间找到最优平衡。

## 4.3 核心架构图

```
                            SLO约束 (TTFT<200ms, TPOT<50ms)
                                    │
                                    ▼
请求流入 → [网关/路由层] → [SLO感知调度器] → [推理引擎] → 流式输出
  │            │               │               │
  │       语义路由        连-续批处理      PagedAttention
  │       Token限流       Prefill/Decode   KV Cache管理
  │       负载均衡        混合调度         量化算子
  │                                        CUDA Graph
  │               │               │               │
  └───────────────┴───────────────┴───────────────┘
              ▲                               ▲
              │                               │
        [健康检查/熔断]                  [TTFT/TPOT监控]
        [自动扩缩容]                     [Goodput报表]

            核心优化栈（可叠加）：
        ┌─────────────────────────────────────────┐
        │ 量化(AWQ/FP8) × 推测解码 × Prefix Caching│
        │ × 长度感知批处理 × PD分离 × FlashAttention│
        └─────────────────────────────────────────┘
```

## 4.4 STAR 总结

### Situation（背景+痛点）

大模型推理服务正处于从"可用"到"高可用"的关键转折期。2025-2026 年，以 GPT-4o、Claude 3.5、DeepSeek V3 为代表的模型参数已达万亿级规模，推理请求量激增。企业面临的核心矛盾是：**GPU 成本居高不下 vs 用户对延迟的容忍度持续走低**。传统做法——塞满 GPU 跑最大批次——虽能提高吞吐，却导致 P99 TTFT 飙升到数秒。同时，推理集群的 GPU 利用率普遍只有 20-40%（解码阶段），资源浪费严重。

### Task（核心问题）

核心挑战是在 **TTFT < 200ms、TPOT < 50ms** 的双重 SLO 约束下最大化有效产出（Goodput）。具体需解决：(1) Prefill 和 Decode 的资源竞争——前者吃算力，后者吃带宽，放在一起互相拖累；(2) KV Cache 随序列长度线性膨胀——128K 上下文时显存被 Cache 吃光，限制并发；(3) 请求异构性——短 Prompt 和长 Prompt、多轮对话和首次查询对资源需求差异巨大，统一调度导致效率低下。

### Action（主流方案）

行业通过三层递进的优化体系应对挑战：**底层加速**（FlashAttention-3、CUDA Graph、AWQ/FP8 量化）降低单次 Token 生成延迟；**中层调度**（PagedAttention 消除显存碎片、连续批处理提升 GPU 利用率、Chunked Prefill 避免长 Prompt 阻塞）最大化吞吐；**顶层架构**（Prefill-Decode 分离、长度感知多实例流水线、SLO 感知强化学习路由）从系统层面打破瓶颈。2025-2026 年的关键突破是 Nitsum（自适应 TP 运行时切换）、CascadeInfer（管道式长度调度）和 AlignedServe（前缀感知批处理），它们将推理调度从"静态配置"全面转向了"动态自适应"。

### Result（效果+建议）

当前推理延迟可做到 P50 TTFT < 200ms、TPOT < 30ms，Goodput 相比两年前提升 5-10 倍。但前沿挑战依然存在：思维链推理模型（如 o1、DeepSeek R1）的输出长度波动极大，导致 KV Cache 管理和调度更加困难；MoE 模型的稀疏专家激活增加了显存规划的复杂性。**实操建议**：(1) **从 vLLM 起步**是最稳妥的选择，遇到瓶颈再评估 SGLang 或 TensorRT-LLM；(2) **量化是成本最低、收益最高的优化**——FP8 几乎无损，INT4 适合成本敏感场景；(3) **先做架构优化再买更多 GPU**——PD 分离和 Prefix Caching 通常能释放 2-4 倍容量，远比加卡划算。

## 4.5 理解确认问题

**问题**：为什么在相同模型和相同 GPU 上，将 Prefill 和 Decode 物理分离到不同的 GPU 集群（而非在同一 GPU 上混合执行）能同时改善 TTFT 和 TPOT？请从 GPU 资源利用性质的角度解释。

**参考答案**：

Prefill 阶段需要一次性处理数百到数千个输入 Token，计算负载密度极高（200-400 ops/byte），能充分利用 GPU 的 Tensor Core 算力。而 Decode 阶段每步只计算一个 Token，核心瓶颈是读取模型权重和 KV Cache 的内存带宽（60-80 ops/byte），GPU 的计算单元大部分时间处于空闲等待状态。

当二者混合在同一 GPU 上时：(1) Prefill 的长计算会阻塞 Decode 的逐 Token 生成，导致 TPOT 飙升——因为 Decode 请求必须等当前 Prefill 完成后才能进入批次；(2) Decode 的低计算密度又拖慢了 Prefill 的吞吐——因为批处理中有大量等待内存读取的空闲 SM。

物理分离后：(1) Prefill 集群专注高计算密度，GPU 利用率可达 90-95%，最大化 Prefill 吞吐，降低 TTFT；(2) Decode 集群专注高内存带宽利用，每台 GPU 专门负责大量并发序列的逐 Token 生成，TPOT 更稳定；(3) 两者可独立扩缩容，避免了"为 Prefill 配置的 GPU 在 Decode 阶段大量空闲"的资源浪费。这也是为什么 DistServe（OSDI 2024）实现请求容量提升 7.4 倍、Splitwise（Microsoft）实现等成本吞吐提升 2.35 倍的根本原因。

---

> **本报告基于 2026 年 5 月公开信息编写。技术发展迅速，建议持续关注各推理引擎的 GitHub Release 和顶级会议（OSDI、SOSP、MLSys、EuroSys）的最新论文。**
