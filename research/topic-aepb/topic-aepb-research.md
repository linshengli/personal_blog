# 大模型推理分离架构与预填充解码优化 — 深度调研报告

> **调研主题**：大模型推理分离架构（Prefill-Decode Disaggregation）与预填充解码优化
> **所属领域**：大模型框架 / LLM Inference Serving
> **调研日期**：2026-05-13
> **报告类型**：四维度全景分析 + 精华整合

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

**大模型推理分离架构（Prefill-Decode Disaggregation, PD Disaggregation）** 是一种将 LLM 推理过程的两个阶段——预填充（Prefill）和解码（Decode）——分配到不同 GPU 或不同计算节点上独立执行的系统架构。预填充阶段并行处理整个输入提示（prompt），生成 KV Cache；解码阶段逐 token 自回归生成输出，反复读取 KV Cache。分离式架构的核心动机是二者资源需求截然不同：预填充是**计算密集型**（受限于 GPU TFLOPS），解码是**内存带宽密集型**（受限于 GPU HBM 带宽）。将二者解耦后可以独立优化、独立扩缩容，从而显著提升整体系统吞吐和延迟满足率。

### 常见误解

1. **误解一：分离架构等同于分布式推理**
   分布式推理（Tensor Parallelism, Pipeline Parallelism）解决的是"单 GPU 放不下模型"的问题，将模型参数切分到多卡；而 PD 分离解决的是"预填充和解码互相干扰"的问题，并不要求模型必须跨卡部署。二者是正交的优化手段。

2. **误解二：预填充和解码可以完全独立于同一批请求**
   实际上分离架构带来了**显存状态转移**（KV Cache 传输）的新开销。如果在物理分离的节点间传输 KV Cache 的网络带宽不足，可能抵消分离带来的收益。这也是当前研究的核心矛盾之一。

3. **误解三：PD 分离只对超大模型有效**
   事实上，对于 7B-13B 规模的中等模型，在长上下文（>4K tokens）或高并发场景下，PD 分离同样能带来显著收益。主要瓶颈在于模型大小决定 KV Cache 体积，而非模型参数规模本身。

### 边界辨析

| 易混淆概念 | 核心区别 |
|-----------|---------|
| **PD 分离 vs 连续批处理（Continuous Batching）** | 连续批处理是在单 GPU 上交错执行不同请求的预填充和解码，以减少 GPU 空闲时间；PD 分离则是将两个阶段完全分到不同 GPU，消除阶段间抢占干扰 |
| **PD 分离 vs 分块预填充（Chunked Prefill）** | 分块预填充（如 Sarathi-Serve）将预填充切成小块，交错插入解码迭代中以减少头阻塞，本质仍是单 GPU 上的时间片调度；PD 分离是空间上的隔离 |
| **PD 分离 vs KV Cache 卸载（KV Offloading）** | KV 卸载是将 Cache 从 GPU HBM 移到 CPU DRAM 或磁盘以节省显存；PD 分离必然涉及 KV Cache 在节点间的网络传输 |

## 1.2 核心架构

```
┌──────────────────────────────────────────────────────────────────┐
│                   PD 分离推理系统架构 (Disaggregated Serving)      │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────┐           │
│  │  Router   │───>│  Prefill     │───>│  Decode      │           │
│  │ (请求路由) │    │  Pool       │    │  Pool        │           │
│  │          │    │ (计算密集)    │    │ (内存带宽密集) │           │
│  └──────────┘    └──────┬───────┘    └──────┬───────┘           │
│         │               │                   │                   │
│         │               │    KV Cache       │                   │
│         │               │◄─── Transfer ────►│                   │
│         │               │   (RDMA/NVLink)   │                   │
│         │               ▼                   ▼                   │
│         │        ┌──────────────┐    ┌──────────────┐           │
│         └────────┤  KV Cache    │    │  Monitor &   │           │
│                  │  Store       │    │  Autoscaler  │           │
│                  │ (分布式缓存)  │    │ (SLO追踪/扩缩容)│          │
│                  └──────────────┘    └──────────────┘           │
│                                                                  │
│  KV Cache 传输后端: Mooncake / NIXL / NCCL / TCP                 │
└──────────────────────────────────────────────────────────────────┘
```

**组件说明**：
- **Router**：接收用户请求，根据请求特征（上下文长度、缓存命中状态、SLO 要求）将请求路由到合适的 Prefill 节点或直接到 Decode 节点（缓存命中时）
- **Prefill Pool**：一组 GPU，专门处理预填充计算——并行处理 prompt tokens，生成 KV Cache，完成后通过高速网络将 KV Cache 发送到 Decode 节点
- **Decode Pool**：一组 GPU，专门处理自回归解码——接收 KV Cache 后逐 token 生成输出，具有低延迟要求
- **KV Cache Store**：分布式 KV 缓存层（跨 GPU DRAM/CPU DRAM/NVMe），支持缓存复用，减少重复预填充
- **Monitor & Autoscaler**：监控 TTFT/TPOT/SLO 达标率，自动调节 Prefill/Decode 节点配比

## 1.3 数学形式化

### (1) 推理阶段的计算特征

$$
T_{\text{total}} = T_{\text{prefill}} + T_{\text{decode}} = \frac{2 \cdot N_{\text{params}} \cdot S}{B_{\text{compute}}} + \frac{2 \cdot N_{\text{params}} \cdot N_{\text{gen}}}{B_{\text{memory}}}
$$

其中 $N_{\text{params}}$ 为模型参数量，$S$ 为输入序列长度，$N_{\text{gen}}$ 为生成 token 数，$B_{\text{compute}}$ 为 GPU 计算带宽（TFLOPS），$B_{\text{memory}}$ 为 GPU 显存带宽（GB/s）。该公式量化说明预填充延迟与输入长度成正比（计算受限），解码延迟与生成 token 数成正比（访存受限）。

### (2) KV Cache 传输开销

$$
T_{\text{transfer}} = \frac{2 \cdot N_{\text{layers}} \cdot N_{\text{heads}} \cdot d_{\text{head}} \cdot S \cdot 2 \text{ bytes}}{B_{\text{network}}}
$$

每层每注意力头的 KV Cache 大小为 $2 \times N_{\text{heads}} \times d_{\text{head}} \times S$（K 和 V 各一份），FP16 精度下每元素 2 字节。传输时间由网络带宽 $B_{\text{network}}$ 决定。对于 Llama 70B 在 4K 上下文下，KV Cache 约 1.34 GB，100GbE 网络下传输约 110ms——这是分离架构的关键开销。

### (3) Goodput 度量

$$
\text{Goodput} = \frac{\text{在 SLO 内完成的请求数}}{\text{总时间}} = \frac{\sum_{i=1}^{N} \mathbf{1}\{\text{TTFT}_i \leq \text{SLO}_{\text{TTFT}} \land \text{TPOT}_i \leq \text{SLO}_{\text{TPOT}}\}}{T_{\text{total}}}
$$

Goodput 是 PD 分离领域最核心的度量指标，同时约束首 token 延迟（TTFT）和每 token 延迟（TPOT），比原始吞吐更能反映用户体验。

### (4) 资源分配优化模型

$$
\min_{N_p, N_d} \left( \frac{\lambda_p}{N_p} + \frac{\lambda_d}{N_d} \right) \quad \text{s.t.} \quad N_p + N_d \leq N_{\text{total}},\ \text{SLO 达标率} \geq 99\%
$$

$N_p$ 和 $N_d$ 分别为预填充和解码的 GPU 数量。$\lambda_p, \lambda_d$ 为两阶段的请求到达率。该优化问题描述了在总 GPU 预算下如何分配 prefill/decode 配比以最小化延迟。

### (5) 缓存复用率影响

$$
\text{Speedup} = \frac{1}{1 - r_{\text{cache}} + r_{\text{cache}} \cdot \frac{T_{\text{prefill,full}}}{T_{\text{prefill,append}}}}
$$

其中 $r_{\text{cache}}$ 为缓存命中率，$T_{\text{prefill,full}}$ 为完整预填充时间，$T_{\text{prefill,append}}$ 为追加预填充时间。当缓存命中率 > 90%（如智能体场景），加速比可达 5-10 倍。

## 1.4 实现逻辑（Python 伪代码）

```python
class DisaggregatedServingSystem:
    """PD 分离推理系统的核心抽象"""

    def __init__(self, config: DisaggConfig):
        self.router = AdaptiveRouter(config.routing_policy)
        self.prefill_pool = GPUPool(
            num_gpus=config.num_prefill_gpus,
            phase="prefill",
            kv_cache_transport=config.kv_transport_backend  # Mooncake/NIXL
        )
        self.decode_pool = GPUPool(
            num_gpus=config.num_decode_gpus,
            phase="decode",
            kv_cache_transport=config.kv_transport_backend
        )
        self.kv_cache_store = DistributedKVCache(
            tiers=["gpu_hbm", "cpu_dram", "nvme"],
            eviction_policy="lru"
        )
        self.monitor = SLOAwareMonitor(
            ttft_slo=config.ttft_slo,   # e.g. 500ms
            tpot_slo=config.tpot_slo    # e.g. 50ms
        )

    def serve_request(self, request: Request) -> Response:
        """请求处理主流程"""
        # Step 1: 路由决策
        route_plan = self.router.decide_route(
            prompt=request.prompt,
            cached_kv_id=self.kv_cache_store.lookup(request.prompt_prefix)
        )

        if route_plan.is_cache_hit:
            # 缓存命中 → 直接发送到 Decode Pool（跳过 Prefill）
            kvcache = self.kv_cache_store.fetch(route_plan.cache_key)
            return self.decode_pool.generate(request, kvcache)

        # Step 2: Prefill 阶段（计算密集型）
        prefill_result = self.prefill_pool.execute(
            phase="prefill",
            prompt=request.prompt,
            output_kv_cache=True
        )

        # Step 3: KV Cache 传输（关键瓶颈）
        kv_cache = prefill_result.kv_cache
        self._transfer_kv_cache(kv_cache, from_pool=self.prefill_pool,
                                to_pool=self.decode_pool)

        # Step 4: Decode 阶段（内存带宽密集型）
        response = self.decode_pool.execute(
            phase="decode",
            input_ids=request.tokenized_prompt,
            kv_cache=kv_cache
        )

        # Step 5: 缓存 KV 供后续复用
        self.kv_cache_store.store(kv_cache, ttl=request.session_ttl)

        # Step 6: SLO 监控
        self.monitor.record(request.id, ttft=... , tpot=...)
        return response

    def _transfer_kv_cache(self, kv_cache, from_pool, to_pool):
        """通过高速网络传输 KV Cache（RDMA 零拷贝优先）"""
        transport = from_pool.get_transport(to_pool)
        transport.async_send(kv_cache)  # 流水线化：边传输边解码
```

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **TTFT**（Time to First Token） | < 500ms（对话）/ < 5s（长文本） | 端到端请求计时，从请求到达至第一个输出 token 生成 | 预填充阶段的核心延迟指标 |
| **TPOT**（Time Per Output Token） | < 50ms（对话）/ < 200ms（推理） | 解码阶段每生成一个 token 的平均时间 | 受解码 GPU 内存带宽和 KV Cache 大小直接影响 |
| **ITL**（Inter-Token Latency） | < 30ms | 连续两个输出 token 之间的时间间隔 | 类似 TPOT，强调流式体验 |
| **Goodput**（有效吞吐） | > 90% SLO 达标率 | 在 SLO 约束内完成的请求数 / 总时间 | PD 分离的核心度量指标 |
| **KV Cache 传输延迟** | < 100ms (100GbE) / < 10ms (NVLink) | 从 Prefill GPU 到 Decode GPU 的 KV Cache 传输时间 | 决定分离架构是否可行的关键瓶颈 |
| **缓存命中率** | > 80%（智能体场景 > 95%） | 请求前缀在 KV Cache Store 中命中的比例 | DeepSeek DualPath 报告智能体场景可达 95%+ |

## 1.6 扩展性与安全性

### 水平扩展

- **Prefill 池独立扩容**：根据 TTFT SLO 和请求到达率，独立增加 Prefill 节点数，无需增加 Decode 节点，反之亦然。NVIDIA Dynamo 和 SGLang 均支持 K8s 原生的独立扩缩容。
- **多级 KV Cache 层级**：GPU HBM → CPU DRAM → NVMe → 分布式缓存（RDMA 网络），形成类似 CDN 的多级缓存架构（如 Together AI 的 CPD 方案）。
- **全局路由负载均衡**：Router 层根据各节点负载、缓存状态、网络拓扑智能分发请求，支持在线和离线混合部署。

### 垂直扩展

- **单节点 Prefill 优化**：使用更大的 batch size、更高的 Tensor Parallelism 度来压满 GPU 计算单元。
- **单节点 Decode 优化**：优化重点在 HBM 带宽利用率，如 PageAttention、FlashAttention 减少显存碎片和访存次数。单节点解码吞吐上限由内存带宽决定。
- **KV Cache 压缩**：CacheGen 等技术可将 KV Cache 压缩 3.5-4.3× 以减轻传输和存储压力。

### 安全考量

- **KV Cache 泄漏风险**：KV Cache 包含用户输入的语义信息，跨节点传输时需加密。在多租户环境下，隔离不同用户的 KV Cache 至关重要。
- **SLA 攻击（SLO 饥饿）**：攻击者可以发送超长提示消耗 Prefill 节点资源，导致正常请求 TTFT 超标。需要请求级隔离和公平调度策略。
- **网络级 DoS**：KV Cache 传输可能占满 RDMA 网络带宽，影响其他推理流量。DeepSeek DualPath 使用 QoS（VL/TC）为推理通信保留 99% 优先级。
- **缓存毒化（Cache Poisoning）**：恶意用户构造特定前缀污染 KV Cache，导致后续用户请求产生错误输出。需实现缓存内容校验和时效管理。

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | ~79,000 | 高性能 LLM 推理引擎，原生支持 PD 分离（v0.8+） | Python, CUDA, Triton | 2026-05（每日更新） | [GitHub](https://github.com/vllm-project/vllm) |
| **SGLang** | ~27,000 | 前沿推理引擎，支持 PD/EPD 分离、RadixAttention | Python, CUDA, Triton | 2026-05（每日更新） | [GitHub](https://github.com/sgl-project/sglang) |
| **Mooncake** | ~4,900 | 以 KV Cache 为中心的 PD 分离传输引擎（FAST 2025 最佳论文） | C++, CUDA, RDMA | 2026-04 | [GitHub](https://github.com/kvcache-ai/Mooncake) |
| **LMCache** | ~7,800 | KV 缓存加速层，支持多级存储和跨实例共享 | Python, CUDA | 2026-05 | [GitHub](https://github.com/LMCache/LMCache) |
| **NVIDIA Dynamo** | N/A（闭源但可获取） | 生产级 PD/EPD 分离框架，1.0 版于 2026-03 发布 | C++, Python | 2026-05 | [NVIDIA 博客](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) |
| **NVIDIA NIXL** | 开源组件 | KV Cache 点对点传输库，支撑 Dynamo 的分离架构 | C++, CUDA | 2026-04 | [NVIDIA 博客](https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/) |
| **nanoPD** | ~200 | 从零实现的 PD 分离推理引擎，含分析成本模型自适应路由器 | Python, CUDA | 2026-03 | [GitHub](https://github.com/HJCheng0602/nanoPD) |
| **EPDServe** | ~100 | EPD 三阶段分离（编码-预填充-解码），ICML 2025 | Python, CUDA, vLLM | 2025-05 | [GitHub](https://github.com/vbdi/epdserve) |
| **Sarathi-Serve** | ~490 | 分块预填充、无气泡调度，OSDI 2024 （微软研究院） | Python, CUDA | 2025-06 | [GitHub](https://github.com/microsoft/sarathi-serve) |
| **DistServe** | ~300 | PD 分离开创性实现，OSDI 2024（北大 + UCSD） | Python, CUDA | 2024-10 | [GitHub](https://github.com/LLMServe/DistServe) |
| **Tetris** | ~50 | 细粒度序列并行 + PD 分离的集成系统（2025） | Python, CUDA | 2025-11 | [arXiv](https://arxiv.org/abs/2511.06247) |
| **TokenScale** | ~50 | 基于 Token Velocity 的 PD 分离自动扩缩容 | Python | 2025-12 | [arXiv](https://arxiv.org/abs/2512.03416) |
| **LoongServe** | ~100 | 弹性序列并行（ESP），SOSP 2024 | Python, CUDA | 2024-10 | [GitHub](https://github.com/loongserve/loongserve) |
| **ARES** | ~30 | 自适应解码重调度 + 输出长度预测（2025） | Python | 2025-10 | [arXiv](https://arxiv.org/html/2510.13668v1) |
| **TaiChi** | ~50 | 聚合 + 分离的统一框架，差异化 GPU 实例（2025） | Python | 2025-08 | [arXiv](https://arxiv.org/html/2508.01989) |

## 2.2 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 | 链接 |
|------|----------|------|----------|---------|-----------|------|
| **Splitwise: Efficient Generative LLM Inference Using Phase Splitting** | Microsoft Research | 2024 | ISCA 2024 | 首次系统提出 PD 阶段拆分概念，验证跨硬件分离可行性 | 引用 300+ | [arXiv](https://arxiv.org/abs/2311.18677) |
| **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving** | 北大 + UCSD | 2024 | OSDI 2024 | 定义 Goodput 指标，提出 PD 分离的 SLO 感知优化框架，实现 4.48× 请求量提升 | 引用 350+ | [arXiv](https://arxiv.org/abs/2401.09670) |
| **Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference** | Microsoft Research | 2024 | OSDI 2024 | 提出分块预填充和 Stall-Free 调度，Falcon-180B 达 6.9× 容量提升 | 引用 250+ | [arXiv](https://arxiv.org/abs/2403.02310) |
| **LoongServe: Efficiently Serving Long-Context LLMs with Elastic Sequence Parallelism** | 北大 + 上海 AI Lab | 2024 | SOSP 2024 | 弹性序列并行，比 chunked prefill 提升 3.85× | 引用 150+ | [arXiv](https://arxiv.org/abs/2404.09526) |
| **Mooncake: A KVCache-Centric Disaggregated Architecture** | Moonshot AI（月之暗面） | 2025 | FAST 2025 **Best Paper** | KV Cache 为中心的分离架构，吞吐提升 525% | 最佳论文奖 | [官网](https://kvcache-ai.github.io/Mooncake/) |
| **CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion** | 芝加哥大学 | 2025 | EuroSys 2025 **Best Paper** | 非前缀 KV 重用，无需前缀匹配，RAG 场景 3× 加速 | 最佳论文奖 | [LMCache](https://github.com/LMCache/LMCache) |
| **EPDServe: Efficiently Serving Large Multimodal Models Using EPD Disaggregation** | VBD AI | 2025 | ICML 2025 | 提出编码-预填充-解码三阶段分离 | 顶会论文 | [arXiv](https://arxiv.org/abs/2501.05460) |
| **Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode** | — | 2025 | arXiv | GPU 内动态资源分区，2.2× vLLM 吞吐 | 新兴工作 | [arXiv](https://ui.adsabs.harvard.edu/abs/2025arXiv250706608S) |
| **TaiChi: Prefill-Decode Aggregation or Disaggregation? Unifying Both** | — | 2025 | arXiv | 统一聚合与分离，差异化 GPU 实例，goodput 提升 77% | 新兴工作 | [arXiv](https://arxiv.org/html/2508.01989) |
| **DUET: Disaggregated Hybrid Mamba-Transformer LLMs** | — | 2026 | DAC 2026 | 专用硬件封装（Prefill/Decode 分别优化芯片设计），4× TTFT 加速 | 顶会论文 | [arXiv](https://arxiv.org/abs/2603.15530) |
| **PPD: Not All Prefills Are Equal** | — | 2026 | arXiv | 区分全量预填充和追加预填充，PPD 路由减少 48-73% TTFT | 新兴工作 | [arXiv](https://arxiv.org/html/2603.13358v2) |
| **DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference** | DeepSeek + 北大 + 清华 | 2026 | arXiv | 双路径 KV-Cache 加载，利用解码引擎闲置网卡，吞吐提升 1.96× | 高关注（DeepSeek V4 前奏） | [arXiv](https://arxiv.org/abs/2602.21548) |

## 2.3 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| **Cache-Aware Prefill-Decode Disaggregation (CPD)** | Together AI Blog | 英文 | 架构解析 | 三级 KV Cache 层级 + 冷热预填充分离，QPS 提升 35-40% | 2025-2026 | [链接](https://www.together.ai/blog/cache-aware-disaggregated-inference) |
| **How NVIDIA Dynamo 1.0 Powers Multi-Node Inference** | NVIDIA Technical Blog | 英文 | 官方技术博客 | Dynamo 1.0 架构详解：KV-aware Routing、NIXL、Planner | 2026-03 | [链接](https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/) |
| **Removing the Guesswork from Disaggregated Serving** | NVIDIA Technical Blog | 英文 | 深度教程 | AIConfigurator 自动优化 PD 分离配比 | 2026 | [链接](https://developer.nvidia.com/blog/removing-the-guesswork-from-disaggregated-serving/) |
| **Disaggregated Inference, Part 1: When & Where to Route** | Momento Blog | 英文 | 系列教程 | PD 分离的路由策略深度分析 | 2025-2026 | [链接](https://www.gomomento.com/blog/disaggregated-inference-part-1-when-and-where-to-route/) |
| **Disaggregated Inference, Part 2: Moving the KV Cache Without Stalling** | Momento Blog | 英文 | 系列教程 | KV Cache 传输优化策略和 Mooncake 实战 | 2026 | [链接](https://www.gomomento.com/blog/moving-the-kv-cache-without-stalling-the-decode/) |
| **Splitting LLM Inference Across Different Hardware Platforms** | Gimlet Labs | 英文 | 技术实践 | B200 + Gaudi 3 混合厂商 PD 分离部署，TCO 优化 1.7-4× | 2025 | [链接](https://gimletlabs.ai/blog/multivendor-prefill-decode-disaggregation) |
| **Unleashing AMD MI300X for LLM Serving: Disaggregating Prefill & Decode with SGLang** | AMD ROCm Blog | 英文 | 官方实践指南 | AMD GPU + SGLang 的 PD 分离实现与调优 | 2025 | [链接](https://rocm.blogs.amd.com/software-tools-optimization/disaggregation/README.html) |
| **Benchmarking Prefill-Decode Ratios: Fixed vs Dynamic** | dstack Blog | 英文 | 性能评测 | 1:3 固定配比的实践评测与是否值得动态调度的探讨 | 2025 | [链接](https://dstack.ai/blog/benchmarking-pd-ratios/) |
| **SGLang 推理引擎的技术要点与部署实践** | InfoQ (中文) | 中文 | 深度技术分享 | SGLang PD 分离原理与大规模部署实践（400K+ GPU） | 2025 | [链接](https://www.infoq.cn/article/dwt6bw3mh76rfs5hu0kk) |
| **LMCache：基于 KV 缓存复用的 LLM 推理优化方案** | 阿里云开发者社区 | 中文 | 深度教程 | LMCache 的多级缓存架构 + PD 分离集成详解 | 2025 | [链接](https://developer.aliyun.com/article/1691956) |

## 2.4 技术演进时间线

```
2023.11 ─┬─ Splitwise (Microsoft) 提出 PD 阶段拆分概念
2024.01 ─┼─ DistServe (北大+UCSD) 定义 Goodput，OSDI'24
2024.03 ─┼─ Sarathi-Serve (Microsoft) 提出分块预填充 + Stall-Free 调度，OSDI'24
2024.04 ─┼─ LoongServe (北大) 弹性序列并行，SOSP'24
2024.06 ─┼─ Splitwise 贡献被合并入 vLLM (PR #2809)
2024.09 ─┼─ CacheGen (芝加哥大学) KV 缓存压缩与流式传输，SIGCOMM'24
2024.11 ─┼─ Mooncake (月之暗面) 开源传输引擎
2025.02 ─┼─ Mooncake 获 FAST 2025 最佳论文奖 → 集成 vLLM/SGLang/TensorRT-LLM
2025.03 ─┼─ SGLang 发布 v0.4.5，原生支持 PD 分离
2025.05 ─┼─ EPDServe (ICML'25) 提出三阶段 EPD 分离，支持多模态
2025.07 ─┼─ Nexus 提出 GPU 内 PD 动态资源分区
2025.08 ─┼─ TaiChi 统一聚合与分离，差异化 GPU 实例
2025.09 ─┼─ NVIDIA 开源 NIXL 传输库；Dynamo 公开
2025.10 ─┼─ ARES：基于输出长度预测的自适应解码重调度
2025.12 ─┼─ TokenScale：基于 Token Velocity 的自动扩缩容
2026.01 ─┼─ Red Hat OpenShift AI 3.0 集成 PD 分离
2026.02 ─┼─ DeepSeek DualPath：双路径 KV-Cache 加载，打破 I/O 瓶颈
2026.02 ─┼─ PrefillShare：多模型共享预填充模块
2026.03 ─┼─ NVIDIA Dynamo 1.0 GA：生产级 PD/EPD 分离
2026.03 ─┼─ DUET (DAC'26) 专用硬件封装，4× TTFT
2026.05 ─┴─ 当前状态：PD 分离进入生产成熟期，从"要不要分离"进入"如何精细化分离"阶段
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
2023    ─┬─ Splitwise 提出分离设想：prefill 和 decode 跑在不同 GPU
         │   → 核心洞察：prefill 计算密集、decode 访存密集
2024    ─┼─ 两个并行突破方向：
         │   ├─ 物理分离路线：DistServe (OSDI'24)、Mooncake (FAST'25)
         │   │   → 证明分离可大幅提升 goodput
         │   └─ 分时交错路线：Sarathi-Serve (OSDI'24) chunked prefill
         │       → 不分离硬件但消除头部阻塞
2025    ─┼─ 分化加剧与融合：
         │   ├─ 新维度：EPD 三阶段分离 (ICML'25)、GPU 内分离（Nexus）
         │   ├─ 工程化：vLLM/SGLang 原生支持、NVIDIA Dynamo 公开
         │   └─ 融合：TaiChi 统一聚合与分离、冷热分离（CPD）
2026    ─┴─ 当前状态：生产级成熟（Dynamo 1.0 + 多后端）；前沿探索从"是否分离"转向
         "分离粒度"（每请求级/每 token 级）和"传输瓶颈"（DeepSeek DualPath）
```

## 3.2 6 种方案横向对比

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **① 全聚合（Colocated Baseline）** | 预填充和解码在相同 GPU 上交错执行，使用连续批处理 | 1. 架构简单，无需网络传输<br>2. KV Cache 本地访问，延迟最低<br>3. 单节点即可部署，运维成本低<br>4. 对小模型友好 | 1. 预填充与解码强干扰，头部阻塞严重<br>2. 长提示抢占解码时间片，TTFT 抖动大<br>3. 无法独立优化两阶段，GPU 利用率受限<br>4. 扩缩容需同时增加 P/D 资源 | 小模型 (<7B)、短文本（<512 tokens）、开发测试环境 | $1-3K/月（单节点） |
| **② 分块预填充（Chunked Prefill, Sarathi-Serve）** | 将预填充切成小块，交错插入解码迭代中，消除头部阻塞 | 1. 软件纯优化，无需额外硬件<br>2. 在单 GPU 内显著降低 TTFT 抖动<br>3. 保持架构简单，兼容连续批处理<br>4. Falcon-180B 达 6.9× 容量提升 | 1. 分块粒度需精细调优，不当反增开销<br>2. 无法从根本上解耦 P/D 资源<br>3. 长上下文下分块仍会干扰解码<br>4. 不支持异构硬件优化 | 中等规模部署（7B-70B）、延迟敏感但预算有限 | $3-10K/月（单/双节点） |
| **③ 物理 PD 分离（Inter-GPU Disagg, DistServe/Dynamo）** | Prefill 和 Decode 分配到不同 GPU 池，通过高速网络传输 KV Cache | 1. 彻底消除 P/D 干扰<br>2. 独立扩缩容，资源效率最高<br>3. 支持异构 GPU 配比（H100 prefill + A100 decode）<br>4. goodput 可达聚合方案的 4-7× | 1. 依赖高速网络（100GbE+/RDMA）<br>2. KV Cache 传输增加额外延迟（50-200ms）<br>3. 系统复杂度高，运维成本大<br>4. 小模型或短文本场景收益甚微 | 大规模生产环境（70B+）、长上下文、严格 SLO | $20-100K+/月（多节点集群） |
| **④ GPU 内动态分离（Intra-GPU, Nexus/MuxWise）** | 在同一 GPU 内部动态划分计算和显存带宽资源，prefill 和 decode 按需占用 | 1. 无需跨节点传输，无网络开销<br>2. 资源利用率高，GPU 无闲置<br>3. 比物理分离更灵活<br>4. 2.2× vLLM 吞吐、20× TTFT 降低 | 1. 实现复杂，需硬件级资源控制<br>2. GPU 内部资源池化上限受单卡限制<br>3. 仍在研究阶段，缺乏生产部署<br>4. 对 GPU 架构有特定要求 | 中规模（13B-70B）、对延迟敏感、单 GPU 集群 | $5-20K/月（中规模集群） |
| **⑤ EPD 三阶段分离（EPDServe）** | 将编码（Encoder）、预填充、解码完全独立，适用于多模态大模型 | 1. 解决多模态推理的编码瓶颈（图像/视频处理）<br>2. 三阶段独立扩缩容，灵活性最高<br>3. 峰值显存降低 15×，batch size 提升 22×<br>4. 支持动态 GPU 资源再分配 | 1. 系统复杂度极高<br>2. 需要至少 4-8 个 GPU<br>3. 对纯文本模型过度设计<br>4. 编码阶段和预填充阶段的同步协调复杂 | 多模态大模型（LMM）、图像/视频理解应用、多智能体系统 | $40-150K+/月（大规模集群） |
| **⑥ 混合聚合-分离（TaiChi/PPD）** | 根据请求特征（缓存命中、SLO 松紧）动态决定聚合还是分离 | 1. 兼顾场景多样性，goodput 最优<br>2. PPD 对多轮对话 append-prefill 路由优化<br>3. TaiChi 实现 77% goodput 提升<br>4. 统一管理，降低运维复杂度 | 1. 路由策略设计极其复杂<br>2. 依赖精确的请求特征预测<br>3. 实现难度大，社区生态不成熟<br>4. 需要大量实际负载数据调优 | 混合负载场景（同时有对话、长文本、智能体应用） | $30-80K/月（大规模集群） |

## 3.3 技术细节对比

| 维度 | 全聚合 | 分块预填充 | 物理 PD 分离 | GPU 内分离 | EPD 三阶段 | 混合聚合-分离 |
|------|--------|-----------|-------------|-----------|-----------|-------------|
| **性能（goodput）** | 基线 (1×) | 2-6× | 4-7× | 2-3× | 3-5×（多模态） | 4-8× |
| **TTFT P99** | 500ms-2s | 200-500ms | 80-200ms | 100-300ms | 100-500ms | 80-300ms |
| **解码 ITL P99** | 30-100ms | 20-50ms | 10-30ms | 15-40ms | 10-30ms | 10-30ms |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐ | ⭐⭐ |
| **生态成熟度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐ | ⭐ |
| **社区活跃度** | vLLM/SGLang | Sarathi-Serve | Dynamo/vLLM | 学术研究 | 学术研究 | 学术研究 |
| **学习曲线** | 低 | 中 | 高 | 极高 | 极高 | 极高 |
| **网络要求** | 无 | 无 | 100GbE+/RDMA | NVLink | 100GbE+/RDMA | 100GbE+/RDMA |
| **推荐 GPU 数** | 1-4 | 1-8 | 4-64+ | 1-8 | 4-32+ | 8-64+ |
| **运维复杂度** | 低 | 低 | 高 | 中 | 高 | 极高 |

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本（GPU 租赁） |
|------|---------|---------|-------------------|
| **小型项目 / 原型验证**<br>（< 7B 模型，短文本，低并发） | 全聚合（基线） | 架构简单，部署快，无需网络配置；小模型下 PD 分离收益有限 | $1-3K/月<br>（1× H100） |
| **中型生产环境**<br>（13B-70B，中等并发，4K+ 上下文） | 分块预填充（Sarathi-Serve 风格）<br>或 SGLang + 轻量级 PD 分离 | 获取关键 latency 改善的同时控制复杂度；SGLang 原生支持一键切换 P/D 模式 | $10-30K/月<br>（2-8× GPU） |
| **大型生产 / 在线服务**<br>（70B-180B，高并发，严格 SLO） | 物理 PD 分离<br>（NVIDIA Dynamo / vLLM + Mooncake） | 独立扩缩容、好 SLO 达标率、支持异构 GPU 混用（H100 prefill + A100 decode）、显著 TCO 优化 | $30-100K/月<br>（8-32× GPU + 100GbE） |
| **多模态大模型服务**<br>（VLM，多图/视频输入） | EPD 三阶段分离<br>（EPDServe / SGLang EPD） | 编码阶段独立扩缩容避免 ViT 瓶颈；峰值显存降低 15×，batch size 提升 22× | $40-150K/月<br>（8-32× GPU） |
| **智能体 / 多轮对话**<br>（极高缓存命中率 > 90%） | 混合聚合-分离（PPD 路由 + DualPath） | 缓存命中时跳过 prefill 直送 decode；DualPath 双路径加载解决 I/O 饱和；DeepSeek 实测吞吐提升 1.96× | $50-200K/月<br>（大规模集群 32+ GPU） |

### 选型决策树

```
模型大小?
├─ < 7B        → 全聚合（无需分离）
├─ 7B-70B ──── 上下文长度?
│              ├─ < 2K → 分块预填充
│              └─ > 2K → 轻量级 PD 分离（SGLang）
└─ > 70B ───── 应用场景?
               ├─ 在线对话 → 物理 PD 分离（Dynamo/vLLM）
               ├─ 多模态   → EPD 三阶段分离
               ├─ 智能体   → 混合聚合-分离 + 缓存优化
               └─ 离线批处理 → 物理 PD 分离（prefill 优先，decode 可略弱）
```

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{PD 分离架构} = \underbrace{\text{Prefill 专用计算池}}_{\text{填满 GPU TFLOPS}} + \underbrace{\text{Decode 专用访存池}}_{\text{跑满 HBM 带宽}} - \underbrace{\text{KV Cache 传输开销}}_{\text{分离的代价——网络是新的瓶颈}}
$$

**核心心智模型**：PD 分离的本质是用**空间换时间**——用一个额外的 KV Cache 网络传输开销换取两个阶段的独立优化空间。当传输成本小于干扰成本时，净收益为正。

## 4.2 一句话解释

**大模型推理分离架构，就是把 LLM 思考问题（预填充）和回答问题的过程（解码）分给不同的 GPU 去执行，让擅长计算的 GPU 负责"思考"，擅长读内存的 GPU 负责"回答"，互不干扰地并行工作。**

## 4.3 核心架构图

```
用户请求 → [Router 路由决策]
               │
        ┌──────┴──────┐
        ▼              ▼
   [缓存命中?]     [缓存未命中]
        │              │
        │      ┌───────┘
        │      ▼
        │   [Prefill GPU] → KV Cache
        │      │
        └──────┼────────── (RDMA 传输)
               ▼
          [Decode GPU] → 流式响应
               │
               ▼
          [KV Cache Store] (多级缓存: GPU→DRAM→NVMe)
```

## 4.4 STAR 总结

### Situation（背景 + 痛点）

LLM 推理中，预填充（Prefill）和解码（Decode）对计算资源的需求截然相反：预填充需要高 TFLOPS 计算能力并行处理所有 prompt token，而解码需要高内存带宽逐 token 生成输出。2023-2024 年主流方案将二者混合在同一 GPU 上执行，导致严重干扰——一个长提示的预填充会阻塞所有并发解码迭代，产生头部阻塞（Head-of-Line Blocking）。随着 70B+ 大模型和多轮对话、智能体等场景普及，这一矛盾急剧恶化，TTFT 抖动可达 10 倍以上。

### Task（核心问题）

如何在避免预填充-解码相互干扰的前提下，同时最大化系统吞吐和满足严格延迟 SLO（TTFT < 500ms, TPOT < 50ms）？核心约束包括：(1) 两阶段间必须传输 KV Cache，其大小与上下文长度线性增长（70B 模型 4K 上下文约 1.34 GB）；(2) 网络带宽有限，传输延迟不可忽略；(3) GPU 集群是昂贵的共享资源，扩缩容效率直接影响 TCO。

### Action（主流方案）

技术进步可分三个关键阶段：(1) **软件优化（2024）**：Sarathi-Serve 提出分块预填充 + 无气泡调度，在不分离硬件的前提下缓解头部阻塞；(2) **物理分离（2024-2025）**：DistServe、Splitwise、Mooncake 等将两阶段分配到独立 GPU 池，证明 goodput 可提升 4-7×，但受限于 KV Cache 跨节点传输延迟；(3) **精细化分离（2025-2026）**：NVIDIA Dynamo 1.0 实现生产级 PD/EPD 分离，SGLang 和 vLLM 原生支持；DeepSeek DualPath 突破 I/O 瓶颈，利用解码引擎闲置网卡实现双路径加载；TaiChi 和 PPD 实现请求级自适应的聚合-分离切换；DUET 甚至为两阶段设计专用芯片。

### Result（效果 + 建议）

当前 PD 分离已进入生产成熟期。Dynamo 1.0 在 Blackwell 上实现 7× 吞吐提升，DeepSeek DualPath 在 660B 模型上实现 1.96× 在线吞吐提升。对技术选型的实操建议是：(1) **< 7B 模型或短文本场景**无需分离——全聚合即可；(2) **70B+ 模型或长上下文**强烈推荐物理 PD 分离，优先使用 NVIDIA Dynamo 或 SGLang 的现成实现；(3) **多模态应用**采用 EPD 三阶段分离提升编码效率；(4) **智能体场景**（缓存命中率 > 90%）需额外关注 I/O 瓶颈和缓存架构，可参考 DeepSeek DualPath。未来方向集中在 GPU 内动态分离、多模型共享预填充（PrefillShare）、以及硬件-算法协同设计。

## 4.5 理解确认问题

> **问题**：假设你的推理集群从 100 个请求/秒增长到 1000 个请求/秒，输入平均长度从 1K tokens 增长到 8K tokens。在"为 Prefill 和 Decode 分别增加等比例 GPU"和"动态调整 Prefill/Decode 配比"两个策略中，哪个策略更优？为什么？

**参考答案**：动态调整配比策略更优。因为输入长度增长会线性增加 Prefill 的计算量（KV Cache 也随之增大），而 Decode 负载主要受生成 token 数影响。此时 Prefill 压力会不成比例地增长，简单等比例增加 GPU 会导致 Decode 资源过剩。正确的做法是使用 TokenScale 或 Dynamo Planner 等工具，基于当前请求特征（输入长度分布、缓存命中率、SLO 要求）动态计算最优 P/D 配比——通常在输入变长时需要将更多 GPU 分配给 Prefill 池。同时建议引入冷热请求分离（CPD），让缓存命中的请求直接发往 Decode，缓解 Prefill 压力。

---

# 附录

## A. 参考资源汇总

| 分类 | 数量 | 说明 |
|------|------|------|
| GitHub 开源项目 | 15+ | 涵盖生产引擎、研究原型、传输库 |
| 学术论文 | 12 篇 | 顶级会议（OSDI/SOSP/ISCA/FAST/EuroSys/ICML/DAC）+ arXiv |
| 技术博客 | 10 篇 | 官方博客（NVIDIA/Together AI/AMD）+ 深度教程 + 中文资源 |
| 时间线里程碑 | 20+ 事件 | 2023.11 至 2026.05 |

## B. 缩写对照表

| 缩写 | 全称 | 说明 |
|------|------|------|
| PD | Prefill-Decode | 预填充-解码两阶段分离 |
| EPD | Encode-Prefill-Decode | 编码-预填充-解码三阶段分离（多模态） |
| TTFT | Time to First Token | 首 token 延迟，衡量预填充速度 |
| TPOT | Time Per Output Token | 每输出 token 延迟，衡量解码速度 |
| ITL | Inter-Token Latency | token 间延迟，流式体验关键指标 |
| SLO | Service Level Objective | 服务质量目标 |
| Goodput | Good + Throughput | 在 SLO 内完成的请求吞吐量 |
| KV Cache | Key-Value Cache | 注意力机制的键值缓存 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问，高速网络传输 |
| TCO | Total Cost of Ownership | 总拥有成本 |

---

*本报告基于 2026 年 5 月公开数据编写，所有 GitHub Stars 数据为近似值，以实际 GitHub 页面为准。*
