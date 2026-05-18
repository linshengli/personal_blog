# 大模型推理服务多模型弹性部署与资源共享技术 — 深度调研报告

> **调研日期**：2026-05-18
> **所属领域**：大模型框架
> **调研范围**：涵盖 2024–2026 年最新论文、开源项目与工业实践

---

# 第一部分：概念剖析

## 1. 定义澄清

### 通行定义

大模型推理服务多模型弹性部署与资源共享技术，是指在一套推理基础设施上同时承载多个 LLM（大语言模型）的在线推理请求，通过动态分配 GPU/CPU 显存与算力资源，实现按需扩缩容、跨模型资源复用、以及异构硬件兼容的部署能力。该技术的核心目标是：**在满足多个模型各自的服务质量目标（SLO，含 TTFT 首Token延迟和 TPOT 输出Token速率）前提下，最大化集群资源利用率、最小化总运营成本（TCO）。**

### 常见误解

1. **误解一："多模型部署就是把多个模型分别部署在各自 Pod 里，用 K8s Service 路由就够了"** —— 实际上，简单的隔离部署忽略了显存资源共享的可能性。现代系统通过弹性显存管理（如 PRISM 的 kvcached）可以在同一 GPU 上按需切换模型，大幅提高利用率。
2. **误解二："弹性伸缩就是 HPA 根据 CPU/内存扩容"** —— LLM 推理的瓶颈在 GPU 显存（尤其是 KV Cache）而非 CPU。真正的弹性需要感知 KV Cache 状态、Token 级调度和 SLA 驱动的扩缩容。
3. **误解三："Prefill 和 Decode 必须绑定在同一 GPU 上完成"** —— PD 分离架构已证明将两者解耦为独立服务池可消除相互干扰，显著提升整体吞吐。

### 边界辨析

| 易混淆概念 | 核心区别 |
|-----------|---------|
| **单体模型部署**（单模型独占 GPU） | 多模型弹性部署强调多个模型在共享基础设施上的共存与资源复用 |
| **模型量化/蒸馏**（压缩模型本身） | 弹性部署是基础设施层的编排技术，而非模型本身的压缩技术 |
| **推理引擎性能优化**（如 vLLM PagedAttention） | 引擎优化解决单模型推理效率；弹性部署则解决多模型间的资源调度与协调 |
| **云原生微服务**（通用 K8s 部署） | LLM 推理弹性部署需要 GPU 显存感知、KV Cache 路由等 AI 特有调度能力 |

---

## 2. 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│              大模型推理多模型弹性部署系统架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [用户请求] ──→ [统一API网关] ──→ [全局调度器] ──→ [推理集群] ──→ [返回]│
│                    │                     │           │              │
│                    ▼                     ▼           ▼              │
│              [模型路由]            [弹性伸缩器]    [多Engine池]       │
│              (路由至目标模型)      (SLA驱动扩缩)  (vLLM/SGLang/TRT-LLM)│
│                                                                  │
│  ┌──────────────────── 推理集群内部 ────────────────────────────┐  │
│  │                                                              │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │  │
│  │  │ Prefill 池    │  │ Decode 池    │  │ Encode 池    │       │  │
│  │  │ (计算密集型)   │←→│ (显存密集型)   │←→│ (多模态编码)  │       │  │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │  │
│  │         │                  │                  │               │  │
│  │         └──────────────────┴──────────────────┘               │  │
│  │                        │                                      │  │
│  │              [KV Cache 传输层]                                 │  │
│  │           (RDMA/以太网/共享显存)                                │  │
│  │                        │                                      │  │
│  │              ┌─────────────────────┐                          │  │
│  │              │ 全局 KV Cache 管理器  │                          │  │
│  │              │ (Mooncake/分布式缓存) │                          │  │
│  │              └──────────┬──────────┘                          │  │
│  │                         │                                     │  │
│  │              ┌─────────────────────┐                          │  │
│  │              │ 弹性显存管理器       │                          │  │
│  │              │ (kvcached/虚拟-物理  │                          │  │
│  │              │  显存解耦)          │                          │  │
│  │              └─────────────────────┘                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  [监控与可观测性] ←── [SLO 管理器] ←── [成本优化器]                  │
│  (Prometheus/DCGM)  (TTFT/TPOT/SLA)  (资源分配策略)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**各组件说明**：

| 组件 | 一句话职责 |
|------|-----------|
| **统一API网关** | 接收请求并根据模型名称路由到对应调度器，支持 OpenAI 兼容接口 |
| **全局调度器** | 基于 KV Cache 状态、模型热度和 SLO 要求做智能路由决策 |
| **弹性伸缩器** | 根据实时流量和排队深度动态调整 Prefill/Decode 实例数 |
| **Prefill 池** | 处理计算密集的预填充阶段，生成 KV Cache |
| **Decode 池** | 处理显存带宽敏感的自回归解码阶段 |
| **KV Cache 管理器** | 跨节点管理和传输 KV Cache，避免重复计算 |
| **弹性显存管理器** | 在多个模型间动态分配 GPU 显存，实现空间共享 |
| **SLO 管理器** | 监控每个请求的延迟指标，驱动扩缩容和调度决策 |
| **成本优化器** | 综合考虑 GPU 类型、模型大小和流量模式，最小化运行成本 |

---

## 3. 数学形式化

### 3.1 KV Cache 内存需求模型

对于序列长度为 $L$、批次大小为 $B$、注意力头数为 $H$、每头维度为 $d_k$ 的推理请求，KV Cache 占用的 GPU 显存为：

$$
M_{KV}(B, L, H, d_k) = 2 \cdot B \cdot L \cdot H \cdot d_k \cdot \text{dtype\_size}
$$

*解释：每个 Token 需要存储 Key 和 Value 两组向量，乘以层数（权重模型参数中已包含层数因子），显存占用随序列长度线性增长，这是多模型共享部署中最重要的资源约束。*

### 3.2 PD 分离的性能增益模型

设 Prefill 阶段计算量为 $C_p$，Decode 阶段每 Token 计算量为 $C_d$，分离后独立扩缩容：

$$
\text{Speedup} = \frac{C_p + N \cdot C_d}{\max\left(\frac{C_p}{n_p}, \frac{N \cdot C_d}{n_d}\right)} \quad \text{s.t.} \quad n_p + n_d \leq N_{total}
$$

*解释：通过为 Prefill（计算密集型）和 Decode（显存密集型）设置独立的 GPU 数量 $n_p$ 和 $n_d$，消除了两者因资源竞争产生的相互阻塞，当 $n_p/n_d$ 与实际计算量比值匹配时可获得最大收益。*

### 3.3 多模型资源共享的成本效率

对于 $M$ 个模型部署在 $G$ 块 GPU 上，资源共享的收益定义为：

$$
\eta = \frac{\sum_{i=1}^{M} C_i^{\text{isolated}}}{\sum_{j=1}^{G} C_j^{\text{shared}}}
$$

其中 $C_i^{\text{isolated}}$ 是模型 $i$ 独占部署的成本，$C_j^{\text{shared}}$ 是共享部署中 GPU $j$ 的实际开销。

*解释：资源共享带来的成本节约来源于两个层面——空间共享（多个模型同时驻留同一 GPU，利用显存的时间片复用）和时域弹性（模型按需加载/卸载，避免 GPU 空闲等待）。*

### 3.4 弹性伸缩的响应延迟模型

从触发扩容到新实例就绪的总延迟：

$$
T_{scale} = T_{detect} + T_{schedule} + T_{load} + T_{warm}
$$

其中 $T_{detect}$ 是检测延迟（通常 1-15s），$T_{schedule}$ 是调度延迟（GPU 资源分配，秒级），$T_{load}$ 是模型权重加载延迟（GB级模型可达 10-60s），$T_{warm}$ 是预热延迟（首次推理慢，可预执行虚拟请求规避）。

*解释：弹性伸缩的瓶颈在于模型加载和预热，PRISM 等系统通过预初始化引擎和并行权重加载，将 $T_{load}$ 压缩至 1.5s 以内。*

### 3.5 多模型部署的 SLO 合规率

对于一个有 $K$ 个模型的系统，总 SLO 合规率为：

$$
\text{SLO\_Attainment} = \frac{1}{K} \sum_{i=1}^{K} P(TTFT_i \leq \tau_i \land TPOT_i \geq \rho_i)
$$

*解释：多模型部署的难度不仅在于单个模型的延迟优化，更在于多个模型之间的资源隔离保证——一个模型的高流量不应对其他模型的延迟分布产生显著影响（即"嘈杂邻居问题"）。*

---

## 4. 实现逻辑（Python 风格伪代码）

```python
class MultiModelElasticServingSystem:
    """多模型弹性部署核心系统"""

    def __init__(self, config: SystemConfig):
        self.gpu_cluster = GPUCluster(config.gpu_list)      # 异构 GPU 集群管理
        self.model_registry = ModelRegistry(config.models)   # 已注册模型清单
        self.memory_manager = ElasticMemoryManager()         # 弹性显存管理器
        self.scheduler = TwoLevelScheduler()                 # 两层调度器
        self.autoscaler = SLADrivenAutoscaler(config.slas)   # SLA 驱动弹性伸缩器
        self.kv_cache_manager = GlobalKVCacheManager()      # 全局 KV Cache 管理

    def serve_request(self, request: InferenceRequest) -> Response:
        """处理推理请求的核心流程"""

        # Step 1: 路由决策 — 选择合适的模型实例
        model_name = request.model_name
        target_gpu = self.scheduler.route(
            model=model_name,
            prompt=request.prompt,
            kv_cache_hint=request.get_prefix_hash()
        )

        # Step 2: 弹性显存分配
        memory_slot = self.memory_manager.allocate(
            model_name=model_name,
            required_memory=estimate_kv_cache_memory(request)
        )

        # Step 3: 模型激活（如需）
        if not memory_slot.is_loaded:
            self._activate_model(model_name, memory_slot.gpu_id)

        # Step 4: 执行推理（可选择 PD 分离模式）
        if self.config.disaggregated_mode:
            return self._disaggregated_infer(request, target_gpu)
        else:
            return self._local_infer(request, target_gpu)

    def _activate_model(self, model_name: str, gpu_id: int):
        """快速模型激活 — 预初始化 + 并行加载"""
        model_handle = self.model_registry.get(model_name)
        # PRISM 式快速激活：权重预加载至 Host，Engine 预热保持
        model_handle.engine = self.model_registry.get_cached_engine(model_name)
        model_handle.load_weights_parallel(gpu_id)

    def _disaggregated_infer(self, request, target_gpu):
        """PD 分离推理流程"""
        # Prefill 阶段 — 在计算型 GPU 上执行
        kv_cache = self.kv_cache_manager.execute_prefill(
            request=request,
            prefill_pool=self.gpu_cluster.prefill_pool
        )
        # KV Cache 传输 (RDMA / 共享显存)
        self.kv_cache_manager.transfer(kv_cache, to_gpu=target_gpu)
        # Decode 阶段 — 在显存型 GPU 上执行
        return self.kv_cache_manager.execute_decode(
            kv_cache=kv_cache,
            decode_pool=self.gpu_cluster.decode_pool
        )

    def auto_scale_loop(self):
        """弹性伸缩主循环"""
        while True:
            metrics = self._collect_metrics()         # 收集 QPS / 排队长度 / GPU 利用率
            scaling_decisions = self.autoscaler.decide(
                current_replicas=self._get_replica_count(),
                metrics=metrics,
                model_slas=self.config.slas
            )
            for decision in scaling_decisions:
                if decision.action == "scale_up":
                    self._provision_gpu(decision.replicas)
                elif decision.action == "scale_down":
                    self._decommission_gpu(decision.replicas)
            time.sleep(self.config.scale_interval)
```

---

## 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **TTFT** (Time to First Token) | < 200ms (p99) | 从请求发起到收到第一个Token的端到端延迟 | 直接影响用户体验，多模型部署中需保证隔离性 |
| **TPOT** (Time Per Output Token) | < 30ms/token | 后续每个Token的平均生成时间 | 决定输出流畅度 |
| **吞吐量** (Throughput) | > 1000 tok/s/GPU | 单位时间生成的Token总数 | 通常为最大吞吐(SLA可接受范围内) |
| **GPU 利用率** | > 70% | DCGM 监控 GPU 计算/显存利用率 | 共享部署的核心优化目标 |
| **SLO 合规率** | > 99% | 满足延迟约束的请求占比 | 多模型服务的关键质量指标 |
| **冷启动时间** | < 5s | 模型加载到首次推理的耗时 | PRISM等系统可压至 1.5s |
| **扩缩容响应时间** | < 30s | 从触发条件到完成扩缩容 | 需结合预测性扩缩容 |
| **资源碎片率** | < 15% | 无法分配的显存碎片比例 | 弹性显存管理的效率指标 |
| **成本效率** (Cost/token) | < $0.1/M tokens | 总运营成本/总生成Token数 | 共享部署的核心价值度量 |

---

## 6. 扩展性与安全性

### 水平扩展

- **多节点 Prefill/Decode 池**：通过 RDMA 或高速以太网实现 KV Cache 跨节点传输，支持千卡级集群扩展
- **模型分片并行**：Tensor Parallelism (TP) 切分单层权重跨 GPU，Pipeline Parallelism (PP) 分层部署，Expert Parallelism (EP) 用于 MoE 模型
- **跨集群联邦**：如 Cognitora 通过 QUIC 协议实现跨集群 KV Cache 共享，打破单集群物理边界

### 垂直扩展

- **单 GPU 计算密度**：FP4/FP8 量化、稀疏计算、FlashAttention 等算子级优化
- **显存扩展**：H200 的 141GB HBM3e、B200 的 192GB HBM3e，单卡可承载更大模型
- **NVLink 域扩展**：NVL72 等机柜级 GPU 互联，大幅降低跨 GPU 通信延迟

### 安全考量

- **模型隔离**：多模型共享 GPU 显存时，需防止恶意模型通过显存侧信道窃取其他模型的 KV Cache 数据
- **请求级隔离**：保证高安全等级请求不会被与其他请求共享 KV Cache 导致信息泄露
- **多租户计费与配额**：多团队共享集群时，需要精确的 Token 级计费和 GPU 时间片计量
- **模型权重保护**：模型从存储加载到 GPU 过程中的加密传输和内存保护

---

# 第二部分：行业情报

## 1. GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | ~74k | PagedAttention 高吞吐推理引擎，支持 PD 分离、prefix caching | Python/C++/CUDA | 持续活跃 | [GitHub](https://github.com/vllm-project/vllm) |
| **SGLang** | ~20k | RadixAttention 前缀树 KV Cache，DeepSeek 原生支持，最佳结构化输出 | Python/C++/CUDA | 持续活跃 | [GitHub](https://github.com/sgl-project/sglang) |
| **TensorRT-LLM** | ~10k | NVIDIA 编译式推理引擎，FP4/FP8 支持，最高吞吐 | C++/CUDA | 持续活跃 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **NVIDIA Dynamo** | ~6.8k | 数据中心级推理编排层，KV 感知路由、PD 分离、SLA 自动伸缩 | Rust/Python | 2026-05 | [GitHub](https://github.com/ai-dynamo/dynamo) |
| **NanoPD** | ~500 | 从零实现的 PD 分离推理引擎，自适应路由策略 | Python/CUDA | 2026 | [GitHub](https://github.com/HJCheng0602/nanoPD) |
| **llm-d** | — | K8s 原生 LLM 部署框架，GIE 集成，智能调度 | Go/Python | 2026-03 | [llm-d.ai](https://llm-d.ai) |
| **KServe** | ~3.5k | K8s 推理服务 CRD，v0.17 引入 LLMInferenceService | Go/Python | 2026-03 | [GitHub](https://github.com/kserve/kserve) |
| **PRISM** | — | 多 LLM 灵活 GPU 共享（时间和空间），弹性显存管理 | Python/C++ | 2026-03 | [GitHub](https://github.com/Multi-LLM/prism-research) |
| **Ollama** | ~120k | 本地 LLM 运行工具，v0.6 支持多模型 pipeline | Go | 持续活跃 | [GitHub](https://github.com/ollama/ollama) |
| **llama.cpp** | ~75k | CPU/边缘设备 LLM 推理，量化支持广泛 | C/C++ | 持续活跃 | [GitHub](https://github.com/ggml-ai/llama.cpp) |
| **Ray Serve** | ~35k | 分布式 Serving 框架，支持 LLM 的 Wide-EP 和 PD 分离 | Python | 持续活跃 | [GitHub](https://github.com/ray-project/ray) |
| **AI Runway** | — | K8s 原生多模型部署平台，Web UI 零 YAML 操作 | Go | 2026-04 | [GitHub](https://github.com/kaito-project/airunway) |
| **Cognitora** | — | 单二进制推理编排，支持多引擎、跨集群 KV Cache 联邦 | Rust | 2026-05 | [GitHub](https://github.com/antonellof/cognitora-inference) |
| **Mooncake** | — | 分布式 KV Cache 传输系统，RDMA 加速，月之暗面出品 | C++ | 2025 | [GitHub](https://github.com/kvcache-ai/Mooncake) |

---

## 2. 关键论文

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|-------|------|
| **vLLM: PagedAttention** | Kwon et al. (UC Berkeley) | 2023 | SOSP | 提出 PagedAttention 和 vLLM，将虚拟内存思想引入 KV Cache 管理 | 奠基性工作，>50k GitHub Stars | [arXiv](https://arxiv.org/abs/2309.06180) |
| **SGLang: Efficient Execution** | Zheng et al. (LMSYS) | 2024 | OSDI | RadixAttention 前缀树 KV Cache + 结构化输出约束引擎 | SOTA 推理引擎之一 | [arXiv](https://arxiv.org/abs/2312.07104) |
| **PRISM: GPU Sharing for Multi-LLM** | Yu et al. (UCLA/Berkeley) | 2025 | — | 弹性显存管理器 kvcached，两层调度器，时间+空间 GPU 共享 | 多模型共享奠基性工作 | [arXiv](https://arxiv.org/abs/2505.04021) |
| **LLM-Mesh: Serverless Elastic Sharing** | Xing et al. | 2025 | — | 异构 CPU+GPU 弹性共享，Token 级资源分配 | 44-63% 服务容量提升 | [arXiv](https://arxiv.org/abs/2507.00507) |
| **ElasticMM (NeurIPS 2025 Oral)** | 多机构 | 2025 | NeurIPS | 多模态 LLM 弹性并行 (EMP)，TTFT 降低 4.2× | NeurIPS Oral，高认可度 | [arXiv](https://arxiv.org/abs/2507.10069) |
| **xLLM** | 华为等 | 2025 | — | 解耦服务-引擎架构，动态 PD/EPD 分离，统一弹性调度 | 吞吐量提升 1.7-2.2× | [arXiv](https://arxiv.org/abs/2510.14686) |
| **Coral: Multi-LLM Cost Efficiency** | Mei et al. (CMU) | 2026 | — | 异构云 GPU 多模型联合优化，两阶段分解算法 | 成本降低 2.79× | [arXiv](https://arxiv.org/abs/2605.04357) |
| **OServe (ICML 2026)** | Jiang et al. (IBM) | 2026 | ICML | 时空工作负载感知编排，异构模型部署自适应切换 | 性能提升 2× | [arXiv](https://arxiv.org/abs/2602.12151) |
| **Nitsum** | 多机构 | 2026 | — | 动态 Tensor Parallel 运行时控制，层次化 SLO | SLO 合规吞吐提升 5.3× | [arXiv](https://arxiv.org/abs/2605.05467) |
| **Star Elastic** | 多机构 | 2026 | — | Many-in-One 嵌套子模型，弹性预算控制的推理 | 精度提升 16%，延迟降低 1.9× | [arXiv](https://arxiv.org/abs/2605.07182) |
| **PPD Disaggregation** | 多机构 | 2026 | — | 针对多轮对话的 Prefill-Prefill-Decode 动态分离策略 | 减少 KV 传输瓶颈 | [arXiv](https://arxiv.org/abs/2603.13358) |
| **MegaScale-Omni (EuroSys 2026)** | 多机构 | 2026 | EuroSys | 超大规模多模态 LLM 训练推理，编码器-LLM 多路复用 | 吞吐量提升 7.57× | [ACM](https://dl.acm.org/doi/10.1145/3767295.3803587) |

---

## 3. 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| The Great LLM Inference Engine Showdown | UltraDune | EN | 评测对比 | vLLM vs TGI vs TRT-LLM vs SGLang vs llama.cpp vs Ollama 六引擎深度对比 | 2026-03 | [链接](https://buttondown.com/ultradune/archive/eval-001-the-great-llm-inference-engine-showdown) |
| 大模型推理架构革新：PD分离架构的实践与评估 | 百度开发者 | CN | 技术博客 | PD 分离原理、部署实践、性能收益数据 | 2025 | [链接](https://developer.baidu.com/article/detail.html?id=6858901) |
| 云原生架构下大模型推理的分布式编排与调度实践 | 百度开发者 | CN | 架构解析 | K8s + HPA/VPA + NUMA 绑定的生产级部署 | 2025 | [链接](https://developer.baidu.com/article/detail.html?id=6891848) |
| EaaS：面向在线推理场景的弹性架构创新 | 百度开发者 | CN | 架构解析 | MoE 模型超细粒度弹性，单 GPU 调度单元，RL 流量感知 | 2025 | [链接](https://developer.baidu.com/article/detail.html?id=6916438) |
| 从IDC到云上GPU：EKS 大模型推理混合云弹性部署 | AWS 中国 | CN | 实践指南 | KEDA+Karpenter 混合云弹性架构，成本降至 11% | 2025 | [链接](https://aws.amazon.com/cn/blogs/china/idc-gpu-based-on-amazon-eks-large-model-inference-hybrid-cloud-elastic-deploy-practice/) |
| Production-Grade LLM Inference with KServe + llm-d + vLLM | llm-d | EN | 实践指南 | 生产级 LLM 推理部署，KV Cache 感知调度 | 2026 | [链接](https://llm-d.ai/blog/production-grade-llm-inference-at-scale-kserve-llm-d-vllm) |
| LLM 推理框架选型指南：SGLang vs vLLM 深度对比 | 百度开发者 | CN | 技术对比 | 两个框架的架构差异、性能对比和选型建议 | 2025 | [链接](https://developer.baidu.com/article/detail.html?id=6860758) |
| Maximize AI Infrastructure Throughput by GPU Consolidation | NVIDIA | EN | 最佳实践 | MIG vs Time-Slicing GPU 共享方案评估 | 2026 | [链接](https://developer.nvidia.cn/blog/maximize-ai-infrastructure-throughput-by-consolidating-underutilized-gpu-workloads/) |
| Ray Serve LLM: Wide-EP and Disaggregated Serving | Anyscale | EN | 技术博客 | Ray 集群上的 LLM Serving、专家并行和 PD 分离 | 2025 | [链接](https://www.anyscale.com/blog/ray-serve-llm-anyscale-apis-wide-ep-disaggregated-serving-vllm) |
| ClearML + NVIDIA Dynamo: Production Control Plane | ClearML | EN | 集成方案 | 企业级多租户推理治理、安全层、成本分配 | 2026 | [链接](https://clear.ml/blog/clearml-nvidia-dynamo-a-production-control-plane-for-distributed-ai-inference-at-scale) |

---

## 4. 技术演进时间线

```
2023 ── vLLM 提出 PagedAttention（SOSP 2023）── 将虚拟内存思想引入 KV Cache 管理，奠定现代推理引擎基础
2024 ── SGLang 发布 RadixAttention（OSDI 2024）── 前缀树共享 KV Cache，DeepSeek 等选择 SGLang 作为官方引擎
2024 ── TensorRT-LLM 开源 ── NVIDIA 编译式推理引擎，FP8/INT4 量化支持，最高吞吐但操作复杂
2024 ── PD 分离架构兴起 ── vLLM、SGLang、TRT-LLM 纷纷支持 Prefill/Decode 独立部署
2025 ── NVIDIA Dynamo v1.0 发布 ── 首个数据中心级推理编排层，KV 感知路由、SLA 驱动伸缩
2025 ── PRISM 提出多模型 GPU 共享 ── 弹性显存管理 kvcached，时间和空间共享，成本节省 >2×
2025 ── KServe v0.17 LLMInferenceService ── K8s 原生 LLM Serving CRD，GIE 集成 KV Cache 路由
2025 ── ElasticMM (NeurIPS 2025 Oral) ── 多模态 LLM 弹性并行，EPD 三阶段分离
2025 ── Mooncake 分布式 KV Cache 系统开源 ── RDMA 加速跨节点 KV 传输，月之暗面出品
2026 ── Coral 提出异构 GPU 多模型联合优化 ── 6 模型 20 种 GPU 配置，成本降低 2.79×
2026 ── OServe (ICML 2026) ── 时空工作负载感知编排，异构模型部署自适应切换
2026 ── Nitsum 动态 Tensor Parallel ── 将 TP 从静态配置变为运行时控制手段
2026 ── Star Elastic Many-in-One ── 单次训练获得嵌套子模型，推理阶段弹性预算控制
2026 ── 当前状态：PD 分离成为标配、KV Cache 感知路由进入生产、多模型共享从研究走向工程、
        弹性伸缩从 Pod 级细化至 GPU 级、行业加速从"单模型优化"转向"多模型系统级效率优化"
```

---

# 第三部分：方案对比

## 1. 历史发展时间线

```
2022 ── 早期阶段：单模型独占 GPU，部署方式简单粗暴，GPU 利用率 < 30%
2023 ── vLLM PagedAttention 问世 → 单 GPU 可承载更大并发，但仍为单模型
2024 ── PD 分离架构成熟 → Prefill/Decode 解耦，可独立扩缩容，吞吐提升 2-3×
2024 ── 多模型共享萌芽 → 简单的时间片轮换 GPU 共享（如 TGI 的多 LoRA）
2025 ── 弹性显存管理突破 → PRISM 的 kvcached 实现虚拟-物理显存解耦，模型加载 < 1.5s
2025 ── 推理编排层诞生 → NVIDIA Dynamo 统一编排多种引擎，KV 感知路由
2026 ── 系统级多模型优化 → Coral/OServe 联合优化资源分配 + 推理策略
2026 ── 当前状态：从"如何部署一个模型"转向"如何高效部署一群模型"
```

---

## 2. 七种方案横向对比

### 方案 A：独立 Pod 部署（传统 K8s 无共享）
### 方案 B：推理引擎内置多模型（vLLM/SGLang 多 LoRA / 多模型）
### 方案 C：PD 分离 + 独立扩缩容
### 方案 D：弹性显存共享（PRISM 方案）
### 方案 E：推理编排层（NVIDIA Dynamo / KServe llm-d）
### 方案 F：异构 GPU 多模型联合优化（Coral 方案）
### 方案 G：动态 Tensor Parallel 运行时控制（Nitsum 方案）

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **A: 独立 Pod 部署** | 每模型独占一套 GPU Pod，通过 K8s Service 路由 | ① 部署简单，运维成本低 ② 模型隔离性好 ③ 故障域小 | ① GPU 利用率极低（<30%） ② 资源碎片严重 ③ 扩缩容粒度粗 ④ 无法跨模型复用 | 原型验证、低流量场景 | $/tokens: 最高 |
| **B: 引擎内置多模型** | 同一 vLLM/SGLang 进程加载多模型/LoRA，共享 KV Cache 池 | ① 零额外组件 ② 进程内共享延迟低 ③ 配置简单 | ① 多模型隔离性差 ② 单进程故障影响全部模型 ③ 显存竞争难控制 ④ 不适合>3个模型 | 少量模型（2-3个）共享、LoRA 微调场景 | $$
| **C: PD 分离+独立扩缩容** | Prefill 和 Decode 各自独立部署 GPU 池，可独立扩缩 | ① 消除 P/D 相互干扰 ② 吞吐提升 2-3× ③ 可根据流量特点灵活配比 | ① 需要 KV Cache 跨节点传输 ② 架构复杂度大幅增加 ③ 小流量下收益不明显 ④ RDMA 网络要求高 | 高负载生产、长序列推理、多轮对话 | $$$
| **D: 弹性显存共享** (PRISM) | 虚拟-物理显存解耦，多模型在 GPU 上时间/空间共享 | ① GPU 利用率 > 70% ② 成本节省 > 2× ③ 模型切换 < 1.5s ④ 58模型同集群 | ① 显存隔离不如物理独占 ② 嘈杂邻居问题 ③ 需要精确的显存压力预测 ④ 生产验证尚不够 | 中等规模多模型服务、GPU 资源紧张团队 | $$
| **E: 推理编排层** (Dynamo/KServe) | 在引擎之上增加编排层，统一调度、路由和伸缩 | ① 引擎无关，支持 vLLM/SGLang/TRT-LLM ② KV 感知路由 ③ SLA 驱动弹性伸缩 ④ K8s 原生集成 | ① 引入额外组件和复杂度 ② 编排层本身有运维成本 ③ 成熟度仍在提升中 ④ 最佳实践尚未标准化 | 企业级生产环境、多团队共享集群 | $$$
| **F: 异构 GPU 联合优化** (Coral) | 跨 A100/H100/L40S 等异构 GPU 联合分配模型与资源 | ① 充分利用不同 GPU 性价比 ② 成本降低 2.79× ③ 20 种 GPU 配置灵活组合 ④ 在线求解仅需秒级 | ① 求解器依赖离线 profiling ② 异构 GPU 通信瓶颈 ③ 需要全面的 GPU 性能数据库 ④ 部署管理复杂 | 云上多 GPU 类型、成本敏感的大型服务 | $$$$
| **G: 动态 TP 控制** (Nitsum) | 运行时动态调整 Tensor Parallel 级别，按请求 SLO 分层服务 | ① 层次化 SLO 灵活适配 ② SLO 合规吞吐提升 5.3× ③ 细粒度资源控制 ④ TP 变化对上层透明 | ① 仅支持同节点 GPU ② 需要 NVLink 支持 ③ 实现复杂度高 ④ 跨 TP 级别切换有开销 | 高 QoS 要求、混合优先级场景、同节点多 GPU | $$$$

---

## 3. 技术细节对比

| 维度 | A:独立Pod | B:引擎多模型 | C:PD分离 | D:弹性显存 | E:编排层 | F:异构优化 | G:动态TP |
|------|:---------:|:-----------:|:---------:|:----------:|:--------:|:----------:|:--------:|
| **吞吐量** | ★☆☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★★★ | ★★★★★ |
| **GPU 利用率** | ★☆☆☆☆(<30%) | ★★★☆☆(40-50%) | ★★★★☆(50-65%) | ★★★★★(>70%) | ★★★★☆(55-70%) | ★★★★★(>75%) | ★★★★☆(55-70%) |
| **易用性** | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ |
| **生态成熟度** | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ |
| **社区活跃度** | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ |
| **学习曲线** | 低 | 低 | 中 | 高 | 中高 | 很高 | 很高 |
| **模型隔离性** | ★★★★★ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| **弹性伸缩能力** | ★☆☆☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★★☆ | ★★★★★ | ★★★★☆ | ★★★☆☆ |
| **多模型扩展性** | ★☆☆☆☆(1:1) | ★★☆☆☆(2-3) | ★★☆☆☆ | ★★★★★(58) | ★★★★☆ | ★★★★☆(6+) | ★★☆☆☆ |

---

## 4. 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证**（1-2 模型，< 100 QPS） | **A: 独立 Pod 部署 + vLLM** | 零学习成本，快速上线，不需要复杂的弹性架构 | $2K-5K（单 GPU 按需） |
| **启动公司 MVP**（2-3 模型，100-500 QPS） | **B: 引擎内置多模型**（SGLang 多模型模式） | 进程内共享延迟低，运维简单，成本可控 | $5K-15K |
| **中型生产服务**（3-5 模型，500-2000 QPS） | **C: PD 分离 + D: 弹性显存混合方案** | PD 分离保证延迟稳定，弹性显存提升利用率，综合性价比最优 | $15K-50K |
| **企业级多租户平台**（5-10+ 模型，> 2000 QPS） | **E: 编排层（Dynamo/KServe）+ F: 异构优化** | 引擎无关统一编排，KV 感知路由，SLA 驱动扩缩，多集群联邦 | $50K-200K+ |
| **云上成本敏感型服务**（多模型混合流量） | **F: Coral 式异构优化 + G: 动态 TP** | 充分利用不同 GPU 型号的性价比，动态 TP 处理突发流量 | 视规模而定，比方案C节省 30-50% |
| **高频峰值抖动场景**（如电商 AI 助手） | **C: PD分离 + KEDA+Karpenter 混合云弹性** | 本地优先 + 云上兜底，通过 Spot 实例降低 67% 成本 | $20K-80K（弹性部分按使用量计费） |

---

### 架构组合模式

```
小规模 → 独立 Pod
    ↓
中规模 → SGLang 多模型 + 基础 PD 分离
    ↓
大规模 → NVIDIA Dynamo / KServe + vLLM/SGLang + PD 分离 + 弹性显存
    ↓
超大规模 → 编排层 + 异构 GPU 联合优化 + 动态 TP + 跨集群 KV 联邦
```

**2026 年推荐栈**：
- **推理引擎**：vLLM（通用稳定）或 SGLang（DeepSeek/结构化输出场景）
- **编排层**：NVIDIA Dynamo（NVIDIA 生态）或 KServe v0.17（K8s 原生、CNCF）
- **弹性伸缩**：KEDA + Karpenter（AWS）/ Cluster Autoscaler（通用）
- **KV Cache 管理**：Mooncake（RDMA）或内置 D2D 传输
- **多模型共享**：PRISM（研究前沿）或 Dynamo 多模型模式（工程成熟）

---

# 第四部分：精华整合

## 1. The One 公式

$$
\text{多模型弹性部署} = \underbrace{\text{PD 分离}}_{\text{消除阶段互扰}} + \underbrace{\text{弹性显存管理}}_{\text{虚拟-物理解耦}} - \underbrace{\text{资源碎片}}_{\text{显存与算力不匹配}}
$$

**翻译**：多模型部署的本质是：将每个请求的 Prefill 和 Decode 阶段解耦到独立 GPU 池（PD 分离），通过虚拟显存管理让多个模型灵活共享物理 GPU（弹性显存），同时尽可能减少因模型大小、请求长度各异造成的显存和算力碎片。

## 2. 一句话解释（费曼技巧）

> "就像 Airbnb 让多组旅客在不同时间共享同一套公寓的卧室和客厅，多模型弹性部署让不同的 AI 模型动态共享 GPU 集群——当模型 A 空闲时，模型 B 可以立即使用它的 GPU 显存，而不需要给每个模型单独买一套'房子'。"

## 3. 核心架构图

```
输入 ──→ [统一网关] ──→ [全局调度器] ──→ [GPU 集群：多模型共存]
          (路由+鉴权)   (KV感知+负载均衡)    ├─ Model A (Active)
                               ↓            ├─ Model B (Active)
                          [弹性伸缩器]        ├─ Model C (Inactive/已卸载)
                          (SLA驱动)          └─ Model D (Partial Loaded)
                               │
                               ↓
                        [成本优化器]
                     (异构GPU+混合云)
```

## 4. STAR 总结

### Situation（背景+痛点）

2025-2026 年，大模型应用进入爆发期，企业与平台需要同时部署多个 LLM（如 GPT-4、Claude、DeepSeek、Llama 以及行业微调模型）来满足不同场景需求。然而传统"一个模型独占一块 GPU"的部署模式导致 GPU 利用率普遍低于 30%，显存和算力碎片严重。同时，推理请求的流量模式波动剧烈（高峰可达低谷的 10-20 倍），静态部署要么资源浪费，要么服务质量不达标。这些挑战使得多模型的高效弹性部署成为大模型基础设施领域的核心前沿问题。

### Task（核心问题）

技术需要解决的核心挑战是：在异构 GPU 集群上同时服务多个 LLM，在满足各自延迟 SLO（TTFT < 200ms, TPOT < 30ms/token）的前提下，通过动态资源调度和弹性伸缩最大化集群利用率、最小化总拥有成本（TCO）。关键约束包括：GPU 显存是瓶颈资源、模型加载有显著延迟、请求流量具有空间和时间双重异质性。

### Action（主流方案）

技术演进经历了四个阶段：**第一阶段**（2023-2024），vLLM 等推理引擎通过 PagedAttention 优化单模型推理效率；**第二阶段**（2024-2025），PD 分离架构解耦 Prefill 和 Decode，实现按需独立扩缩容；**第三阶段**（2025），PRISM 等系统引入弹性显存管理（kvcached），实现多模型在同一 GPU 上的灵活时间和空间共享，模型切换压缩至 1.5s 内；**第四阶段**（2025-2026），NVIDIA Dynamo 和 KServe 等编排系统在引擎之上提供统一调度层，KV Cache 感知路由、SLA 驱动弹性伸缩和异构 GPU 联合优化（Coral）进一步将成本降低 2-3 倍。

### Result（效果+建议）

当前技术已实现：GPU 利用率从 < 30% 提升至 > 70%，多模型部署成本降低 2-3 倍，PD 分离带来 1.7-3.2× 的吞吐收益，模型冷启动从分钟级降至秒级。然而，弹性显存共享的"嘈杂邻居"问题、编排层的标准化、以及跨集群联邦 KV 共享仍是未完全解决的问题。

**实操建议**：
- **起步期**（< 3 模型）：采用 vLLM/SGLang 多模型模式 + 基础 HPA，快速上线
- **成长期**（3-10 模型）：引入 PD 分离 + Dynamo/KServe 编排层 + KEDA 弹性伸缩
- **成熟期**（> 10 模型）：叠加弹性显存共享 + 异构 GPU 联合优化 + KV 感知路由

## 5. 理解确认问题

> **问题**：为什么 PD（Prefill-Decode）分离能同时提升延迟和吞吐？如果给你 8 块 GPU，假设 Prefill 和 Decode 的计算需求比为 1:3，你会如何分配这 8 块 GPU 的 P/D 比例？

**参考答案**：
PD 分离之所以能同时提升延迟和吞吐，是因为 Prefill 是计算密集型（瓶颈在 GPU 算力），而 Decode 是显存带宽密集型（瓶颈在显存带宽和容量）。当两者绑定在同一 GPU 时：
1. **Prefill 阶段阻塞 Decode**：Prefill 处理长输入时 GPU 算力打满，导致正在进行的 Decode 步骤延迟飙升。
2. **显存竞争**：Prefill 需要大量临时显存作注意力计算，挤压了 Decode 的 KV Cache 空间。

分离后两者不再相互阻塞，可以各自独占优化的资源环境。根据 1:3 的计算需求比，最优分配应偏向 Decode：**2 块 GPU 做 Prefill，6 块做 Decode**（即 2p6d）。这个比例使 Prefill 可快速处理输入生成 KV Cache，然后 Decode 并行处理多个请求的逐一 Token 生成。但实践中还需根据序列长度和查询分布动态调整——短请求多则需更多 Prefill 容量，长对话多则需更多 Decode。

---

# 附录

## 术语表

| 术语 | 英文 | 说明 |
|------|------|------|
| TTFT | Time to First Token | 首Token生成时间，衡量用户感知的响应速度 |
| TPOT | Time Per Output Token | 后续每Token生成时间，衡量输出流畅度 |
| SLO | Service Level Objective | 服务质量目标，如 TTFT < 200ms |
| PD 分离 | Prefill/Decode Disaggregation | 将预填充和解码阶段部署到不同的 GPU 池 |
| KV Cache | Key-Value Cache | 注意力机制中缓存的中间结果，避免重复计算 |
| TP | Tensor Parallelism | 张量并行，将单层权重切分到多个 GPU |
| EP | Expert Parallelism | 专家并行，MoE 模型将不同专家分配到不同 GPU |
| GIE | Gateway Inference Extension | K8s Gateway API 的推理扩展，感知 KV Cache 路由 |
| MIG | Multi-Instance GPU | NVIDIA 的 GPU 硬件分区技术 |
| TCO | Total Cost of Ownership | 总拥有成本，包括 GPU 租赁、网络、运维等 |

## 参考来源

本报告基于 2026 年 5 月 18 日的 Web 搜索和文献调研，主要来源包括：
- arXiv、ACM Digital Library、NeurIPS/ICML/EuroSys 等学术来源
- GitHub 公开仓库的实时 Stars 和更新数据
- NVIDIA、AWS、Red Hat、Anyscale、ClearML 等厂商的技术博客
- 百度开发者社区、CSDN 等中文技术社区
- 具体链接已在各表格中标注

---

*报告生成日期：2026-05-18 | 字数：约 8500 字*
