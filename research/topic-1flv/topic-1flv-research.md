# 大模型推理服务弹性伸缩机制深度调研报告

**调研主题：** 大模型推理服务弹性伸缩机制
**所属域：** 大模型框架
**调研日期：** 2026-04-05
**报告版本：** 2.0

---

## 目录

1. [第一部分：概念剖析](#第一部分概念剖析)
2. [第二部分：行业情报](#第二部分行业情报)
3. [第三部分：方案对比](#第三部分方案对比)
4. [第四部分：精华整合](#第四部分精华整合)
5. [参考文献](#参考文献)

---

## 第一部分：概念剖析

### 1.1 定义澄清

#### 通行定义

**大模型推理服务弹性伸缩机制**（Elastic Autoscaling for LLM Inference Serving）是指在大语言模型（LLM）推理服务场景中，根据实时负载动态调整计算资源（GPU 实例数量、KVCache 分配、计算单元规模等）的系统能力。其核心目标是在满足服务等级目标（SLO）的前提下，最大化资源利用率和成本效益。

与传统的微服务弹性伸缩不同，LLM 推理弹性伸缩需要考虑以下特殊性：
- **请求异质性**：不同请求的 token 数量差异巨大（从几 token 到数十万 token）
- **两阶段特性**：Prefill（预填充）和 Decode（解码）阶段具有完全不同的资源特征
- **显存约束**：GPU 显存是核心瓶颈资源，而非 CPU 或内存
- **冷启动延迟**：模型加载和 KVCache 预热需要数秒至数十秒

#### 常见误解

| 误解 | 正确理解 |
|------|----------|
| **误解 1：弹性伸缩就是简单的 HPA（水平 Pod 自动伸缩）** | 传统 HPA 基于 CPU/内存指标，而 LLM 推理需要基于队列深度、等待请求数、GPU KVCache 利用率等专用指标进行弹性决策 |
| **误解 2：伸缩粒度只能是整卡/整节点** | 现代系统支持更细粒度的弹性，包括 KVCache 池的动态分配、专家并行（Expert Parallelism）的弹性调整、以及 Prefill/Decode 阶段的独立伸缩 |
| **误解 3：弹性伸缩只关注扩容，不关注缩容** | 缩容（Scale-down）同样重要，需要优雅地处理正在进行的请求、迁移 KVCache 状态，避免请求中断 |
| **误解 4：弹性伸缩是纯基础设施问题** | 弹性伸缩与调度算法、KVCache 管理、请求路由等应用层逻辑深度耦合，需要全栈协同设计 |

#### 边界辨析

| 概念 | 核心区别 |
|------|----------|
| **弹性伸缩 vs. 负载均衡** | 弹性伸缩关注资源数量的动态调整；负载均衡关注请求在现有资源间的分配。两者通常配合使用，但目标不同 |
| **弹性伸缩 vs. 模型压缩** | 弹性伸缩是系统层面的资源调度；模型压缩（量化、剪枝、蒸馏）是模型层面的优化。前者不影响模型精度，后者会 |
| **弹性伸缩 vs. 批处理优化** | 弹性伸缩是粗粒度的资源调整（秒级）；批处理优化（Continuous Batching）是细粒度的请求调度（毫秒级） |
| **弹性伸缩 vs. 多租户隔离** | 弹性伸缩关注资源总量调整；多租户隔离关注资源在租户间的公平分配和 QoS 保障 |

---

### 1.2 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    大模型推理服务弹性伸缩系统架构                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐ │
│   │  客户端请求   │ →  │  负载均衡器   │ →  │      请求路由器       │ │
│   └──────────────┘    └──────────────┘    └──────────┬───────────┘ │
│                                                       ↓             │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                      弹性伸缩控制器                            │ │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐│ │
│   │  │  指标采集器  │  │  伸缩决策器  │  │     资源执行器          ││ │
│   │  │  - 队列深度  │  │  - 阈值判断  │  │  - Kubernetes HPA     ││ │
│   │  │  - GPU 利用率  │  │  - 预测算法  │  │  - KEDA 事件驱动      ││ │
│   │  │  - KVCache   │  │  - 冷却策略  │  │  - 自定义资源调度      ││ │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────┘│ │
│   └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                      推理服务集群                              │ │
│   │  ┌─────────────────┐           ┌─────────────────────────────┐│ │
│   │  │  Prefill 节点池  │           │      Decode 节点池          ││ │
│   │  │  (计算密集型)    │ ←KVCache→ │     (内存密集型)            ││ │
│   │  │  - 高吞吐优化    │  共享池   │     - 低延迟优化            ││ │
│   │  └─────────────────┘           └─────────────────────────────┘│ │
│   └───────────────────────────────────────────────────────────────┘ │
│                              ↓                                      │
│   ┌───────────────────────────────────────────────────────────────┐ │
│   │                      监控与可观测性                            │ │
│   │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │ │
│   │  │   Prometheus  │  │  Grafana     │  │  分布式追踪 (Jaeger)│  │ │
│   │  └──────────────┘  └──────────────┘  └─────────────────────┘  │ │
│   └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**各组件职责说明：**

| 组件 | 职责 |
|------|------|
| **负载均衡器** | 接收客户端请求，基于一致性哈希或最少连接数进行初步分发 |
| **请求路由器** | 根据请求特征（prompt 长度、模型版本）路由到合适的服务实例 |
| **指标采集器** | 实时采集推理服务的核心指标（队列深度、等待请求数、GPU 显存使用率、KVCache 命中率） |
| **伸缩决策器** | 基于采集的指标，结合阈值规则或预测算法，做出伸缩决策 |
| **资源执行器** | 执行伸缩动作，如 Kubernetes HPA、KEDA 事件驱动伸缩、或自定义的资源调度 |
| **Prefill 节点池** | 专门处理计算密集的预填充阶段，可独立弹性伸缩 |
| **Decode 节点池** | 专门处理内存密集的解码阶段，可独立弹性伸缩 |
| **KVCache 共享池** | 跨节点的分布式 KVCache 存储，支持预填充和解耦节点间的 Cache 传输 |

---

### 1.3 数学形式化

#### 公式 1：弹性伸缩决策函数

$$\text{scale\_decision}(t) = \begin{cases}
\text{scale\_up} & \text{if } Q(t) > \theta_{high} \land \Delta Q > 0 \\
\text{scale\_down} & \text{if } Q(t) < \theta_{low} \land t - t_{last\_scale} > T_{cooldown} \\
\text{keep} & \text{otherwise}
\end{cases}$$

**解释：** 伸缩决策基于队列深度 $Q(t)$ 与阈值的比较，$\theta_{high}$ 和 $\theta_{low}$ 分别为扩容和缩容阈值，$T_{cooldown}$ 为防止频繁伸缩的冷却时间。

---

#### 公式 2：最优资源利用率模型

$$U^* = \arg\max_{n} \left[ \frac{\sum_{i=1}^{n} T_i}{n \cdot C \cdot T_{total}} \right] \quad \text{s.t.} \quad P_{slo\_violation} < \epsilon$$

**解释：** 最优资源利用率 $U^*$ 是在满足 SLO 违约概率小于 $\epsilon$ 的约束下，最大化总有效计算时间 $T_i$ 与总资源成本 $n \cdot C \cdot T_{total}$ 的比值。

---

#### 公式 3：Prefill-Decode 解耦延迟模型

$$L_{total} = L_{prefill} + L_{transfer} + L_{decode} = \frac{S_{prompt}}{B_{prefill}} + \frac{S_{kv}}{BW_{rdma}} + \frac{S_{output} \cdot T_{gen}}{B_{decode}}$$

**解释：** 总延迟由预填充时间（prompt 大小除以预填充吞吐）、KVCache 传输时间（KVCache 大小除以 RDMA 带宽）和解码时间（生成 token 数乘以每 token 时间除以解码吞吐）组成。

---

#### 公式 4：KVCache 弹性分配

$$C_{kv}^{(i)} = \frac{M_{gpu}^{(i)} - M_{model} - M_{overhead}}{S_{layer} \cdot N_{layer}} \cdot \alpha_{dynamic}$$

**解释：** 第 $i$ 个实例的 KVCache 容量由 GPU 显存减去模型权重和系统开销后，除以每层大小和层数，再乘以动态分配系数 $\alpha_{dynamic}$。

---

#### 公式 5：成本效益比

$$ROI = \frac{R_{revenue} \cdot N_{served}}{C_{gpu} \cdot N_{instances} \cdot T_{running} + C_{idle} \cdot N_{idle} \cdot T_{idle}}$$

**解释：** 投资回报率由服务收入与总成本的比值决定，总成本包括运行中的 GPU 成本和空闲时的资源浪费成本。

---

### 1.4 实现逻辑

```python
class ElasticLLMInferenceSystem:
    """
    大模型推理弹性伸缩系统核心抽象

    职责:
    - component_a (MetricsCollector): 实时采集推理服务指标
    - component_b (ScalingDecisionEngine): 基于指标做出伸缩决策
    - component_c (ResourceExecutor): 执行资源伸缩操作
    - component_d (RequestRouter): 请求路由与负载均衡
    """

    def __init__(self, config):
        self.metrics_collector = MetricsCollector(
            metrics=['queue_depth', 'gpu_utilization', 'kv_cache_usage']
        )
        self.scaling_engine = ScalingDecisionEngine(
            scale_up_threshold=config['scale_up_threshold'],
            scale_down_threshold=config['scale_down_threshold'],
            cooldown_period=config['cooldown_seconds']
        )
        self.resource_executor = ResourceExecutor(
            orchestrator=config['orchestrator'],  # kubernetes, keda, custom
            min_instances=config['min_instances'],
            max_instances=config['max_instances']
        )
        self.request_router = RequestRouter(
            strategy=config['routing_strategy'],  # round_robin, least_connections, kv_aware
            pd_disaggregation=config.get('pd_disaggregation', False)
        )

    def core_operation(self, request_batch):
        """
        核心操作：处理请求批并执行弹性伸缩决策

        流程:
        1. 采集当前系统指标
        2. 路由请求到合适的服务实例
        3. 评估是否需要伸缩
        4. 执行伸缩操作（如果需要）
        """
        # Step 1: 采集指标
        current_metrics = self.metrics_collector.collect()

        # Step 2: 请求路由
        if self.request_router.pd_disaggregation:
            prefill_requests, decode_requests = self._split_by_phase(request_batch)
            self.request_router.route(prefill_requests, target='prefill_pool')
            self.request_router.route(decode_requests, target='decode_pool')
        else:
            self.request_router.route(request_batch, target='unified_pool')

        # Step 3: 伸缩决策
        scaling_action = self.scaling_engine.decide(current_metrics)

        # Step 4: 执行伸缩
        if scaling_action != 'keep':
            self.resource_executor.execute(scaling_action)

        return self._get_serving_status()

    def _split_by_phase(self, requests):
        """将请求按 Prefill/Decode 阶段分离"""
        prefill_reqs = [r for r in requests if r.phase == 'prefill']
        decode_reqs = [r for r in requests if r.phase == 'decode']
        return prefill_reqs, decode_reqs


class MetricsCollector:
    """指标采集器：负责采集推理服务的核心指标"""

    def __init__(self, metrics):
        self.metrics = metrics
        self.prometheus_client = PrometheusClient()

    def collect(self):
        """采集所有配置的指标"""
        collected = {}
        for metric in self.metrics:
            if metric == 'queue_depth':
                collected[metric] = self._get_queue_depth()
            elif metric == 'gpu_utilization':
                collected[metric] = self._get_gpu_utilization()
            elif metric == 'kv_cache_usage':
                collected[metric] = self._get_kv_cache_usage()
        return collected

    def _get_queue_depth(self):
        """获取当前等待队列深度"""
        return self.prometheus_client.query('vllm:num_requests_waiting')

    def _get_gpu_utilization(self):
        """获取 GPU 利用率"""
        return self.prometheus_client.query('DCGM_FI_DEV_GPU_UTIL')

    def _get_kv_cache_usage(self):
        """获取 KVCache 使用率"""
        return self.prometheus_client.query('vllm:gpu_cache_usage_perc')


class ScalingDecisionEngine:
    """伸缩决策引擎：基于指标和策略做出伸缩决策"""

    def __init__(self, scale_up_threshold, scale_down_threshold, cooldown_period):
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_period = cooldown_period
        self.last_scale_time = None

    def decide(self, metrics):
        """基于当前指标做出伸缩决策"""
        if self._in_cooldown():
            return 'keep'

        queue_depth = metrics.get('queue_depth', 0)

        if queue_depth > self.scale_up_threshold:
            self.last_scale_time = time.time()
            return 'scale_up'
        elif queue_depth < self.scale_down_threshold:
            self.last_scale_time = time.time()
            return 'scale_down'

        return 'keep'

    def _in_cooldown(self):
        """检查是否在冷却期内"""
        if self.last_scale_time is None:
            return False
        return time.time() - self.last_scale_time < self.cooldown_period
```

---

### 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **弹性延迟** | < 30 秒 | 从触发条件到实例就绪的端到端时间 | 包括冷启动时间，对突发流量的响应能力至关重要 |
| **伸缩精度** | > 90% | 实际资源与目标资源的比值 | 避免过度伸缩造成的资源浪费 |
| **SLO 满足率** | > 99% | 满足延迟 SLO 的请求占比 | 弹性伸缩的首要目标是保障 SLO |
| **资源利用率** | 60-80% | GPU 有效计算时间占比 | 平衡成本和性能的关键指标 |
| **冷启动时间** | < 60 秒 | 从创建 Pod 到服务就绪的时间 | 对 scale-from-zero 场景至关重要 |
| **KVCache 迁移延迟** | < 100ms | 跨节点传输 KVCache 的时间 | PD 解耦架构的关键性能指标 |
| **成本节省率** | 30-50% | 相比静态资源配置的成本降低 | 弹性伸缩的核心价值体现 |

---

### 1.6 扩展性与安全性

#### 水平扩展（Scale-Out）

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **无状态扩展** | 通过增加服务实例来线性扩展容量 | 通用场景，最简单 |
| **PD 解耦扩展** | Prefill 和 Decode 节点池独立扩展 | 长文本、高并发场景 |
| **KVCache 池化** | 跨节点共享 KVCache，支持动态迁移 | 多轮对话、高 Cache 命中率场景 |
| **专家并行弹性** | 动态调整 MoE 模型中激活的专家数量 | 大模型、多租户场景 |

#### 垂直扩展（Scale-Up）

| 策略 | 说明 | 上限 |
|------|------|------|
| **GPU 显存优化** | 通过量化、Offloading 等技术扩展单卡支持的最大模型 | 受限于物理显存 |
| **张量并行扩展** | 单请求跨多卡并行处理 | 受限于互联带宽（NVLink/NVSwitch） |
| **流水线并行** | 模型层分布在多卡上 | 受限于流水线气泡 |

#### 安全考量

| 风险 | 防护措施 |
|------|----------|
| **资源耗尽攻击** | 请求速率限制、租户配额管理、异常检测 |
| **KVCache 污染** | 租户隔离、Cache 分区、TTL 过期策略 |
| **弹性震荡攻击** | 冷却时间、滞回阈值、异常模式检测 |
| **多租户侧信道** | 计算隔离、显存加密、时序混淆 |

---

## 第二部分：行业情报

### 2.1 GitHub 热门项目（15+ 个）

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | ~75k | 高吞吐推理引擎，PagedAttention、Continuous Batching | Python, CUDA, Triton | 2026-04 | [GitHub](https://github.com/vllm-project/vllm) |
| **SGLang** | ~25.4k | 高性能服务框架，RadixAttention、多轮对话优化 | Python, CUDA | 2026-04 | [GitHub](https://github.com/sgl-project/sglang) |
| **Text Generation Inference (TGI)** | ~15k | Hugging Face 官方推理服务，生产级部署 | Rust, Python, gRPC | 2026-03 | [GitHub](https://github.com/huggingface/text-generation-inference) |
| **Ray Serve** | ~35k (Ray 整体) | 分布式服务框架，LLM 应用弹性伸缩 | Python | 2026-04 | [GitHub](https://github.com/ray-project/ray) |
| **KServe** | ~5k | CNCF 孵化项目，Kubernetes 标准化推理服务 | Go, Kubernetes CRD | 2026-03 | [GitHub](https://github.com/kserve/kserve) |
| **BentoML** | ~18k | AI 应用部署框架，支持多种推理后端 | Python | 2026-04 | [GitHub](https://github.com/bentoml/BentoML) |
| **OpenLLM** | ~8k | 运行任意开源 LLM，BentoML 生态 | Python | 2026-03 | [GitHub](https://github.com/bentoml/OpenLLM) |
| **Mooncake** | ~3k | KVCache 中心化解耦架构，Moonshot AI 出品 | C++, Python, RDMA | 2026-02 | [GitHub](https://github.com/kvcache-ai/Mooncake) |
| **DistServe** | ~1.5k | Prefill-Decode 解耦推理系统 | Python, CUDA | 2025-12 | [GitHub](https://github.com/LLMServe/DistServe) |
| **vLLM Production Stack** | ~500 | vLLM 生产级参考架构，含 KEDA 集成 | Helm, Kubernetes | 2026-03 | [GitHub](https://github.com/vllm-project/production-stack) |
| **LMDeploy** | ~4k | 高效推理部署工具包，支持量化和 PD 解耦 | Python, C++ | 2026-04 | [GitHub](https://github.com/InternLM/lmdeploy) |
| **TensorRT-LLM** | ~10k | NVIDIA 官方推理优化库 | C++, CUDA | 2026-04 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **KEDA** | ~8k | Kubernetes 事件驱动自动伸缩 | Go, Kubernetes | 2026-03 | [GitHub](https://github.com/kedacore/keda) |
| **Ollama** | ~80k | 本地 LLM 运行工具，简化部署 | Go | 2026-04 | [GitHub](https://github.com/ollama/ollama) |
| **Triton Inference Server** | ~8k | NVIDIA 通用推理服务 | C++, Python | 2026-04 | [GitHub](https://github.com/triton-inference-server/server) |

---

### 2.2 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 | 链接 |
|------|----------|------|----------|---------|-----------|------|
| **Taming the Titans: A Survey of Efficient LLM Inference Serving** | Zhou et al. | 2025 | arXiv:2504.19720 | 系统性综述，覆盖实例级和集群级优化技术 | 高引用 | [arXiv](https://arxiv.org/abs/2504.19720) |
| **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** | Qin et al. (Moonshot AI) | 2024/2025 | FAST '25 Best Paper | KVCache 中心化架构，PD 解耦，支持 Kimi 服务 | FAST 2025 最佳论文 | [arXiv](https://arxiv.org/abs/2407.00079) |
| **DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving** | Zhong et al. (PKU, UCSD) | 2024 | OSDI '24 | 首创 Prefill-Decode 解耦架构，优化 goodput | OSDI 接收 | [arXiv](https://arxiv.org/abs/2401.09670) |
| **Hierarchical Autoscaling for Large Language Model Serving with Chiron** | Patke et al. | 2025 | arXiv:2501.08090 | Chiron 层次化弹性伸缩，基于背压估计 | 新兴 SOTA | [arXiv](https://arxiv.org/abs/2501.08090) |
| **Enabling Fast Scaling for Serverless Large Language Model Inference** | Wang et al. | 2025 | arXiv:2502.09922 | λScale 无服务器快速弹性系统 | 新兴 SOTA | [arXiv](https://arxiv.org/abs/2502.09922) |
| **A Survey of LLM Inference Systems** | Liu et al. | 2025 | arXiv:2506.21901 | 从算子到系统级的推理技术综述 | 高引用 | [arXiv](https://arxiv.org/abs/2506.21901) |
| **Unlock the Potential of Fine-grained LLM Serving via Dynamic Module Migration** | Chen et al. | 2025 | arXiv:2507.18006 | 细粒度模块级迁移和复制 | 前沿研究 | [arXiv](https://arxiv.org/abs/2507.18006) |
| **Towards Resiliency in Large Language Model Serving with KevlarFlow** | Li et al. | 2026 | arXiv:2601.22438 | KevlarFlow 弹性恢复系统，MTTR 降低 20 倍 | 最新研究 | [arXiv](https://arxiv.org/abs/2601.22438) |
| **WindServe: Efficient Phase-Disaggregated LLM Serving with Stream Scheduling** | Zhang et al. | 2025 | ACM SIGMETRICS | 基于流的阶段解耦调度 | 顶会论文 | [ACM DL](https://dl.acm.org/doi/10.1145/3695053.3730999) |
| **Prefill-Decode Aggregation or Disaggregation? Unifying Both with TaiChi** | Wang et al. | 2025 | arXiv:2508.01989 | TaiChi 统一聚合与解耦架构 | 前沿研究 | [arXiv](https://arxiv.org/html/2508.01989v1) |
| **NanoFlow: Towards Optimal Large Language Model Serving Throughput** | Yu et al. | 2025 | USENIX NSDI | 最优吞吐调度策略，层次化弹性参考 | 顶会论文 | [USENIX](https://dl.acm.org/doi/10.5555/3767901.3767942) |
| **JITServe: SLO-aware LLM Serving with Imprecise Request Information** | Zhang et al. | 2025 | arXiv:2504.20068 | 不确定请求信息下的 SLO 感知调度 | 新兴研究 | [arXiv](https://arxiv.org/abs/2504.20068) |

---

### 2.3 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| **State of the Model Serving Communities - January 2026** | InferenceOps | 英文 | 社区动态 | vLLM、SGLang 等社区最新进展和路线图 | 2026-01 | [Substack](https://inferenceops.substack.com/p/state-of-the-model-serving-communities-3d1) |
| **A Practical Guide to LLM Inference at Scale** | The Neural Maze | 英文 | 深度教程 | 大规模推理架构模式和最佳实践 | 2025-11 | [Substack](https://theneuralmaze.substack.com/p/a-practical-guide-to-llm-inference) |
| **What goes into an inference stack?** | Nikitha's Newsletter | 英文 | 架构解析 | 现代推理栈各层组件详解 | 2025-08 | [Substack](https://nikitha.substack.com/p/what-goes-into-an-inference-stack) |
| **Ollama vs vLLM: A Comprehensive Guide to Local LLM Serving** | Mustafa Genc | 英文 | 对比评测 | 两大开源引擎的全面对比 | 2026-02 | [Medium](https://medium.com/@mustafa.gencc94/ollama-vs-vllm-a-comprehensive-guide-to-local-llm-serving-91705ec50c1d) |
| **Hosting LLMs — From Fundamentals to Scaled Production** | Xiaxiami | 英文 | 实战教程 | 从基础到生产级部署的完整指南 | 2025-12 | [Medium](https://medium.com/@xiaxiami/hosting-llms-from-fundamentals-to-scaled-production-with-hands-on-tutorial-6598d16810e0) |
| **ML Inference Runtimes in 2026: An Architect's Guide** | Digvijay July | 英文 | 架构指南 | Triton、TensorRT、ONNX Runtime 对比 | 2025-12 | [Medium](https://medium.com/@digvijay17july/ml-inference-runtimes-in-2026-an-architects-guide-to-choosing-the-right-engine-d3989a87d052) |
| **Reducing LLM Inference Cost: A Practical Guide** | Vyaswanth | 英文 | 优化指南 | 推理成本优化和工程实践 | 2026-03 | [Medium](https://medium.com/@vyaswanth965/reducing-llm-inference-cost-a-practical-guide-to-optimization-inference-engineering-984022586def) |
| **How to set up KServe autoscaling for vLLM with KEDA** | Red Hat AI Team | 英文 | 实战教程 | KServe + vLLM + KEDA 集成配置 | 2025-09 | [Red Hat](https://developers.redhat.com/articles/2025/09/23/how-set-kserve-autoscaling-vllm-keda) |
| **Scaling LLM Workloads on Kubernetes: A Production Engineer's Guide** | Zartis Engineering | 英文 | 生产实践 | Kubernetes 上 LLM 负载的扩展经验 | 2025-10 | [Zartis](https://www.zartis.com/scaling-llm-workloads-on-kubernetes-a-production-engineers-guide/) |
| **基于 vllm 自定义指标的多集群 HPA 实践** | 阿里云容器团队 | 中文 | 实战教程 | 阿里云 ACK 上 vLLM 多集群 HPA 实践 | 2025-11 | [阿里云](https://help.aliyun.com/zh/ack/distributed-cloud-container-platform-for-kubernetes/use-cases/implement-multi-cluster-hpa-with-vllm-custom-metrics) |

---

### 2.4 技术演进时间线

| 时间 | 事件 | 发起方 | 影响 |
|------|------|--------|------|
| **2023 Q2** | vLLM 发布，引入 PagedAttention | UC Berkeley | 开启了高效 LLM 推理的新纪元 |
| **2023 Q3** | TGI 成为 Hugging Face 官方推理后端 | Hugging Face | 推动了 Rust 在推理服务中的应用 |
| **2023 Q4** | Ray Serve 增强 LLM 支持 | Anyscale | 为分布式 LLM 应用提供了弹性基础 |
| **2024 Q1** | DistServe 提出 Prefill-Decode 解耦 | 北大 & UCSD | 开创了 PD 解耦架构新方向 |
| **2024 Q2** | KServe 成为 CNCF 孵化项目 | CNCF | 标准化了 Kubernetes 推理服务 |
| **2024 Q3** | Mooncake 架构支撑 Kimi 服务 | Moonshot AI | 验证了 KVCache 中心化架构的生产可行性 |
| **2025 Q1** | vLLM Production Stack 发布 | vLLM Team | 提供了生产级参考架构 |
| **2025 Q2** | KServe v0.15 发布，增强 KEDA 集成 | KServe Team | 改进了事件驱动弹性伸缩能力 |
| **2025 Q3** | Chiron 层次化弹性伸缩论文发表 | IBM Research | 提出了基于背压估计的新方法 |
| **2025 Q4** | λScale 无服务器快速弹性系统 | 学术界 | 解决了 serverless 场景的冷启动问题 |
| **2026 Q1** | KevlarFlow 弹性恢复系统 | 学术界 | 将故障恢复时间缩短至 30 秒 |
| **2026 Q2** | **当前状态**：PD 解耦成为主流，KEDA + 自定义指标成为弹性伸缩标准实践 | 行业共识 | 弹性伸缩进入成熟应用阶段 |

---

## 第三部分：方案对比

### 3.1 历史发展时间线

```
2023 ─┬─ vLLM 发布 (PagedAttention) → 开启高效 LLM 推理新纪元
      │
2024 ─┼─ DistServe 提出 PD 解耦 → 分离 Prefill/Decode 阶段优化
      │
2024 ─┼─ Mooncake 架构落地 (Kimi 服务) → KVCache 中心化架构验证
      │
2025 ─┼─ Chiron/λScale 等弹性伸缩系统发表 → 层次化、快速弹性成为研究热点
      │
2025 ─┼─ vLLM Production Stack 发布 → 生产级参考架构成熟
      │
2026 ─┴─ 当前状态：PD 解耦 + KEDA 事件驱动弹性成为行业标准实践
```

---

### 3.2 五种方案横向对比

| 方案 | 原理 | 优点（3+） | 缺点（3+） | 适用场景 | 成本量级 |
|------|------|-----------|-----------|---------|---------|
| **Kubernetes HPA + 自定义指标** | 基于 Prometheus 采集的推理专用指标（队列深度、等待请求数）触发 Pod 伸缩 | 1. 原生 K8s 支持，无需额外组件<br>2. 配置简单，学习曲线低<br>3. 生态成熟，文档丰富 | 1. 响应延迟较高（30-60 秒）<br>2. 不支持 scale-to-zero<br>3. 指标更新频率受限 | 中小型生产环境，已有 K8s 基础设施的团队 | $ |
| **KEDA 事件驱动弹性** | 基于外部事件源（Prometheus、Kafka 等）触发伸缩，支持 scale-to-zero | 1. 支持 scale-to-zero，节省空闲成本<br>2. 响应更快（10-30 秒）<br>3. 支持多种事件源，灵活性强 | 1. 需要额外部署 KEDA 组件<br>2. 配置相对复杂<br>3. 对事件源依赖性强 | 流量波动大的场景，Serverless 部署 | $$ |
| **Ray Serve 应用级弹性** | Ray 框架内的应用级弹性伸缩，支持跨部署协调 | 1. 支持复杂的多模型流水线<br>2. 细粒度的资源控制<br>3. 与 Ray 生态深度集成 | 1. 学习曲线较陡<br>2. 需要 Ray 运行时环境<br>3. 社区相对较小 | 复杂 LLM 应用，多模型协同场景 | $$$ |
| **PD 解耦弹性 (Mooncake/DistServe)** | 将 Prefill 和 Decode 阶段分离到不同节点池，独立弹性伸缩 | 1. 资源利用率显著提升（30-50%）<br>2. 针对两阶段特性优化<br>3. 支持长文本和高并发 | 1. 架构复杂度高<br>2. 需要 KVCache 传输机制<br>3. 部署运维成本高 | 长文本、高并发、生产级大规模部署 | $$$$ |
| **层次化弹性 (Chiron)** | 基于背压估计的层次化伸缩，结合实例级和集群级决策 | 1. 伸缩决策更精准<br>2. 减少过度伸缩<br>3. 适应复杂负载模式 | 1. 实现复杂度高<br>2. 需要定制化开发<br>3. 社区成熟度低 | 超大规模部署，对成本敏感的场景 | $$$$ |

---

### 3.3 技术细节对比

| 维度 | HPA+ 自定义指标 | KEDA | Ray Serve | PD 解耦 | 层次化弹性 |
|------|---------------|------|-----------|---------|-----------|
| **性能** | 响应延迟 30-60 秒 | 响应延迟 10-30 秒 | 响应延迟 5-20 秒 | 资源利用率提升 30-50% | 过度伸缩减少 40% |
| **易用性** | ⭐⭐⭐⭐⭐ (最简单) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐ |
| **生态成熟度** | ⭐⭐⭐⭐⭐ (K8s 原生) | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ (新兴) | ⭐⭐ (研究阶段) |
| **社区活跃度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **学习曲线** | 低 | 中 | 高 | 高 | 高 |
| **scale-to-zero** | ❌ | ✅ | ⚠️ (部分支持) | ❌ | ⚠️ |
| **多集群支持** | ⚠️ (需额外配置) | ⚠️ | ✅ | ❌ | ❌ |
| **KVCache 感知** | ❌ | ❌ | ⚠️ | ✅ | ⚠️ |

---

### 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证** | Kubernetes HPA + 自定义指标 | 配置简单，快速上手，成本最低 | $500-2,000 |
| **流量波动大的 ToC 应用** | KEDA 事件驱动弹性 | 支持 scale-to-zero，闲时成本极低 | $2,000-10,000 |
| **中型生产环境** | KEDA + vLLM Production Stack | 生产级参考架构，平衡性能和成本 | $10,000-50,000 |
| **多模型复杂应用** | Ray Serve | 支持复杂流水线，细粒度资源控制 | $20,000-100,000 |
| **长文本高并发场景** | PD 解耦 (Mooncake/DistServe) | 资源利用率提升显著，适合大规模 | $50,000-200,000 |
| **超大规模成本敏感** | 层次化弹性 (Chiron) | 减少过度伸缩，最大化资源效率 | $100,000+ |

**成本估算假设：** 基于 100 万 tokens/日的基准流量，GPU 实例采用 NVIDIA A10/A100，按需计费。实际成本因云厂商、地域、预留实例等因素而异。

---

## 第四部分：精华整合

### 4.1 The One 公式

$$\text{LLM 弹性伸缩} = \underbrace{\text{指标感知}}_{\text{队列深度 + GPU 利用率}} + \underbrace{\text{决策智能}}_{\text{阈值/预测}} - \underbrace{\text{冷启动损耗}}_{\text{模型加载 + Cache 预热}}$$

**解读：** 弹性伸缩的本质是在准确的负载感知和智能决策的基础上，最小化冷启动带来的性能损耗。

---

### 4.2 一句话解释

> **大模型推理弹性伸缩就像"智能电梯调度系统"：根据候梯人数（队列深度）和楼层分布（请求特征），动态调整运行的电梯数量（GPU 实例），既不让乘客等太久（SLO 保障），也不让电梯空跑浪费电（成本优化）。**

---

### 4.3 核心架构图

```
请求 → [指标采集] → [伸缩决策] → [资源执行] → 服务响应
           ↓             ↓             ↓
       队列深度      阈值/预测    K8s/KEDA
       GPU 利用率     冷却策略    自定义调度
       KVCache      滞回控制    PD 节点池
```

---

### 4.4 STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation（背景 + 痛点）** | 大模型推理服务面临流量波动大、资源成本高的双重挑战。传统静态资源配置要么在高峰期无法满足 SLO，要么在低谷期造成大量资源浪费。同时，LLM 推理的 Prefill/Decode 两阶段特性、KVCache 显存约束、冷启动延迟等问题，使得通用弹性伸缩方案难以直接适用。 |
| **Task（核心问题）** | 如何在保障 SLO（延迟、吞吐）的前提下，动态调整 GPU 资源，最大化资源利用率并降低成本？关键约束包括：GPU 冷启动时间（30-60 秒）、KVCache 状态迁移、Prefill/Decode 阶段异质性、多租户隔离需求。 |
| **Action（主流方案）** | 技术演进经历了三个阶段：(1) **基础弹性**：基于 K8s HPA 的 CPU/内存指标伸缩；(2) **指标感知弹性**：基于推理专用指标（队列深度、GPU 利用率、KVCache）的 KEDA 事件驱动伸缩；(3) **架构感知弹性**：PD 解耦（Mooncake/DistServe）将 Prefill/Decode 分离独立伸缩，层次化弹性（Chiron）基于背压估计做出精准决策。 |
| **Result（效果 + 建议）** | 当前最佳实践可将资源利用率提升至 60-80%，成本降低 30-50%，SLO 满足率>99%。建议：中小项目采用 KEDA + vLLM Production Stack；大规模生产环境考虑 PD 解耦架构；成本敏感场景可探索层次化弹性。未来趋势是无服务器快速弹性（λScale）和细粒度模块级迁移。 |

---

### 4.5 理解确认问题

**问题：** 为什么基于 CPU/内存利用率的传统 HPA 不适合大模型推理服务的弹性伸缩？请从 LLM 推理的资源特征和指标敏感性两个角度分析。

**参考答案：**

从**资源特征**角度：LLM 推理的核心瓶颈是 GPU 显存（用于 KVCache）和计算单元，而非 CPU 或系统内存。一个 GPU 实例可能 CPU 利用率很低（<10%），但 KVCache 已满无法接收新请求，此时基于 CPU 的 HPA 不会触发扩容，导致请求排队延迟飙升。

从**指标敏感性**角度：传统 HPA 的指标更新频率通常为 15-30 秒，而 LLM 推理的队列深度可能在几秒内从 0 暴涨到数百。基于推理专用指标（如 `vllm:num_requests_waiting`）的弹性系统可以在 5-10 秒内检测到负载变化并触发扩容，响应速度提升 3 倍以上。

---

## 参考文献

### 核心论文
1. Taming the Titans: A Survey of Efficient LLM Inference Serving - arXiv:2504.19720
2. Mooncake: A KVCache-centric Disaggregated Architecture - arXiv:2407.00079
3. DistServe: Disaggregating Prefill and Decoding - OSDI '24
4. Hierarchical Autoscaling with Chiron - arXiv:2501.08090

### 开源项目
1. vLLM: https://github.com/vllm-project/vllm
2. SGLang: https://github.com/sgl-project/sglang
3. KServe: https://github.com/kserve/kserve
4. KEDA: https://github.com/kedacore/keda

### 技术博客
1. InferenceOps Newsletter: https://inferenceops.substack.com
2. The Neural Maze: https://theneuralmaze.substack.com
3. Red Hat AI Blog: https://developers.redhat.com

---

**报告完成日期：** 2026-04-05
**总字数：** 约 8,500 字
**数据来源：** WebSearch 实时检索（2026-04-05）
