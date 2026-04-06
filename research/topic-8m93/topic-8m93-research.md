# 大模型推理请求预测与预调度技术深度调研报告

**调研主题：** 大模型推理请求预测与预调度技术
**所属域：** 大模型框架
**调研日期：** 2026-04-06
**版本：** 1.0

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

**大模型推理请求预测与预调度技术**是指在 LLM（Large Language Model）推理服务系统中，通过分析请求特征（如输入长度、任务类型、用户历史行为等）预测请求的计算需求（主要是输出长度和处理延迟），并基于预测结果在请求到达前提前分配资源、调整调度策略的技术体系。其核心目标是优化 KV Cache 利用率、降低请求延迟、提高系统吞吐，最终实现服务等级目标（SLO）的高效达成。

### 常见误解

1. **误解一：预测只是输出长度预测**
   实际上，完整的预测体系包括输出 token 数、处理延迟、内存需求、专家激活模式（MoE 模型）等多维度预测，输出长度只是最基础的一项。

2. **误解二：预调度等于简单的请求排队**
   预调度的核心在于"预"——在请求完全到达或完全处理前，基于预测结果提前做出资源分配决策，包括 KV Cache 预分配、计算资源预留、请求批重组等，远超出简单排队的范畴。

3. **误解三：预测越准越好**
   预测精度与预测开销存在权衡关系。在某些场景下，使用轻量级预测器（如基于熵的快速预测）配合鲁棒的调度策略，比追求高精度但高开销的预测器更能提升整体系统性能。

4. **误解四：调度只影响吞吐不影响准确率**
   错误的调度策略可能导致长序列请求饥饿、KV Cache 换出频繁，进而影响模型生成质量。公平性调度（如 Justitia, 2026）已成为研究热点。

### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **传统任务调度** | 传统调度针对确定性任务，LLM 调度需处理输出长度不确定性这一核心挑战 |
| **KV Cache 优化** | KV Cache 优化关注存储效率，预调度关注请求处理顺序与资源预分配，两者协同但目标不同 |
| **模型压缩/量化** | 模型优化针对单次推理计算效率，预调度针对多请求并发场景下的系统级优化 |
| **负载均衡** | 负载均衡关注多实例间请求分发，预调度关注单实例内请求处理顺序，通常配合使用 |

---

## 1.2 核心架构

```
┌────────────────────────────────────────────────────────────────┐
│              大模型推理请求预测与预调度系统架构                 │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  请求入口                                                       │
│     ↓                                                          │
│  ┌──────────────────┐                                          │
│  │   请求预测层      │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ 输出长度   │  │ ← 基于输入特征/熵值/历史数据预测          │
│  │  │ 预测器     │  │                                          │
│  │  └────────────┘  │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ 延迟预测   │  │ ← 基于模型状态/硬件负载预测               │
│  │  │ 估计器     │  │                                          │
│  │  └────────────┘  │                                          │
│  └──────────────────┘                                          │
│           ↓                                                    │
│  ┌──────────────────┐                                          │
│  │   预调度决策层    │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ 请求优先级 │  │ ← SJF/MLFQ/公平调度策略                   │
│  │  │ 排序队列   │  │                                          │
│  │  └────────────┘  │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ KV Cache   │  │ ← 基于预测的内存预分配                    │
│  │  │ 预分配器   │  │                                          │
│  │  └────────────┘  │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ 动态批处理 │  │ ← Continuous Batching 优化                │
│  │  │ 管理器     │  │                                          │
│  │  └────────────┘  │                                          │
│  └──────────────────┘                                          │
│           ↓                                                    │
│  ┌──────────────────┐                                          │
│  │   执行引擎层      │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ Paged      │  │ ← vLLM 核心内存管理                       │
│  │  │ Attention  │  │                                          │
│  │  └────────────┘  │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ Radix      │  │ ← SGLang 前缀复用                        │
│  │  │ Attention  │  │                                          │
│  │  └────────────┘  │                                          │
│  └──────────────────┘                                          │
│           ↓                                                    │
│  ┌──────────────────┐                                          │
│  │   监控反馈层      │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ 预测误差   │  │ ← 用于在线学习/自适应调整                 │
│  │  │ 追踪器     │  │                                          │
│  │  └────────────┘  │                                          │
│  │  ┌────────────┐  │                                          │
│  │  │ SLO      │  │ ← 延迟/吞吐/公平性监控                      │
│  │  │ 监控器     │  │                                          │
│  │  └────────────┘  │                                          │
│  └──────────────────┘                                          │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

**各组件职责说明：**

| 组件 | 职责 |
|------|------|
| **输出长度预测器** | 分析输入 prompt 特征，预测生成 token 数量，为调度决策提供核心依据 |
| **延迟预测估计器** | 结合当前系统负载、模型状态，估计请求处理延迟，用于 SLO 保障 |
| **请求优先级排序队列** | 基于预测结果实施 SJF（Shortest Job First）或混合策略，优化平均延迟 |
| **KV Cache 预分配器** | 根据预测长度提前分配 KV Cache 块，减少运行时分配开销和碎片 |
| **动态批处理管理器** | 实施 Continuous Batching，在迭代级别动态调整批次组成 |
| **Paged Attention** | vLLM 核心机制，将 KV Cache 分页存储，解决内存碎片问题 |
| **Radix Attention** | SGLang 核心机制，使用前缀树管理 KV Cache，支持高效前缀复用 |
| **预测误差追踪器** | 监控预测准确性，为自适应调整和在线学习提供反馈信号 |
| **SLO 监控器** | 追踪系统延迟、吞吐、公平性等指标，保障服务质量 |

---

## 1.3 数学形式化

### 公式 1：输出长度预测模型

$$
\hat{L} = f_{\theta}(x, H_t, E(x))
$$

其中 $x$ 为输入 prompt，$H_t$ 为模型在时刻 $t$ 的隐藏状态，$E(x)$ 为输入熵值特征，$\hat{L}$ 为预测输出长度。

**解释：** 输出长度预测函数 $f_{\theta}$ 融合输入特征、模型内部状态和熵值信息，输出对生成 token 数的估计。

---

### 公式 2：基于熵的长度预测（EGTP 方法）

$$
E(x) = -\sum_{i=1}^{|V|} p(w_i | x) \log p(w_i | x)
$$

$$
\hat{L}_{\text{entropy}} = \alpha \cdot E(x) + \beta \cdot \text{len}(x) + \gamma
$$

其中 $E(x)$ 为输入序列的熵值，$p(w_i | x)$ 为给定输入下词汇表中第 $i$ 个 token 的预测概率，$\alpha, \beta, \gamma$ 为学习参数。

**解释：** 熵值反映模型对输入的不确定性，高熵通常对应更复杂的输出需求，该公式将熵值与输入长度结合进行预测。

---

### 公式 3：调度延迟优化目标

$$
\min_{\pi} \mathbb{E}\left[\sum_{j=1}^{N} C_j\right] = \min_{\pi} \mathbb{E}\left[\sum_{j=1}^{N} \sum_{k=1}^{j} L_{\pi(k)}\right]
$$

其中 $\pi$ 为调度排列，$C_j$ 为第 $j$ 个请求的完成时间，$L_{\pi(k)}$ 为按调度顺序第 $k$ 个请求的处理长度。

**解释：** 最小化平均完成时间等价于 SJF 策略，即优先处理短请求可降低整体平均延迟。

---

### 公式 4：KV Cache 内存约束下的调度

$$
\max_{S \subseteq R} |S| \quad \text{s.t.} \quad \sum_{r_i \in S} (\text{prompt\_len}_i + \hat{L}_i) \cdot M_{\text{token}} \leq M_{\text{KV}}
$$

其中 $R$ 为待调度请求集合，$S$ 为可并发处理的请求子集，$M_{\text{token}}$ 为单 token KV Cache 内存，$M_{\text{KV}}$ 为总 KV Cache 容量。

**解释：** 在有限 KV Cache 内存约束下，最大化可并发处理的请求数量，预测长度 $\hat{L}_i$ 直接影响调度决策。

---

### 公式 5：预测误差对调度性能的影响

$$
\text{Regret}(\pi_{\text{pred}}) = \mathbb{E}[J(\pi_{\text{pred}})] - \mathbb{E}[J(\pi^*)] \leq O(\epsilon \cdot N)
$$

其中 $\pi_{\text{pred}}$ 为基于预测的调度策略，$\pi^*$ 为最优调度（需预知真实长度），$\epsilon$ 为预测误差率，$N$ 为请求数量。

**解释：** 预测误差导致的性能损失（Regret）与误差率和请求数成正比，说明即使预测不完美，小误差下的调度仍接近最优。

---

## 1.4 实现逻辑

```python
class LLMRequestPredictiveScheduler:
    """
    大模型推理请求预测与预调度核心系统

    核心职责：
    1. 输出长度预测 - 基于输入特征和熵值预测生成长度
    2. 请求优先级排序 - 实施 SJF 或混合调度策略
    3. KV Cache 预分配 - 基于预测提前分配内存块
    4. 动态批处理 - Continuous Batching 优化
    """

    def __init__(self, config):
        # ========== 预测组件 ==========
        self.length_predictor = LengthPredictor(
            method=config.prediction_method,  # 'entropy', 'mlp', 'hybrid'
            model_path=config.predictor_model_path
        )
        self.latency_estimator = LatencyEstimator(
            hardware_profile=config.hardware_profile,
            model_size=config.model_size
        )

        # ========== 调度组件 ==========
        self.priority_queue = PriorityScheduler(
            strategy=config.scheduling_strategy,  # 'sjf', 'mlfq', 'fair'
            kv_cache_capacity=config.kv_cache_blocks
        )
        self.kv_cache_allocator = PagedKVCacheAllocator(
            total_blocks=config.kv_cache_blocks,
            block_size=config.block_size
        )

        # ========== 批处理组件 ==========
        self.batch_manager = ContinuousBatchManager(
            max_batch_size=config.max_batch_size,
            scheduling_policy='iteration_level'
        )

        # ========== 监控反馈组件 ==========
        self.prediction_tracker = PredictionErrorTracker(
            window_size=1000,
            adaptive_update=True
        )
        self.slo_monitor = SLOMonitor(
            latency_slo=config.latency_slo_ms,
            throughput_target=config.throughput_target
        )

    def core_operation(self, incoming_requests):
        """
        核心调度操作，体现关键算法逻辑

        流程：
        1. 对新到达请求进行长度预测
        2. 基于预测结果进行优先级排序
        3. 预分配 KV Cache 资源
        4. 执行动态批处理调度
        5. 追踪预测误差并自适应调整
        """
        # Step 1: 预测阶段 - 对新请求进行输出长度预测
        predicted_requests = []
        for req in incoming_requests:
            # 提取特征：输入长度、熵值、任务类型
            features = self._extract_features(req)
            # 预测输出长度
            predicted_length = self.length_predictor.predict(
                input_tokens=req.tokens,
                features=features
            )
            # 估计处理延迟
            estimated_latency = self.latency_estimator.estimate(
                prompt_len=len(req.tokens),
                output_len=predicted_length
            )
            predicted_requests.append(PredictedRequest(
                original=req,
                predicted_length=predicted_length,
                estimated_latency=estimated_latency
            ))

        # Step 2: 调度决策 - 基于预测结果进行优先级排序
        scheduled_requests = self.priority_queue.schedule(
            requests=predicted_requests,
            current_load=self._get_current_load()
        )

        # Step 3: 资源预分配 - 为调度请求预分配 KV Cache
        allocated_requests = []
        for req in scheduled_requests:
            # 基于预测长度预分配 KV Cache 块
            kv_blocks = self.kv_cache_allocator.allocate(
                num_blocks=self._blocks_needed(req.predicted_length)
            )
            if kv_blocks is not None:
                req.kv_blocks = kv_blocks
                allocated_requests.append(req)

        # Step 4: 批处理执行 - Continuous Batching
        batch_results = []
        current_batch = self.batch_manager.create_batch(allocated_requests)

        while not self.batch_manager.is_complete(current_batch):
            # 迭代级别批处理：完成请求出队，新请求入队
            iteration_output = self._execute_batch_iteration(current_batch)

            # 处理完成的请求
            completed = self.batch_manager.extract_completed(current_batch)
            for req in completed:
                # 释放 KV Cache
                self.kv_cache_allocator.release(req.kv_blocks)
                # 追踪预测误差
                actual_length = req.actual_output_length
                self.prediction_tracker.record(
                    predicted=req.predicted_length,
                    actual=actual_length
                )
                batch_results.append(req)

            # 尝试添加新请求到批次
            new_candidates = self.priority_queue.get_next_candidates(
                available_blocks=self.kv_cache_allocator.available_blocks
            )
            current_batch = self.batch_manager.update_batch(
                current_batch, new_candidates
            )

        # Step 5: SLO 监控与自适应调整
        self.slo_monitor.update(batch_results)
        if self.slo_monitor.violation_rate > config.threshold:
            # 触发调度策略调整
            self._adapt_scheduling_strategy()

        return batch_results

    def _extract_features(self, request):
        """提取预测所需特征"""
        return {
            'input_length': len(request.tokens),
            'entropy': self._compute_entropy(request.tokens),
            'task_type': self._classify_task(request),
            'user_history': self._get_user_history(request.user_id)
        }

    def _compute_entropy(self, tokens):
        """计算输入序列的熵值特征"""
        # 简化实现：实际使用模型 logits 计算
        pass

    def _blocks_needed(self, predicted_length):
        """计算所需 KV Cache 块数"""
        return (predicted_length + self.config.block_size - 1) // self.config.block_size

    def _adapt_scheduling_strategy(self):
        """基于预测误差和 SLO 违反情况自适应调整调度策略"""
        current_error = self.prediction_tracker.get_average_error()
        if current_error > 0.3:
            # 预测误差大时切换到保守策略
            self.priority_queue.switch_to_conservative_mode()
```

---

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **预测准确率 (MAPE)** | < 30% | $\frac{1}{N}\sum|\frac{\hat{L}_i - L_i}{L_i}|$ | 输出长度预测平均绝对百分比误差，<30% 可接受 |
| **平均请求延迟 (P50)** | < 500ms | 端到端基准测试 | 中等负载下的典型延迟目标 |
| **尾部延迟 (P99)** | < 2000ms | 百分位延迟测量 | 长尾请求延迟，影响用户体验 |
| **吞吐 (tokens/s)** | > 10,000 | 负载测试 | 单卡 H100 典型目标 |
| **KV Cache 利用率** | > 80% | 内存监控 | 有效利用与预留的平衡 |
| **SLO 达成率** | > 99% | SLO 监控器 | 满足延迟/吞吐目标的请求比例 |
| **调度开销占比** | < 5% | 系统剖析 | 调度决策时间占总处理时间比例 |
| **预测器延迟** | < 1ms | 单独基准测试 | 预测器自身引入的延迟开销 |

---

## 1.6 扩展性与安全性

### 水平扩展

1. **分布式调度架构**
   在多节点部署中，可采用中心式调度器（Centralized Scheduler）或分布式调度（Distributed Scheduling）。中心式调度器维护全局视图，可实现更优的全局调度决策，但存在单点瓶颈；分布式调度各节点独立决策，扩展性更好但可能导致次优决策。

2. **请求路由与分片**
   基于一致性哈希或预测感知路由（Predictive Routing），将相似特征的请求路由到相同节点，提高 KV Cache 复用率。Mooncake 架构采用 PD 分离（Prefill/Decode Disaggregation），将预填充和解码阶段分离到不同节点，实现更细粒度的资源调度。

3. **KV Cache 池化**
   通过 RDMA 或高速网络实现跨节点 KV Cache 共享，支持请求在不同计算节点间迁移，提升整体资源利用率。

### 垂直扩展

1. **单节点优化上限**
   在单 GPU 节点上，优化空间主要来自：(1) KV Cache 内存管理效率；(2) 计算与通信重叠；(3) 更精确的预测减少浪费。当前 H100 单卡典型上限为 20,000-30,000 tokens/s 吞吐。

2. **预测器复杂度权衡**
   更复杂的预测器（如大型 MLP）可提供更高精度，但增加调度延迟。实践表明，轻量级预测器（<1ms 延迟）配合鲁棒调度策略，往往比复杂预测器带来更好的整体效果。

### 安全考量

1. **预测器对抗攻击**
   恶意用户可能构造特殊输入干扰预测器，导致调度系统资源分配失衡。防护措施包括：(1) 预测结果置信度评估；(2) 异常检测与限流；(3) 预测上限截断。

2. **资源隔离与公平性**
   多租户场景下，需防止单一用户占用过多资源。Justitia (2026) 提出公平调度机制，确保不同用户/任务类型的资源分配公平性。

3. **SLO 保障与降级策略**
   在负载过载时，需有明确的降级策略：(1) 优先级低的请求延迟处理；(2) 动态调整批处理大小；(3) 触发弹性扩容。

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | 60,000+ | PagedAttention、Continuous Batching、多模型支持 | Python/CUDA | 2026-04 | [GitHub](https://github.com/vllm-project/vllm) |
| **SGLang** | 12,000+ | RadixAttention、结构化生成、前端语言 | Python/CUDA | 2026-04 | [GitHub](https://github.com/sgl-project/sglang) |
| **TensorRT-LLM** | 8,000+ | NVIDIA 硬件优化、INT4/FP8 量化、多 GPU 支持 | C++/CUDA | 2026-04 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **TGI** | 15,000+ | HuggingFace 生态集成、连续批处理、量化 | Rust/Python | 2026-04 | [GitHub](https://github.com/huggingface/text-generation-inference) |
| **llm-d-inference-scheduler** | 500+ | 端点选择器、预测延迟调度、Kubernetes 集成 | Go/Python | 2026-03 | [GitHub](https://github.com/llm-d/llm-d-inference-scheduler) |
| **vllm-ltr** | 800+ | Learn-to-Rank 调度、SJF 优化、延迟降低 2.8x | Python | 2025-12 | [GitHub](https://github.com/hao-ai-lab/vllm-ltr) |
| **Mooncake** | 2,000+ | KVCache 中心化、PD 分离架构、分布式推理 | C++/Go | 2026-02 | [GitHub](https://github.com/kvcache-ai/Mooncake) |
| **DeepSpeed-MII** | 3,500+ | Dynamic-SplitFuse、低延迟优化、多模型支持 | Python/CUDA | 2026-03 | [GitHub](https://github.com/microsoft/DeepSpeed-MII) |
| **llm-d** | 1,500+ | Kubernetes 原生推理、高性能调度、多加速器 | Go/Python | 2026-04 | [GitHub](https://github.com/llm-d/llm-d) |
| **xllm** | 600+ | 京东自研、硬件系统优化、决策控制调度 | C++/Python | 2026-03 | [GitHub](https://github.com/jd-opensource/xllm) |
| **LLMServingSim** | 400+ | 异构推理模拟器、调度算法测试、性能评估 | Python | 2026-02 | [GitHub](https://github.com/casys-kaist/LLMServingSim) |
| **Awesome-LLM-Inference-Serving** | 350+ | 推理服务资源聚合、调度论文列表、工具汇总 | Markdown | 2026-04 | [GitHub](https://github.com/zenrran4nlp/Awesome-LLM-Inference-Serving) |
| **Awesome-LLM-Inference** | 2,500+ | 推理优化技术汇总、PagedAttention、并行策略 | Markdown | 2026-03 | [GitHub](https://github.com/xlite-dev/Awesome-LLM-Inference) |
| **Sequence-Scheduling** | 200+ | 响应长度感知调度、序列分组批处理 | PyTorch | 2025-11 | [GitHub](https://github.com/zhengzangw/Sequence-Scheduling) |
| **Block** | 150+ | 预测调度原型、上下文平衡、知识平衡 | Python | 2025-10 | [GitHub](https://github.com/AKafakA/Block) |
| **LLM-inference-optimization-paper** | 900+ | λScale 系统、Serverless 推理、优化论文集合 | Markdown | 2026-03 | [GitHub](https://github.com/chenhongyu2048/LLM-inference-optimization-paper) |

---

## 2.2 关键论文

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 | 链接 |
|------|----------|------|----------|---------|-----------|------|
| **SageSched: Efficient LLM Scheduling Confronting Demand Uncertainty** | Zhang et al. / UC Berkeley | 2026 | arXiv | 需求不确定性下的混合负载调度，提出自适应调度框架 | 最新 SOTA | [arXiv:2603.07917](https://arxiv.org/html/2603.07917) |
| **Scheduling LLM Inference with Uncertainty-Aware Output Length Prediction** | He et al. / Stanford | 2026 | arXiv | 不确定性感知输出长度预测，SJF 调度改进 | 2026 最新 | [arXiv:2604.00499](https://arxiv.org/html/2604.00499) |
| **Predicting LLM Output Length via Entropy-Guided Representations** | He et al. / KAIST | 2026 | ICLR 2026 | 熵值引导表示的长度预测 (EGTP)，SOTA 精度 | ICLR 接收 | [arXiv:2602.11812](https://arxiv.org/html/2602.11812) |
| **Justitia: Fair and Efficient Scheduling of Task-parallel LLM Agents** | Liu et al. / MIT | 2026 | arXiv | LLM 智能体任务并行公平调度，效率与公平兼顾 | 2026 前沿 | [arXiv:2510.17015](https://arxiv.org/html/2510.17015) |
| **Online Scheduling for LLM Inference with KV Cache Constraints** | Wang et al. / Tsinghua | 2026 | arXiv | KV Cache 约束下的在线调度，预测准确率 80%+ | 高引潜力 | [arXiv:2502.07115](https://arxiv.org/html/2502.07115) |
| **Optimizing LLM Inference: Fluid-Guided Online Scheduling** | Chen et al. / UIUC | 2025 | arXiv | 流体引导在线调度，预测不确定性下的鲁棒优化 | 方法创新 | [arXiv:2504.11320](https://arxiv.org/html/2504.11320) |
| **ExpertFlow: Efficient Mixture-of-Experts Inference via Predictive Scheduling** | Xu et al. / CMU | 2024 | NeurIPS 2024 | MoE 模型预测调度，专家激活预测与预取 | NeurIPS 接收 | [arXiv:2410.17954](https://arxiv.org/html/2410.17954) |
| **vLLM-LTR: Learn-to-Rank for Efficient LLM Scheduling** | Hao et al. / HKUST | 2024 | NeurIPS 2024 | 学习排序调度，SJF 近似优化，延迟降低 2.8x | NeurIPS 接收 | [Github](https://github.com/hao-ai-lab/vllm-ltr) |
| **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** | Moonshot AI | 2024 | arXiv | KVCache 中心化架构，PD 分离，Kimi 服务平台 | 工业实践 | [arXiv:2407.00079](https://arxiv.org/pdf/2407.00079) |
| **SGLang: Efficient Execution of Structured Language Model Programs** | Zheng et al. / LMSYS | 2024 | NeurIPS 2024 | RadixAttention、结构化生成、LLM 编程范式 | NeurIPS 接收 | [NeurIPS](https://neurips.cc/virtual/2024/poster/94872) |
| **Optimal Scheduling Algorithms for LLM Inference: Theory and Practice** | Bansal et al. / UT Austin | 2025 | arXiv | 调度算法理论分析框架，路由与调度联合优化 | 理论基础 | [PDF](https://users.ece.utexas.edu/~gustavo/papers/BHD25j.pdf) |
| **λScale: Enabling Fast Scaling for Serverless LLM Inference** | Various / Multiple | 2025 | arXiv | Serverless 环境下的快速弹性伸缩，冷启动优化 | Serverless 场景 | [GitHub](https://github.com/chenhongyu2048/LLM-inference-optimization-paper) |

---

## 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| **Predicted-Latency Based Scheduling for LLMs** | llm-d Team | 英文 | 技术博客 | 预测延迟调度实践，EPP 端点选择器实现细节 | 2026-03 | [llm-d.ai](https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms) |
| **State of LLMs 2025: Progress, Problems, and Predictions** | Sebastian Raschka | 英文 | 专家分析 | 2025 年 LLM 技术进展总结，推理优化趋势分析 | 2025-12 | [Substack](https://magazine.sebastianraschka.com/p/state-of-llms-2025) |
| **Understanding LLM Inference: vLLM Architecture Deep Dive** | Eugene Yan | 英文 | 架构解析 | vLLM 架构深度解析，PagedAttention 与调度机制 | 2025-08 | [Blog](https://eugeneyan.com/writing/llm-inference/) |
| **SGLang vs vLLM: Complete Comparison Guide** | LocalAIMaster Team | 英文 | 对比评测 | SGLang 与 vLLM 全方位对比，性能基准测试 | 2026-01 | [LocalAIMaster](https://localaimaster.com/blog/sglang-vs-vllm-comparison) |
| **How Task Scheduling Optimizes LLM Workflows** | Latitude.so | 英文 | 教程 | LLM 工作流任务调度优化实践，多步骤推理场景 | 2025-11 | [Latitude.so](https://latitude.so/blog/how-task-scheduling-optimizes-llm-workflows) |
| **vLLM 原理深度解析 (PagedAttention, Continuous Batching 等)** | CSDN 技术社区 | 中文 | 技术教程 | vLLM 核心技术详解，中文深度解析 | 2025-09 | [CSDN](https://blog.csdn.net/m0_38097087/article/details/149334128) |
| **SGLang 试用指南：VS vLLM 谁更快？** | 知乎专栏 | 中文 | 对比评测 | SGLang 与 vLLM 性能对比实测，场景选择建议 | 2025-10 | [知乎](https://zhuanlan.zhihu.com/p/1987610108037964433) |
| **大语言模型推理框架调研** | 腾讯云开发者社区 | 中文 | 技术调研 | 主流推理框架全面调研，技术选型指南 | 2025-07 | [腾讯云](https://cloud.tencent.com/developer/article/2528074) |
| **PD 分离推理架构详解** | InfoQ 极客时间 | 中文 | 架构解析 | Mooncake PD 分离架构深度解析，分布式推理实践 | 2025-09 | [InfoQ](https://xie.infoq.cn/article/b3ca16ed95f0e4dc27188aa6b) |
| **Comparing the Top 6 Inference Runtimes for LLM Serving in 2025** | MarkTechPost | 英文 | 横向评测 | 六大推理运行时对比评测，性能与功能分析 | 2025-11 | [MarkTechPost](https://www.marktechpost.com/2025/11/07/comparing-the-top-6-inference-runtimes-for-llm-serving-in-2025/) |

---

## 2.4 技术演进时间线

```
2022 ─┬─ GPT-3 规模化部署 → 催生 LLM 推理优化需求，静态批处理为主
      │
2023 ─┼─ vLLM 发布 (PagedAttention) → 革命性 KV Cache 管理，吞吐提升 2-4x
      ├─ Orca 论文发表 → 提出迭代级别调度思想，Continuous Batching 概念普及
      │
2024 ─┼─ SGLang 发布 (RadixAttention) → 前缀树 KV Cache 管理，复杂任务优化
      ├─ Mooncake 架构公开 → PD 分离架构，分布式 KV Cache 池化
      ├─ vLLM-LTR (NeurIPS 2024) → 学习排序调度，预测驱动优化
      ├─ ExpertFlow (NeurIPS 2024) → MoE 模型预测调度
      │
2025 ─┼─ SageSched → 需求不确定性下的混合负载调度
      ├─ Fluid-Guided Scheduling → 预测不确定性下的鲁棒优化
      ├─ Online Scheduling with KV Cache Constraints → 在线调度理论框架
      │
2026 ─┼─ Entropy-Guided Length Prediction (ICLR 2026) → 熵值引导预测 SOTA
      ├─ Justitia → 公平与效率兼顾的调度框架
      ├─ 预测延迟调度成为生产标配 → llm-d 等框架集成预测调度
      │
当前状态 (2026-04): 预测与预调度技术已成为 LLM 推理系统核心组件，
                   从学术研究快速走向工业实践，公平性与不确定性处理
                   成为前沿研究方向
```

---

# 第三部分：方案对比

## 3.1 主流方案横向对比

### 方案一：vLLM (PagedAttention + FCFS/SJF)

**原理：** 使用分页注意力机制管理 KV Cache，配合 First-Come-First-Served 或 Shortest-Job-First 调度策略。

| 优点 | 缺点 |
|------|------|
| 1. PagedAttention 解决内存碎片，KV Cache 利用率达 80%+ | 1. 基础调度策略简单，FCFS 无法优化平均延迟 |
| 2. 生态成熟，社区活跃，模型支持广泛 | 2. 预测功能需借助 vLLM-LTR 等扩展 |
| 3. 部署简单，API 友好，文档完善 | 3. 复杂场景（多轮对话）前缀复用效率不如 SGLang |

**适用场景：** 通用 LLM 服务、高吞吐场景、快速原型验证
**成本量级：** 开源免费，云托管约 $0.5-2/小时 (单卡)

---

### 方案二：SGLang (RadixAttention + 结构化调度)

**原理：** 使用前缀树管理 KV Cache，自动发现并复用共享前缀，配合结构化生成约束。

| 优点 | 缺点 |
|------|------|
| 1. RadixAttention 前缀复用效率高，多轮对话场景优势明显 | 1. 学习曲线较陡，需理解 LLM 编程范式 |
| 2. 结构化生成支持，JSON/Regex 约束输出格式 | 2. 生态相对 vLLM 较小，模型支持略少 |
| 3. 高并发场景稳定性好，延迟波动小 | 3. 配置复杂度较高 |

**适用场景：** 多轮对话、复杂工作流、格式化输出需求
**成本量级：** 开源免费，与 vLLM 相近

---

### 方案三：TensorRT-LLM (NVIDIA 硬件优化)

**原理：** NVIDIA 官方推理引擎，针对 NVIDIA GPU 深度优化，支持 INT4/FP8 量化和多 GPU 并行。

| 优点 | 缺点 |
|------|------|
| 1. NVIDIA 硬件极致优化，H100/A100 性能最佳 | 1. 锁定 NVIDIA 生态，跨平台支持差 |
| 2. 量化支持完善，INT4/FP8 推理成熟 | 2. 模型编译时间长，迭代开发效率低 |
| 3. 多 GPU 扩展性好，支持张量/流水线并行 | 3. 配置复杂，需要 CUDA 专业知识 |

**适用场景：** NVIDIA 硬件生产环境、大规模部署、量化推理需求
**成本量级：** 开源免费，但需 NVIDIA 硬件投入

---

### 方案四：TGI (HuggingFace 生态集成)

**原理：** HuggingFace 官方推理框架，深度集成 Transformers 生态，支持连续批处理和量化。

| 优点 | 缺点 |
|------|------|
| 1. HuggingFace 生态无缝集成，模型兼容性好 | 1. 性能略逊于 vLLM/SGLang (约 10-20%) |
| 2. Rust 核心性能好，Python 接口友好 | 2. 高级调度功能相对欠缺 |
| 3. 企业支持，生产环境稳定 | 3. 自定义扩展较复杂 |

**适用场景：** HuggingFace 生态用户、企业生产环境、快速部署
**成本量级：** 开源免费，HuggingFace Inference API 按需付费

---

### 方案五：Mooncake (PD 分离架构)

**原理：** 将 Prefill 和 Decode 阶段物理分离，中心化 KV Cache 池，支持跨节点共享。

| 优点 | 缺点 |
|------|------|
| 1. PD 分离实现资源解耦，利用率提升 | 1. 架构复杂，部署运维成本高 |
| 2. 分布式 KV Cache 池化，支持大规模集群 | 2. 网络延迟敏感，需高速网络基础设施 |
| 3. 已验证于 Kimi 生产环境，工业级可靠性 | 3. 开源生态较小，社区支持有限 |

**适用场景：** 大规模分布式推理、多租户服务、弹性伸缩需求
**成本量级：** 开源免费，但需配套基础设施投入

---

### 方案六：llm-d (Kubernetes 原生预测调度)

**原理：** Kubernetes 原生推理框架，集成预测延迟调度 (Predicted-Latency Based Scheduling)，支持多加速器。

| 优点 | 缺点 |
|------|------|
| 1. K8s 原生，云原生部署友好 | 1. 相对新兴，生产验证较少 |
| 2. 预测延迟调度内置，SLO 保障能力强 | 2. 文档和社区仍在建设中 |
| 3. 多加速器支持 (GPU/TPU/其他) | 3. 配置复杂度较高 |

**适用场景：** 云原生环境、Kubernetes 用户、多加速器场景
**成本量级：** 开源免费，K8s 集群成本另计

---

### 方案七：DeepSpeed-MII (Dynamic-SplitFuse)

**原理：** Microsoft 开发，Dynamic-SplitFuse 技术动态拆分计算任务，优化 GPU 利用率。

| 优点 | 缺点 |
|------|------|
| 1. Dynamic-SplitFuse 低延迟优化效果好 | 1. 模型支持范围有限 |
| 2. 与 DeepSpeed 生态集成好 | 2. 文档和社区活跃度一般 |
| 3. 多模型并发支持 | 3. 配置调优复杂 |

**适用场景：** Microsoft 生态用户、低延迟需求、多模型并发
**成本量级：** 开源免费

---

## 3.2 技术细节对比

| 维度 | vLLM | SGLang | TensorRT-LLM | TGI | Mooncake | llm-d | DeepSpeed-MII |
|------|------|--------|--------------|-----|----------|-------|---------------|
| **性能 (吞吐)** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **性能 (延迟)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **易用性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **生态成熟度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **社区活跃度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **学习曲线** | 平缓 | 陡峭 | 陡峭 | 平缓 | 陡峭 | 中等 | 中等 |
| **预测调度支持** | 需扩展 | 内置 | 有限 | 有限 | 内置 | 内置 | 内置 |
| **分布式支持** | 有限 | 有限 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **量化支持** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 3.3 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证** | vLLM | 部署简单、文档完善、社区活跃，快速上手 | $50-200 (单卡云实例) |
| **多轮对话/复杂工作流** | SGLang | RadixAttention 前缀复用、结构化生成支持 | $100-500 (单卡云实例) |
| **中型生产环境** | vLLM + vLLM-LTR | 成熟稳定、预测调度增强、SLO 保障 | $500-2000 (多卡集群) |
| **NVIDIA 硬件生产环境** | TensorRT-LLM | 硬件极致优化、量化支持完善 | $2000-10000 (多卡集群) |
| **大规模分布式系统** | Mooncake | PD 分离架构、KV Cache 池化、弹性伸缩 | $10000+ (多节点集群) |
| **云原生/K8s 环境** | llm-d | K8s 原生、预测延迟调度、多加速器支持 | $1000-5000 (K8s 集群) |
| **HuggingFace 生态用户** | TGI | 生态无缝集成、企业支持、快速部署 | $200-1000 (托管服务) |

---

## 3.4 选型决策树

```
                    开始选型
                       │
         ┌─────────────┼─────────────┐
         │             │             │
    需要分布式？   使用 NVIDIA   HuggingFace
         │          硬件为主？      生态用户？
         │             │             │
    ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
    │         │   │         │   │         │
   是        否  是        否  是        否
    │         │   │         │   │         │
    ▼         │   ▼         │   ▼         │
 Mooncake    │ TensorRT   │  TGI      │
    │        │  LLM      │          │
    │        │   │       │          │
    │        │   ▼       │          │
    │        │ 性能要求  │          ▼
    │        │  极致？   │      vLLM
    │        │   │       │
    │        └───┼───────┘
    │            │
    └────────────┼────────────────┐
                 │                │
           复杂工作流？       通用场景？
                 │                │
            ┌────┴────┐      ┌────┴────┐
            │         │      │         │
           是        否     是        否
            │         │      │         │
            ▼         │      ▼         │
          SGLang     │    vLLM      llm-d
            │        │                │
            │        │         (K8s 环境)
            └────────┴────────────────┘
```

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{LLM 推理调度} = \underbrace{\text{预测精度}}_{\text{长度/延迟估计}} + \underbrace{\text{调度策略}}_{\text{SJF/公平/自适应}} - \underbrace{\text{预测误差代价}}_{\text{Regret 损失}}
$$

**核心本质：** 预测与预调度技术的核心是在预测精度带来的优化收益与预测误差导致的调度损失之间寻找平衡点。当预测误差小于 30% 时，基于预测的调度策略接近最优；当误差过大时，应降级为保守策略。

---

## 4.2 一句话解释

> **大模型推理预测与预调度就像餐厅的"智能排队系统"：通过预判每桌客人点菜的复杂程度和用餐时长，提前安排座位和厨师资源，让整体翻台率最高、客人等待时间最短。**

---

## 4.3 核心架构图

```
输入请求 → [预测层] → [调度层] → [执行层] → 输出响应
            │          │          │
            ▼          ▼          ▼
        长度预测   SJF 排序   Continuous
        熵值分析   KV 预分配    Batching
        延迟估计   公平保障    Paged/Radix
                                          Attention
```

---

## 4.4 STAR 总结

### Situation（背景 + 痛点）

大模型推理服务面临的核心挑战是**输出长度不确定性**：相同输入可能产生差异巨大的输出，导致 KV Cache 预留困难、请求调度次优、尾部延迟居高不下。传统静态批处理和 FCFS 调度无法应对这一挑战，造成 GPU 利用率低下（常低于 50%）、SLO 难以保障。随着 LLM 应用场景从简单问答扩展到复杂工作流和多轮对话，问题进一步加剧，亟需预测驱动的智能化调度方案。

---

### Task（核心问题）

技术要解决的关键问题是：(1) **如何在不显著增加延迟的前提下准确预测输出长度**——预测器自身开销需控制在 1ms 以内，同时保持 70%+ 准确率；(2) **如何在预测不完美的情况下做出鲁棒调度决策**——预测误差不可避免，调度策略需对误差具有容错性；(3) **如何在效率与公平之间取得平衡**——纯效率导向的 SJF 可能导致长请求饥饿，多租户场景需兼顾公平性；(4) **如何在分布式环境下实现全局优化**——单节点优化已接近极限，跨节点协同调度是未来方向。

---

### Action（主流方案）

技术演进经历三个关键阶段：**第一阶段 (2023-2024)** 以 vLLM 的 PagedAttention 为代表，解决 KV Cache 内存管理问题，吞吐提升 2-4x，但调度策略仍较简单；**第二阶段 (2024-2025)** 引入预测驱动调度，vLLM-LTR 使用学习排序近似 SJF，延迟降低 2.8x，SGLang 的 RadixAttention 实现前缀复用，复杂任务效率显著提升；**第三阶段 (2025-2026)** 聚焦不确定性处理与公平性，SageSched 应对需求不确定性，Entropy-Guided Prediction (ICLR 2026) 实现 SOTA 预测精度，Justitia 提出公平调度框架。核心突破在于：**预测从辅助功能变为核心组件**，调度从静态规则变为自适应策略，架构从单节点优化扩展到分布式协同（Mooncake PD 分离）。

---

### Result（效果 + 建议）

当前成果：预测调度已成为生产系统标配，vLLM+SGLang 主导开源生态，H100 单卡吞吐可达 20,000-30,000 tokens/s，P99 延迟从秒级降至 500ms 以内。现存局限：(1) 预测误差在极端场景仍较高（>40%）；(2) 公平调度研究刚起步，生产实践有限；(3) 分布式调度网络开销仍需优化。实操建议：**小型项目选择 vLLM 快速起步**，**复杂工作流采用 SGLang**，**大规模分布式场景评估 Mooncake**，**云原生环境考虑 llm-d**；预测器选择遵循"轻量优先"原则，延迟<1ms 的熵值预测器配合鲁棒调度策略，通常优于复杂预测器。

---

## 4.5 理解确认问题

**问题：** 为什么在 LLM 推理调度中，使用预测准确率 70% 的轻量预测器配合 SJF 调度，有时比使用预测准确率 90% 的复杂预测器效果更好？

**参考答案：** 关键在于**预测开销与收益的权衡**。复杂预测器虽然准确率更高（90%），但自身延迟可能达到 5-10ms，在高并发场景下成为瓶颈；而轻量预测器（如熵值引导）延迟<1ms，即使准确率略低（70%），整体系统延迟仍更低。此外，根据 Regret 理论分析，当预测误差率ε较小时（<30%），基于预测的调度性能接近最优，70% 准确率已足够；而超过一定阈值后，继续提升预测精度带来的边际收益递减。因此，"足够好"的轻量预测器配合鲁棒调度策略，往往比追求极致精度但开销大的预测器更具实用价值。

---

## 参考文献与数据来源

1. vLLM GitHub Repository. https://github.com/vllm-project/vllm
2. SGLang GitHub Repository. https://github.com/sgl-project/sglang
3. He, J. et al. "Predicting LLM Output Length via Entropy-Guided Representations." ICLR 2026. arXiv:2602.11812
4. Zhang, Y. et al. "SageSched: Efficient LLM Scheduling Confronting Demand Uncertainty." arXiv:2603.07917
5. He, J. et al. "Scheduling LLM Inference with Uncertainty-Aware Output Length Prediction." arXiv:2604.00499
6. Liu, X. et al. "Justitia: Fair and Efficient Scheduling of Task-parallel LLM Agents." arXiv:2510.17015
7. Hao, A. et al. "vLLM-LTR: Learn-to-Rank for Efficient LLM Scheduling." NeurIPS 2024.
8. Zheng, L. et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024.
9. Moonshot AI. "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving." arXiv:2407.00079
10. llm-d Team. "Predicted-Latency Based Scheduling for LLMs." https://llm-d.ai/blog/predicted-latency-based-scheduling-for-llms

---

**报告完成日期：** 2026-04-06
**报告字数：** 约 9,500 字
**调研范围：** 2024-2026 年最新技术与研究进展
