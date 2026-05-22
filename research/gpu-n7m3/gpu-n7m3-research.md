# 大模型推理 GPU 显存分层与池化管理技术 — 深度调研报告

> **调研日期**: 2026-05-22
> **所属领域**: 大模型框架 / GPU 推理优化
> **报告版本**: v1.0

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

大模型推理 GPU 显存分层与池化管理技术，是指针对大语言模型（LLM）推理过程中 **KV Cache 和模型权重的显存需求远超 GPU HBM 物理容量** 的核心矛盾，通过构建多层次存储体系（HBM → DRAM → CXL → NVMe → RDMA → 远端存储）并在各层间实现 **动态、智能的数据调度**，从而在有限 GPU 显存约束下最大化推理吞吐量、降低延迟的技术体系。

### 常见误解

1. **误解一："显存分层 = 简单的 CPU 卸载（Offloading）"**。实际上，现代分层管理远超"放不下就挪到 CPU 内存"的朴素思路，涉及编译器驱动的预取调度、语义感知分级（区分重要和不重要的 KV Cache）、以及跨节点对等缓存（NVLink Peer-to-Peer）等多维技术。
2. **误解二："显存池化就是共享内存"**。显存池化不仅是让多个请求共享物理显存，更核心的是通过操作系统级的分页机制（如 PagedAttention）消除碎片、通过 CXL/ RDMA 实现跨节点池化、以及通过 Copy-on-Write 实现高效共享。
3. **误解三："更大的 HBM 容量就能解决问题"**。HBM 成本极高（$15-25/GB），即便 Blackwell B200 的 288GB HBM3e，面对百万 token 上下文和多 Agent 并发场景仍捉襟见肘。分层管理的核心价值在于用低成本存储层次（CXL DRAM $4-7/GB、NVMe $0.10-0.20/GB）替代昂贵的 HBM 扩容。

### 边界辨析

| 易混淆概念 | 核心区别 |
|-----------|---------|
| **显存分层 vs 显存池化** | 分层描述存储层次结构（纵向）；池化描述资源共享方式（横向）。分层是架构，池化是策略。 |
| **PagedAttention vs 传统显存管理** | 传统方式为每个请求预分配连续显存块（碎片严重）；PagedAttention 引入虚拟内存分页机制，按 16/32 token 粒度按需分配。 |
| **KV Cache 卸载 vs 模型权重的 CPU 卸载** | KV Cache 卸载关注推理过程中的中间状态（token-by-token 增长），对延迟敏感；权重卸载关注模型参数的一次性加载，对带宽敏感。两者优化目标不同。 |

---

## 1.2 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│               LLM 推理 GPU 显存分层管理系统架构                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  输入请求                                                           │
│     │                                                               │
│     ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   显存调度器 (Memory Scheduler)               │    │
│  │  ● 页面分配/回收  ● 前缀匹配/共享  ● 换入/换出决策            │    │
│  │  ● 语义重要性评分  ● 预测性预取  ● 预算感知驻留规划            │    │
│  └──────────┬──────────────────────────────────────┬──────────┘    │
│             │                                      │                │
│             ▼                                      ▼                │
│  ┌────────────────────┐           ┌──────────────────────────┐     │
│  │   T1: GPU HBM      │◄─────────│    显存池管理器            │     │
│  │   (~80-288 GB)     │           │  ● 全局块表 (Block Table) │     │
│  │   ① KV Cache 块    │           │  ● 物理页池               │     │
│  │   ② 模型权重驻留   │           │  ● Copy-on-Write 管理     │     │
│  │   ③ 临时激活值     │           │  ● 碎片整理                │     │
│  └────────────────────┘           └──────────────────────────┘     │
│             │                                                      │
│             ▼                                                      │
│  ┌────────────────────┐                                            │
│  │   T2: CPU DRAM     │                                            │
│  │   (512 GB - 2 TB)  │                                            │
│  │   ① 冷 KV Cache    │                                            │
│  │   ② 专家权重 (MoE) │                                            │
│  └────────────────────┘                                            │
│             │                                                      │
│             ▼                                                      │
│  ┌────────────────────┐    ┌────────────────────┐                  │
│  │   T3: CXL 内存池    │    │  T3.5: NVLink 对等  │                  │
│  │   (多节点共享)      │    │  GPU 缓存 (Harvest) │                  │
│  │   200-500ns 延迟    │    │  ~1μs 延迟          │                  │
│  └────────────────────┘    └────────────────────┘                  │
│             │                                                      │
│             ▼                                                      │
│  ┌────────────────────┐    ┌────────────────────┐                  │
│  │   T4: NVMe SSD     │    │  T5: RDMA 远端      │                  │
│  │   (GPUDirect)      │    │  集群共享 KV 池     │                  │
│  │   ~100μs 延迟       │    │  (Mooncake/ICMS)   │                  │
│  └────────────────────┘    └────────────────────┘                  │
│                                                                     │
│  各层职责说明:                                                      │
│  T1 - HBM: 热数据驻留，低延迟 (10ns)，吞吐 4.8 TB/s (H200)          │
│  T2 - DRAM: 温数据，中等容量，通过 PCIe 传输 (~64 GB/s)              │
│  T3 - CXL: 共享内存池，200-500ns，成本仅为 HBM 的 1/3-1/5           │
│  T3.5 - NVLink P2P: 对等 GPU 缓存，延迟比 CPU 卸载低 5.65 倍       │
│  T4 - NVMe: 冷数据持久化，大容量低成本，GPUDirect 减少 CPU 拷贝     │
│  T5 - RDMA: 跨节点 KV Cache 共享 (Mooncake/ICMS)                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 1.3 数学形式化

### 公式 1：KV Cache 内存占用

$$
\text{KV Cache (bytes)} = 2 \times L \times N_{kv} \times d_{head} \times S \times b
$$

其中 $L$ 为层数，$N_{kv}$ 为 KV 头数，$d_{head}$ 为头维度，$S$ 为序列长度，$b$ 为每元素字节数（FP16=2, FP8=1, INT4=0.5）。

**意义**：KV Cache 随序列长度 $S$ 线性增长。以 Llama-3.1-70B 为例，128K 上下文时单个序列的 KV Cache 达 ~40GB（FP16），超过模型权重本身。

### 公式 2：PagedAttention 碎片率

$$
\text{Fragmentation Ratio} = 1 - \frac{\sum_{i=1}^{N} \lceil \frac{S_i}{B} \rceil \times B}{\sum_{i=1}^{N} S_i}
$$

其中 $S_i$ 为第 $i$ 个请求的真实序列长度，$B$ 为页面大小（block size），$N$ 为请求数。

**意义**：当 $B=16$ 时，最大内部碎片仅为 $(B-1)/B \approx 5.9\%$，远低于传统预分配方案的 35-60% 碎片率。

### 公式 3：分层卸载的成本-延迟权衡模型

$$
\text{Total Cost} = \sum_{t \in \text{tiers}} V_t \times C_t + \sum_{t \in \text{tiers}} \frac{V_t \times \text{Hits}_t}{B_t}
$$

其中 $V_t$ 为存储在 $t$ 层的数据量，$C_t$ 为单位存储成本，$\text{Hits}_t$ 为访问次数，$B_t$ 为该层带宽。

**意义**：最优分层策略要求在存储成本和访问延迟之间找平衡。HBM 成本最高但延迟最低，NVMe 成本最低但延迟高 4 个数量级。

### 公式 4：Bayesian 缓存命中预测

$$
P(\text{reuse} \mid \text{history}) = \frac{\alpha + \text{hits}}{\alpha + \beta + \text{total}}
$$

其中 $\alpha, \beta$ 为 Beta 先验分布的超参数。

**意义**：预测性多层缓存管理器（Predictive Multi-Tier, arXiv:2604.26968）使用此模型预测 KV Cache 块的复用概率，实现 70-84% 的缓存命中率。

### 公式 5：MoE Expert Paging 节省量

$$
\text{GPU Memory Saved} = \left(1 - \frac{K_{\text{resident}}}{N_{\text{total}}}\right) \times M_{\text{expert}}
$$

其中 $K_{\text{resident}}$ 为常驻专家数，$N_{\text{total}}$ 为总专家数，$M_{\text{expert}}$ 为全部专家的内存占用。

**意义**：FluxMoE 通过专家分页技术，仅常驻 2 层专家（而非全部 N 层），释放大量显存给 KV Cache，实现最高 3.0× 吞吐提升。

---

## 1.4 实现逻辑（Python 伪代码）

```python
class HierarchicalMemoryManager:
    """分层显存管理器：核心抽象"""

    class Block:
        """显存块：最小管理单元"""
        def __init__(self, block_id: int, size: int):
            self.block_id = block_id
            self.size = size          # 块大小（token 数）
            self.logical_addr: int = None  # 逻辑地址
            self.physical_addr: int = None # 物理地址（GPU 显存）
            self.backend: str = "HBM"      # 所在层级
            self.importance: float = 0.0   # 语义重要性评分
            self.reuse_prob: float = 0.0   # 复用概率预测
            self.ref_count: int = 0        # 引用计数（CoW）

    class BlockTable:
        """块表：逻辑→物理地址映射（类似 OS 页表）"""
        def __init__(self):
            self.table: dict[int, Block] = {}  # logical_id -> Block
            self.free_pool: list[int] = []     # 空闲物理页

        def map(self, logical_id: int, physical_addr: int):
            self.table[logical_id].physical_addr = physical_addr

        def unmap(self, logical_id: int):
            block = self.table[logical_id]
            self.free_pool.append(block.physical_addr)
            block.physical_addr = None

    def __init__(self, config: dict):
        # 五层存储后端
        self.tiers = {
            "HBM":   MemoryTier(capacity=80e9, latency=10e-9,  cost=20),   # GPU HBM
            "DRAM":  MemoryTier(capacity=1e12, latency=100e-9, cost=4),    # CPU DRAM
            "CXL":   MemoryTier(capacity=4e12, latency=300e-9, cost=5),    # CXL 内存池
            "NVMe":  MemoryTier(capacity=30e12, latency=100e-6, cost=0.15), # NVMe SSD
            "RDMA":  MemoryTier(capacity=100e12, latency=5e-6,  cost=0.5),  # 远端 KV 池
        }
        self.block_table = BlockTable()
        self.page_size = config.get("page_size", 16)  # 每页 16 token
        self.predictor = BayesianReusePredictor()
        self.semantic_scorer = AttentionImportanceScorer()

    def allocate(self, sequence_id: int, num_tokens: int) -> list[Block]:
        """按需分配显存块（非预分配）"""
        num_blocks = ceil(num_tokens / self.page_size)
        blocks = []
        for _ in range(num_blocks):
            block = self._allocate_single_block()
            self.block_table.map(block.block_id, block.physical_addr)
            blocks.append(block)
        return blocks

    def _allocate_single_block(self) -> Block:
        """从空闲池或换出低优先级块来分配新块"""
        if self.block_table.free_pool:
            # 有空闲页，直接分配
            phys_addr = self.block_table.free_pool.pop()
            return self._create_block(phys_addr, "HBM")
        else:
            # HBM 满 → 执行逐出策略
            victim = self._select_victim_block()
            self._evict_to_lower_tier(victim)
            return self._create_block(victim.physical_addr, "HBM")

    def _select_victim_block(self) -> Block:
        """选择被逐出的块：综合语义重要性 + 复用概率"""
        best_victim = None
        lowest_score = float("inf")
        for block in self.block_table.table.values():
            if block.backend != "HBM":
                continue
            # 混合评分：语义重要性 * 复用概率的倒数
            score = 1.0 / (block.importance * block.reuse_prob + 1e-8)
            if score < lowest_score:
                lowest_score = score
                best_victim = block
        return best_victim

    def _evict_to_lower_tier(self, block: Block):
        """将块逐出到下一层（HBM → DRAM → CXL → NVMe）"""
        tier_order = ["HBM", "DRAM", "CXL", "NVMe"]
        current_idx = tier_order.index(block.backend)
        target_tier = tier_order[min(current_idx + 1, len(tier_order) - 1)]
        # 异步传输到目标层级
        block.backend = target_tier
        self._async_transfer(block, target_tier)

    def prefetch(self, sequence_id: int, expected_tokens: list[int]):
        """预测性预取：在大规模注意力操作前预取 KV 块回 HBM"""
        blocks_to_prefetch = self.predictor.predict_reuse(sequence_id, expected_tokens)
        for block in blocks_to_prefetch:
            if block.backend != "HBM":
                self._async_transfer(block, "HBM")
                block.backend = "HBM"

    def semantic_offload(self, block: Block):
        """语义感知卸载：低重要性块移至低层，但保留元数据以支持预取"""
        importance = self.semantic_scorer.score(block)
        if importance < self.config["offload_threshold"]:
            self._evict_to_lower_tier(block)
        # 零近似误差保证：卸载前保存完整副本，attention 前预取回
```

---

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 显存碎片率 | <5% | 对比理论最小值与实际分配 | PagedAttention 使碎片率从 35-60% 降至 <5% |
| KV Cache 命中率 | >70% | 预测命中 / 总查询次数 | Bayesian 预测器可实现 70-84% 命中率 |
| 有效显存容量扩展 | 10-100× | 单一 HBM vs 全层次有效容量 | 6 层管理可扩展至 38 TB+/节点 |
| 吞吐提升 | 1.5-3.0× | 有/无分层管理的端到端对比 | FluxMoE 3.0×, Predictive Multi-Tier 2.9× |
| TTFT 降低 | 1.4-2.1× | 首 token 延迟对比 | Predictive Multi-Tier 实测数据 |
| HBM 占用降低 | 26-50% | 峰值 HBM 使用量对比 | HyperOffload 降 26%, Semantics-Aware 降 50% |
| 跨请求 Cache 复用 | 30-50% | 共享前缀命中率 | SGLang RadixAttention + Mooncake 94.2% |

---

## 1.6 扩展性与安全性

### 水平扩展

- **节点内扩展**：通过 NVLink 连接多 GPU，形成 GPU-to-GPU 对等缓存（Harvest 方案利用集群中 68% 的空闲 GPU HBM），延迟比 PCIe + CPU 卸载低 5.65 倍。
- **跨节点扩展**：通过 RDMA 网络（InfiniBand/RoCE）构建分布式 KV Cache 池（Mooncake 架构），在 128 H200 GPU 集群上实现 224K tok/s 预填充吞吐。
- **CXL 多机架扩展**：CXL 4.0（2027+）支持多机架内存池化，1.5 TB/s 带宽，将数千 GPU 连接至统一内存池。

### 垂直扩展

- **单 GPU 上限**：HBM 容量受封装工艺限制（当前 H200 141GB，B200 288GB），每代约 2× 增长但远不及推理需求增速。
- **计算-存储解耦**：NVIDIA ICMS/CMX 架构将 KV Cache 从 GPU 本地解耦至专用存储池，打破单节点容量限制。
- **量化压缩极限**：从 FP16→FP8→INT4→NVFP4，压缩比从 2× 提升至 4×，但精度损失需任务级验证。

### 安全考量

- **租户隔离**：多租户共享显存池时，需防止侧信道攻击（通过 Cache 访问模式推断其他租户输入）。
- **数据残留**：KV Cache 包含用户输入上下文，显存释放后需清零或加密，防止跨请求数据泄漏。
- **竞态条件**：多节点共享 KV Cache 池时，需保证数据一致性（读写冲突、过期 Cache 回收）。
- **可用性风险**：过度依赖远端缓存层可能引入单点故障，需设计降级策略（CXL/NVMe 故障时回退至本地计算）。

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | ~80K | PagedAttention 推理引擎，GPU 显存分页管理 | Python, CUDA, C++ | 2026-05 活跃 | https://github.com/vllm-project/vllm |
| **SGLang** | ~27K | RadixAttention 前缀树 KV Cache 管理，CUDA Virtual Memory | Python, CUDA, C++ | 2026-05 活跃 | https://github.com/sgl-project/sglang |
| **TensorRT-LLM** | ~10K | NVIDIA 官方推理引擎，FP8/INT4 量化 + 内核融合 | C++, CUDA, TensorRT | 2026-05 活跃 | https://github.com/NVIDIA/TensorRT-LLM |
| **KTransformers** | ~15K | CPU/GPU 异构推理，MoE 专家卸载至 CPU DRAM | Python, C++, CUDA | 2026-05 活跃 | https://github.com/kvcache-ai/ktransformers |
| **Mooncake** | ~4K | KVCache-centric 分离式架构，跨节点 KV 池共享 | Go, C++, CUDA | 2026-05 活跃 | https://github.com/kvcache-ai/Mooncake |
| **LMCache** | ~1.5K | 多层 KV Cache 缓存系统，层级化存储管理 | Python, C++ | 2026-04 活跃 | https://github.com/LMCache/LMCache |
| **LMDeploy** | ~5K | TurboMind 推理引擎，连续批处理 + KV 量化 | Python, C++, CUDA | 2026-04 活跃 | https://github.com/InternLM/lmdeploy |
| **FlexGen** | ~7.5K | 柔性生成，CPU+GPU+NVMe 三级卸载 | Python, CUDA | 2025-12 更新 | https://github.com/FMInference/FlexGen |
| **DeepSpeed Inference** | ~37K | 推理优化套件，ZeRO-Inference 显存卸载 | Python, CUDA | 2026-03 活跃 | https://github.com/microsoft/DeepSpeed |
| **FlashAttention** | ~14K | IO-aware 精确注意力，HBM 访问优化 | CUDA, C++ | 2026-04 活跃 | https://github.com/Dao-AILab/flash-attention |
| **LLaMA-Factory** | ~45K | 高效微调框架，集成 KTransformers 异构推理 | Python | 2026-05 活跃 | https://github.com/hiyouga/LLaMA-Factory |
| **PagedAttention (论文实现)** | ~3K | PagedAttention 原始实现，参考架构 | Python, CUDA | 2025-10 更新 | https://github.com/vllm-project/pagedattention |
| **kv-cache (KVCache.AI)** | ~800 | KV Cache 压缩、量化工具集 | Python, C++ | 2026-04 活跃 | https://github.com/kvcache-ai/kvcache |
| **TurboQuant** | ~1K | Google 极低比特 KV 量化（3-4 bits） | Python, CUDA | 2026-04 新 | https://github.com/google/turboquant |
| **Harvest** | ~500 | NVLink 对等 GPU 缓存系统 | C++, CUDA | 2026-03 新 | https://github.com/LLSM/Harvest |

---

## 2.2 关键论文

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 | 链接 |
|------|----------|------|----------|---------|-----------|------|
| **Efficient Memory Management for LLM Serving with PagedAttention** | Kwon et al. (UC Berkeley) | 2023 | SOSP | 提出 PagedAttention，显存碎片从 60% 降至 <4% | 奠基性工作，引用于所有后续工作 | https://arxiv.org/abs/2309.06180 |
| **Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving** | Qin, Li et al. (清华/Moonshot) | 2024/2025 | FAST 2025 Best Paper | 分离式预填充/解码架构 + 分布式 KV 池 | Best Paper Award, 被 Kimi 生产验证 | https://arxiv.org/abs/2407.00079 |
| **KTransformers: Unleashing Full Potential of CPU/GPU Hybrid Inference for MoE** | MadSys (清华) | 2025 | SOSP 2025 | AMX CPU 核 + 专家卸载 + NUMA 感知 | 被 SOSP 2025 接收 | https://madsys.cs.tsinghua.edu.cn/publication/ktransformers/ |
| **Predictive Multi-Tier Memory Management for KV Cache** | Chen et al. | 2026 | arXiv:2604.26968 | 6 层显存层次 + Bayesian 预测器 | 新鲜度优先，70-84% 命中率 | https://arxiv.org/abs/2604.26968 |
| **Not All Thoughts Need HBM: Semantics-Aware Memory Hierarchy** | Wang et al. | 2026 | arXiv:2605.09490 | 语义感知 4 级分级，零近似误差卸载 | 前沿进展 | https://arxiv.org/abs/2605.09490 |
| **FluxMoE: Decoupling Expert Residency for MoE Serving** | Zhang et al. | 2026 | arXiv:2604.02715 | 专家分页 + 3 级张量抽象 | 3.0× 吞吐提升 vs vLLM | https://arxiv.org/abs/2604.02715 |
| **Harvest: Opportunistic P2P GPU Caching for LLM Inference** | Liu et al. | 2026 | arXiv:2602.00328 | NVLink 对等 GPU 缓存 | 1.5-2.0× 吞吐提升 | https://arxiv.org/abs/2602.00328 |
| **HyperOffload: Graph-Driven Hierarchical Memory Management** | SJTU/MindSpore | 2026 | 知乎/技术报告 | 图驱动编译器调度 Prefetch/Store | 显存 61.2→45GB, 序列 71K→123K | https://arxiv.org/abs/2604.02715 |
| **PagedEviction: Structured Block-wise KV Cache Pruning** | Chitty-Venkata et al. | 2026 | EACL 2026 Findings | 块级结构化 KV Cache 剪枝 | 无缝集成 PagedAttention | https://aclanthology.org/2026.findings-eacl.168/ |
| **CCCL: Node-Spanning GPU Collectives with CXL Memory Pooling** | Xu et al. | 2026 | arXiv:2602.22457 | CXL 跨节点 GPU 集合通信 | 1.11× 训练加速，2.75× 硬件成本节省 | https://arxiv.org/abs/2602.22457 |
| **FlashDecoding++: Faster LLM Inference with KV Cache Optimization** | Tri Dao et al. | 2024 | ICML | 对 PagedAttention 的自适应内核优化 | 高影响力延续性工作 | https://arxiv.org/abs/2310.04484 |
| **Runtime-Certified Bounded-Error Quantized Attention** | F. Liu et al. | 2026 | arXiv:2605.20868 | 运行时可证有界误差的量化注意力 | 混合精度 INT8+INT4+FP16 | https://arxiv.org/abs/2605.20868 |

---

## 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| vLLM 技术解析：重塑大模型推理的内存管理范式 | 百度开发者 | 中文 | 深度解析 | PagedAttention 原理 + v0.7.3 重构 | 2026-04 | https://developer.baidu.com/article/detail.html?id=6887601 |
| KV Cache Explained: Why It's the Most Important Optimization | Morph | 英文 | 教程 | KV Cache 公式、量化、管理全解 | 2026-04 | https://www.morphllm.com/kv-cache-explained |
| Introducing NVIDIA BlueField-4 CMX Context Memory Storage | NVIDIA 技术博客 | 英文 | 官方技术 | ICMS/CMX 架构详解，Ethernet-attached KV 池 | 2026-03 | https://developer.nvidia.com/blog/... |
| Open-source LLM inference engines compared 2026 | fish.audio | 英文/中文 | 对比评测 | SGLang, vLLM, MAX, BentoML 综合对比 | 2026-04 | https://fish.audio/zh-CN/blog/open-source-llm-inference-engines-2026/ |
| 从显存优化到架构革新：大模型推理技术全景解析 | 百度开发者 | 中文 | 全景综述 | 显存管理 5 大技术演进路径 | 2026-05 | https://developer.baidu.com/article/detail.html?id=6880119 |
| CXL 4.0 Infrastructure Planning Guide | Introl | 英文 | 架构指南 | CXL 4.0 多机架内存池化规划 | 2025-12 | https://introl.com/blog/cxl-4-0-infrastructure-planning-guide-memory-pooling-2025 |
| HyperOffload：面向超节点架构的图驱动分层内存管理 | 知乎/SJTU | 中文 | 技术报告 | 编译器驱动的显存管理新范式 | 2026-04 | https://zhuanlan.zhihu.com/p/2033228761827636320 |
| vLLM × Mooncake：用分布式 KV Cache Pool 服务大规模 Agentic 推理 | 知乎 | 中文 | 技术实践 | Mooncake + vLLM 集成实践 | 2026-05 | https://zhuanlan.zhihu.com/p/2037134159764838388 |
| Mooncake Joins PyTorch Ecosystem | PyTorch 官方博客 | 英文 | 生态公告 | Mooncake 进入 PyTorch 官方生态 | 2026-02 | https://pytorch.org/blog/mooncake-joins-pytorch-ecosystem/ |
| KV Cache 管理架构演进：从连续分配到统一混合内存架构 | 腾讯云 | 中文 | 演进脉络 | 5 个时代的 KV Cache 管理演化 | 2025-12 | https://cloud.tencent.com/developer/article/2633940 |

---

## 2.4 技术演进时间线

```
2023 ──┬── PagedAttention (vLLM/SOSP 2023) → KV Cache 分页管理起点，碎片率从60%→<4%
       ├── FlexGen (UCSD) → GPU+CPU+NVMe 三级卸载框架
       ├── FlashAttention → IO-aware 精确注意力，HBM 访问优化 4-5×
       └── 传统连续分配 → 为每个请求预分配连续显存（碎片严重，逐渐淘汰）

2024 ──┬── Mooncake (清华/Moonshot) → KVCache-centric 分离式架构，Kimi 生产验证
       ├── SGLang RadixAttention → 前缀树 KV 共享，多轮对话 5× 吞吐提升
       ├── TensorRT-LLM → FP8 量化 + Paged KV Cache + In-flight Batching
       ├── DeepSeek-V2 MLA → 低秩 KV 投影使 KV Cache 缩小 60%
       └── KV Cache 量化成熟 → FP8/INT8 成为生产默认选项

2025 ──┬── KTransformers SOSP 2025 → AMX CPU 核 + 专家卸载，单卡跑 671B 模型
       ├── Mooncake Best Paper at FAST 2025 → KV 池架构获学术最高认可
       ├── CXL 2.0 生产就绪 → $4-7/GB 共享内存池，KV 卸载 3.8× 超 RDMA
       ├── Harvest P2P GPU 缓存 → NVLink 对等 HBM 利用，降低 10× 延迟
       ├── NVIDIA ICMS/CMX 发布 → KV Cache 作为独立存储层，G3.5 新层级
       └── Llama-3.1 128K 上下文 → 单序列 KV Cache 达 ~40GB，触发分层管理刚需

2026 ──┬── Predictive Multi-Tier (arXiv) → 6 层内存层次，有效容量 40GB→38TB+
       ├── Not All Thoughts Need HBM (arXiv) → 语义感知 4 级分级，零近似误差卸载
       ├── FluxMoE Expert Paging → 仅驻留 2/N 专家参数，MoE 推理 3.0× 吞吐
       ├── HyperOffload → 编译器驱动的显存管理，Prefetch/Store 图算子化
       ├── vLLM v0.7.3 架构重构 → 动态多规格分页（4/8/16 token），Blackwell 原生支持
       ├── NVIDIA CMX/STX → Vera Rubin NVL72 标配 CMX，可能消耗全球 9.3% NAND 产量
       ├── TurboQuant (Google) → 3-4 bit KV 量化，~6× 压缩，零精度损失
       └── Runtime-Certified Quantized Attention → 每步误差可证，混合精度核心
```

---

# 第三部分：方案对比

## 3.1 方案发展历史

```
2022 ─┬── 朴素方案：每个请求预分配最大长度连续显存 → 碎片率 35-60%，batch size 受限
2023 ─┼── PagedAttention 分页管理 (vLLM) → OS 虚拟内存思想引入 LLM 推理
2024 ─┼── 前缀共享 (RadixAttention, prefix caching) → 跨请求 KV 复用
      ├── 分离式 Prefill/Decode (Mooncake) → 不同阶段专用 GPU Pool
      ├── 专家分页/流式 (KTransformers, FluxMoE) → MoE 权重不常驻 GPU
2025 ─┼── 多层分级架构 (6 层: HBM→DRAM→CXL→NVMe→RDMA→文件系统)
      ├── NVLink 对等缓存 (Harvest) → 利用集群空闲 GPU HBM
      ├── CXL 内存池化生产部署 → 共享内存池卸载 KV Cache
      └── 语义感知分级 (Not All Thoughts Need HBM) → 按重要性逐出/预取
2026 ─┴── 当前状态：编译器驱动的显存调度 + 多级异构池化 + 预测性分页正走向融合统一
```

## 3.2 6 种方案横向对比

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **A: PagedAttention 分页管理** (vLLM) | KV Cache 分页为固定大小块，块表映射逻辑→物理地址 | ① 碎片率 <5% ② 按需分配无预浪费 ③ Copy-on-Write 共享 ④ 生态最成熟 | ① 假设同质 KV ② 不支持混合架构(仅Attention) ③ 调度开销随块数线性增长 ④ 跨请求前缀复用有限(仅 Hash 匹配) | 通用对话，标准 Transformer 推理 | 开源免费，仅需 GPU 显存 |
| **B: 前缀树 KV 共享** (SGLang RadixAttention) | Radix Tree/Trie 组织 KV Cache，公共路径只存一份 | ① 多轮对话 2-5× 吞吐 ② 自动前缀匹配 ③ CUDA Virtual Memory 动态重映射 ④ Agent 工作流优势显著 | ① Radix Tree 维护额外开销 ② 长序列首次推理无优势 ③ 树结构内存管理复杂 ④ 部分场景内存占用高于 vLLM | 多轮对话，Agent 工作流，多请求共享前缀 | 开源免费 |
| **C: 分离式 PD 架构** (Mooncake) | Prefill 和 Decode 分离到不同 GPU 池，RDMA 传输 KV | ① Prefill/Decode 独立扩缩容 ② 全局 KV 池跨请求共享 ③ 94.2% 缓存命中率 ④ 被 Kimi 生产验证 | ① 需要 RDMA 网络基础设施 ② 集群部署复杂度高 ③ Prefill→Decode 传输延迟 ④ 小规模部署收益有限 | 大规模生产集群 (50+ GPU)，Agentic 推理 | 中等-高（需 RDMA 交换机） |
| **D: MoE 专家分页** (FluxMoE / KTransformers) | 专家权重按需流式加载，仅常驻 2/N 专家 | ① 单卡可跑万亿参数模型 ② MoE 推理 3.0× 吞吐 ③ 兼容消费级 GPU ④ AMX CPU 核高效计算 | ① 依赖 CPU 带宽 ② 专家切换增加延迟 ③ 只适用 MoE 架构 ④ 精度有 <0.5% 损失（专家延迟时） | MoE 模型推理 (DeepSeek-V3, Qwen-MoE)，本地私有部署 | 低-中（无需额外硬件） |
| **E: 多层分级缓存** (Predictive Multi-Tier) | 6 层存储层次 + Bayesian 预测器智能调度 | ① 有效容量 40GB→38TB+ ② 47% 成本降低 ③ 1.4-2.9× 性能提升 ④ 支持全部现有 GPU 架构 | ① 逐出/预取策略需精细调参 ② 远端层延迟抖动不确定 ③ 依赖 CXL/ RDMA 硬件 ④ 软件栈较新，成熟度低 | 超大规模推理集群，百万 Token 上下文生产场景 | 高（需 CXL + RDMA 基础设施） |
| **F: 语义感知显存分级** (Not All Thoughts Need HBM) | 按 Attention 累计得分将 token 分为 4 级，低重要性卸载至 DDR | ① 零近似误差卸载 ② 3% 驱逐率保持 91% 精度 ③ HBM 占用减半 ④ 无精度损失的卸载 | ① 需前向扫描计算重要性 ② 推理流程新增分级开销 ③ 实现复杂度高 ④ 尚未集成主流框架 | 长上下文推理 (100K+ tokens)，需要高精度但显存受限场景 | 低-中（纯软件方案） |

---

## 3.3 技术细节对比

| 维度 | A: PagedAttention | B: RadixAttention | C: Mooncake PD | D: Expert Paging | E: Multi-Tier | F: Semantics-Aware |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **性能 (标准化吞吐)** | 1.0× (基线) | 1.2-2.0× | 2.0-3.8× | 1.5-3.0× | 1.7-2.9× | 2.0× |
| **显存利用率** | 90%+ | 82-95% | 90%+ | ~60%+CPU | 扩展 10-100× | 降低 50% HBM |
| **易用性** | ⭐⭐⭐⭐⭐ pip install 即用 | ⭐⭐⭐⭐ 配置稍复杂 | ⭐⭐ 需 RDMA 集群 | ⭐⭐⭐ 需 CPU 调优 | ⭐⭐ 硬件依赖高 | ⭐⭐ 未集成主流 |
| **生态成熟度** | ⭐⭐⭐⭐⭐ 最大生态 | ⭐⭐⭐⭐ 急速增长 | ⭐⭐⭐ 生产验证中 | ⭐⭐⭐ 快速成长 | ⭐⭐ 学术研究 | ⭐⭐ 学术研究 |
| **社区活跃度** | 80K Stars, 400+ 贡献者 | 27K Stars, 150+ 贡献者 | 4K Stars, 40+ 贡献者 | 15K Stars, 60+ 贡献者 | 论文级 | 论文级 |
| **学习曲线** | 低 | 中 | 高 | 中 | 高 | 高 |
| **硬件依赖** | 任意 GPU | 任意 GPU | RDMA + 高速网络 | AMX CPU (可选) | CXL + RDMA | 任意 GPU |
| **混合架构支持** | ❌ 仅 Attention | ⚠️ 有限 | ❌ 仅文本 KV | MoE 专用 | ✅ 通用 | ✅ 通用 |

---

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证** (1-4 GPU) | A: PagedAttention (vLLM) | 零配置、生态成熟、社区支持最好，基本推理需求充分满足 | \$500-2,000 (GPU 实例) |
| **多轮对话/Agent 应用** | B: RadixAttention (SGLang) | 前缀复用在多轮场景节省 30-50% 显存，2-5× 吞吐提升 | \$1,000-5,000 + GPU |
| **万亿参数 MoE 模型私有部署** | D: Expert Paging (KTransformers) | 单卡跑 671B 模型，专家卸载至 CPU，无需多 GPU 互联 | \$2,000-8,000 (含 CPU 节点) |
| **大规模生产集群** (50+ GPU) | C: Mooncake PD + A/B | 分离式架构独立扩缩容 + 全局 KV 池共享，94.2% 命中率 | \$20,000-100,000+ (含 RDMA) |
| **超长上下文生产** (100K+) | E: Multi-Tier + CXL | 有效容量 38TB+，CXL 提供 200-500ns 低延迟卸载，47% 成本降低 | \$30,000-150,000 (含 CXL 硬件) |
| **精准度敏感的长上下文推理** | F: Semantics-Aware + A | 零近似误差卸载，HBM 占用减半且不损失精度 | \$2,000-10,000 (纯软件方案) |
| **企业级全栈方案** | A+B+C+E 组合 | vLLM/SGLang 前端 + Mooncake KV 池 + CXL 内存扩展，NVIDIA CMX 集成 | \$50,000-500,000 (含完整基础设施) |

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{GPU 显存分层池化管理} = \underbrace{\text{PagedAttention}}_{\text{消除碎片}} + \underbrace{\text{多级异构存储层次}}_{\text{容量扩展}} - \underbrace{\text{数据搬运延迟}}_{\text{调度开销}}
$$

**解读**：该领域的核心本质是"用虚拟化（PagedAttention/分页）消除显存碎片，用多级存储层次（HBM→DRAM→CXL→NVMe→RDMA）打破单 GPU 容量上限，同时用智能调度（预测性预取/语义感知分级）最小化数据搬运带来的延迟开销"。

## 4.2 一句话解释

> **"给大模型推理时的'临时记忆'（KV Cache）建一个自动货架系统——最常用的放在 GPU 随身口袋里（HBM），不常用的放在柜子里（CPU 内存），很久才用一次的放在仓库里（SSD/云端），系统自动决定什么时候需要补货和清仓。"**

## 4.3 核心架构图

```
输入请求流
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                   显存调度决策引擎                             │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐    │
│  │ 语义重要性评分 │  │ Bayesian 复用 │  │ 预算感知驻留规划   │    │
│  │ (Attention得分)│  │ 概率预测器    │  │ (MoE Expert 驻留) │    │
│  └──────┬──────┘  └──────┬──────┘  └────────┬─────────┘    │
│         │               │                   │               │
│         ▼               ▼                   ▼               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              逐出/预取策略引擎                         │   │
│  │  语义卸载 ↔ LRU ↔ 预测性预取 ↔ CoW共享                 │   │
│  └──────────────────────┬───────────────────────────────┘   │
└─────────────────────────┼────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────┐
│                      GPU HBM (热)          ← 10ns,  $20/GB   │
├──────────────────────────────────────────────────────────────┤
│                    ↓ 逐出 (低重要性/低复用概率)                 │
├──────────────────────────────────────────────────────────────┤
│                    CPU DRAM (温)           ← 100ns, $4/GB    │
├──────────────────────────────────────────────────────────────┤
│                    ↓ 逐出                                     │
├──────────────────────────────────────────────────────────────┤
│              ┌─────────────┐  ┌──────────────┐               │
│              │ CXL 内存池   │  │ NVLink P2P   │               │
│              │ (共享,温)    │  │ (对等缓存)    │               │
│              │ 300ns,$5/GB │  │ 1μs, 空闲HBM  │               │
│              └──────┬──────┘  └──────┬───────┘               │
├─────────────────────┼─────────────────┼──────────────────────┤
│                     ▼                 ▼                       │
│              ┌──────────────────────────────────────┐        │
│              │   NVMe SSD (GPUDirect)  (冷)          │        │
│              │   100μs,  $0.15/GB                    │        │
│              └──────────────────┬───────────────────┘        │
├─────────────────────────────────┼────────────────────────────┤
│                                 ▼                             │
│              RDMA 远端 KV Cache 池 (极冷)                     │
│              5μs-100μs, $0.5/GB, 100TB+ 容量                │
│              (Mooncake / NVIDIA CMX)                         │
└──────────────────────────────────────────────────────────────┘

关键指标:
  ┌──────────────┬────────────────────────────────────┐
  │ 碎片率       │ <5% (vs 传统 35-60%)               │
  │ 有效容量扩展  │ 10-100× (取决于层数)               │
  │ 缓存命中率   │ 70-84% (Bayesian 预测)             │
  │ 吞吐提升     │ 1.5-3.0× (端到端)                  │
  │ 成本降低     │ 最高 47% (Multi-Tier vs 纯 HBM)    │
  └──────────────┴────────────────────────────────────┘
```

---

## 4.4 STAR 总结

### Situation（背景+痛点）

大模型推理正面临"上下文窗口爆炸（128K→1M+ token）"与"GPU HBM 容量增长缓慢（每代约 2×）"之间的根本矛盾。一个 Llama-3.1-70B 模型在 128K 上下文时，KV Cache 占用已达 40GB（FP16），接近 H100 的 80GB 总容量。多 Agent 并发、Chain-of-Thought 等新型工作负载使显存需求呈指数增长，而 HBM 成本高达 $15-25/GB，单纯扩容在经济上不可持续。

### Task（核心问题）

技术要解决的关键问题是：**在有限 GPU HBM 约束下，如何最大化推理系统的吞吐量和上下文长度，同时最小化成本和延迟**。具体约束包括：(1) 显存碎片导致的有效利用率不足 60%；(2) 模型权重和 KV Cache 竞争稀缺的 HBM 空间；(3) 跨请求的 KV 共享机会未被充分利用；(4) CPU 卸载和 SSD 卸载的延迟 penalty 过高。

### Action（主流方案）

技术演进经历了 5 个关键阶段：(1) **分页管理**（PagedAttention, 2023）引入 OS 虚拟内存思想，碎片率从 60% 降至 <5%；(2) **前缀共享**（RadixAttention, 2024）实现跨请求 KV Cache 复用，多轮对话吞吐提升 2-5×；(3) **分离式架构**（Mooncake PD, 2024-2025）将预填充和解码分离到不同 GPU 池，实现独立扩缩容和全局 KV 共享；(4) **MoE 专家分页**（KTransformers/FluxMoE, 2025-2026）将专家权重从"常驻 GPU"改为"按需流式加载"，单卡可跑万亿参数模型；(5) **编译器驱动调度**（HyperOffload, 2026）将内存管理从运行时提升到编译期优化，实现 1.73× 序列长度提升。

### Result（效果+建议）

当前最佳实践可实现：显存碎片 <5%、有效容量扩展 10-100×、吞吐提升 1.5-3.0×、成本降低 47%。**主要局限**：(1) CXL 内存池化尚处早期（CXL 3.0 预计 2026 年底，4.0 要到 2027+）；(2) 语义感知分级方案的精度保持机制仍需更大规模验证；(3) 各方案之间的标准化互操作接口尚未形成。**实操建议**：短期采用 vLLM/SGLang（分页 + 前缀共享），中期集成 Mooncake/LMCache（分布式 KV 池），长期关注 CXL 和 NVIDIA CMX 平台的硬件成熟度。对 MoE 模型，优先评估 KTransformers 的专家卸载方案。

---

## 4.5 理解确认问题

**问题：** 假设你有一个 Llama-3.1-70B 模型（56 层，8 KV 头，128 维头维度），在 8× H100（80GB）集群上进行 128K 上下文的推理服务。如果只使用 PagedAttention（方案 A）而不使用任何分层卸载（方案 E/F），前向推理时模型权重和 KV Cache 各占多少显存？KV Cache 是否已超过单卡 HBM 容量？如果超了，最少需要引入哪种分层策略才能将服务跑通？

**参考答案：**
- 模型权重（FP16）：70B × 2 bytes = 140GB，8 卡 TP 后每卡 ÷8 = ~17.5GB
- KV Cache（单序列，FP16）：2 × 56 × 8 × 128 × 128K × 2 = ~29.36GB
- 若 batch=4，单卡需 29.36×4/8（TP 分摊）+ 17.5 = ~32.2GB，未超 80GB
- 但 batch=16 时：29.36×16/8 + 17.5 = ~76.2GB，已接近 80GB 极限
- batch=32 时：29.36×32/8 + 17.5 = ~134.9GB，远超 80GB，必须引入分层管理
- **最少策略**：使用 FlexGen/KTransformers 风格的三级卸载（HBM→DRAM→NVMe），将部分低复用概率序列的 KV Cache 换出至 CPU 内存（约降 40-50% HBM 占用），即可在 batch=32 时维持运行。

---

> **报告结束**
> 本报告基于 2025-2026 年的公开技术资料整理，数据截至 2026 年 5 月 22 日。
