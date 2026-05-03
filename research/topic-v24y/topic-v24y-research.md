# 大模型推理前缀缓存与缓存复用优化技术 —— 深度调研报告

**调研日期**：2026-05-03
**所属域**：大模型框架
**技术标签**：`KV Cache` `Prefix Caching` `RadixAttention` `Cache Reuse` `PD Disaggregation`

---

# 第一部分：概念剖析

## 1.1 定义澄清

**通行定义**：大模型推理前缀缓存（Prefix Caching / KV Cache Reuse）是一种针对 Transformer 自回归解码过程的优化技术。其核心思想是：当多个推理请求共享相同的输入前缀（如 system prompt、few-shot 示例、RAG 检索上下文）时，将首轮计算得到的 **Key-Value Cache（KV Cache）** 存储在 GPU 显存/主机内存/SSD 中，后续请求直接复用已缓存的 KV Cache，避免对共享前缀的重复 Prefill 计算，从而显著降低首 Token 时延（TTFT）并提升系统吞吐量。

**常见误解**：

1. **误解一："前缀缓存和 KV Cache 是同一个概念"**
   实际 KV Cache 是 Transformer 解码过程中生成的 Key、Value 矩阵的缓存，是每个请求必然产生的中间结果；而前缀缓存是在此之上的一种**复用策略**，决定不同请求间如何共享 KV Cache。

2. **误解二："只要前缀相同就能完美复用"**
   实际上，前缀复用受诸多约束：block 对齐、LoRA 适配器 ID、多模态输入的 image hash、cache salt（租户隔离）以及 Attention 计算的数值精度，任何一项不匹配都会导致缓存失效。

3. **误解三："前缀缓存仅对固定 system prompt 有效"**
   现代技术（如 CacheBlend、LMCache）已支持**非前缀位置**的 KV Cache 复用与融合，即重复文本块即使出现在 prompt 中间位置也能部分复用，这对 RAG 和 Agent 场景至关重要。

**边界辨析**：与 **Speculative Decoding（投机解码）** 的区别在于——投机解码通过小模型生成候选 token 然后用大模型验证来加速解码，而前缀缓存加速的是 **Prefill 阶段**而非 Decode 阶段，两者可正交组合。与 **Flash Attention** 的区别在于——Flash Attention 优化的是单次 Attention 计算的访存效率，而前缀缓存优化的是跨请求的计算复用。

## 1.2 核心架构

```
┌──────────────────────────────────────────────────────────────────┐
│                  Prefix Caching 系统架构                           │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  请求输入                                                         │
│     │                                                             │
│     ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │                   调度层 (Scheduler)                      │      │
│  │  - 解析请求前缀    - 查询缓存索引    - 计算缓存命中      │      │
│  │  - 请求调度策略 (FCFS / Longest-Prefix-First / 亲和性)   │      │
│  └──────────┬──────────────────────────────────────────────┘      │
│             │                                                      │
│       ┌─────┴─────┐                                               │
│       ▼           ▼                                               │
│  ┌─────────┐ ┌──────────┐                                         │
│  │ 缓存命中 │ │ 缓存未命中│                                         │
│  │ (复用)   │ │ (计算)   │                                         │
│  └────┬────┘ └────┬─────┘                                         │
│       │           │                                                │
│       ▼           ▼                                                │
│  ┌──────────────────────────────────────────┐                     │
│  │            KV Cache 管理层                 │                     │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐  │                     │
│  │  │ 哈希索引  │  │ 树/前缀树 │  │ 引用计数│  │                     │
│  │  │ (O(1)查) │  │ (灵活匹配)│  │ (共享) │  │                     │
│  │  └────┬─────┘  └────┬─────┘  └───┬────┘  │                     │
│  └───────┼──────────────┼────────────┼───────┘                     │
│          │              │            │                              │
│          ▼              ▼            ▼                              │
│  ┌──────────────────────────────────────────┐                     │
│  │          多级缓存存储 (Tiered Storage)     │                     │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐  │                     │
│  │  │ GPU显存   │ │ 主机DRAM │ │ SSD/NVMe │  │                     │
│  │  │ (热数据)  │ │ (温数据)  │ │ (冷数据)  │  │                     │
│  │  └──────────┘ └──────────┘ └──────────┘  │                     │
│  └──────────────────────────────────────────┘                     │
│             │                                                      │
│             ▼                                                      │
│  ┌──────────────────────────────────────────┐                     │
│  │            Prefill / Decode 执行层        │                     │
│  │  - 增量 Prefill (仅计算未缓存部分)        │                     │
│  │  - Decode (复用已缓存 KV Cache)          │                     │
│  └──────────────────────────────────────────┘                     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

**各层职责**：
- **调度层**：决定请求分配策略，优先将请求调度到已有其缓存前缀的节点
- **KV Cache 管理层**：核心数据结构层，维护缓存索引与共享引用
- **多级缓存存储**：GPU→DRAM→SSD 分层存储，兼顾容量与速度
- **Prefill/Decode 执行层**：执行实际的 Attention 计算，增量填充缺失部分

## 1.3 数学形式化

### 公式1：KV Cache 内存占用

$$
\text{Mem}_{\text{KV}} = 2 \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times L \times \text{dtype\_bytes}
$$

> 每个 token 需要存储 K 和 V 两个矩阵，总内存与层数、注意力头数、头维度、序列长度和数据类型精度成正比。例如，Llama 3.1 70B 在 16K 上下文、FP16 下，KV Cache 单请求占用约 4.8 GB。

### 公式2：Prefix Caching 加速比

$$
\text{Speedup} = \frac{T_{\text{prefill\_full}}}{T_{\text{prefill\_cached}}} = \frac{L_{\text{total}}}{L_{\text{total}} - L_{\text{cached}}}
$$

> 加速比等于完整 Prefill 时间与增量 Prefill 时间之比。当缓存前缀长度占总长度的 90% 时，理论加速比为 10 倍（忽略缓存查找开销）。

### 公式3：缓存命中率与吞吐量关系

$$
\text{Throughput} = \frac{B}{\frac{N_{\text{miss}} \cdot T_{\text{prefill}}}{S_{\text{batch}}} + T_{\text{decode}}}
$$

> 其中 \( B \) 为批次大小，\( N_{\text{miss}} \) 为缓存未命中的请求比例，\( T_{\text{prefill}} \) 为全量 Prefill 时间，\( S_{\text{batch}} \) 为批处理加速因子。缓存命中率越高（\( N_{\text{miss}} \) 越小），整体吞吐量越大。

### 公式4：Radix Tree 查找复杂度

$$
\text{Lookup}(P) = O(|P|) \quad \text{(Trie)}
$$

$$
\text{Lookup}_{\text{block}}(P) = O\left(\frac{|P|}{B}\right) \quad \text{(Hash Map)}
$$

> 前缀树（Radix Tree）的查找时间复杂度正比于前缀长度（SGLang），而基于块哈希的方案（vLLM APC）复杂度降为块数，通过减少查找次数换取块对齐的存储效率损失。\( B \) 为块大小（vLLM 默认 16 tokens）。

### 公式5：多级缓存的有效容量

$$
\text{Capacity}_{\text{eff}} = C_{\text{GPU}} + \frac{C_{\text{CPU}}}{\alpha_{\text{latency}}} + \frac{C_{\text{SSD}}}{\beta_{\text{latency}}}
$$

> 有效缓存容量不是各层容量的简单加和，而是按访问延迟的倒数加权。\( \alpha_{\text{latency}} \) 为 CPU DRAM 相对于 GPU HBM 的延迟倍率（~5-10×），\( \beta_{\text{latency}} \) 为 SSD 相对于 GPU 的延迟倍率（~50-100×）。通过 CacheGen 压缩后可进一步提升有效容量。

## 1.4 实现逻辑（Python 伪代码）

```python
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

# ── 配置 ──────────────────────────────────────────────
@dataclass
class CacheConfig:
    block_size: int = 16           # 每个缓存块的 token 数
    max_gpu_blocks: int = 10000    # GPU 显存可容纳的最大块数
    enable_offload: bool = True    # 是否启用 CPU/SSD 卸载
    cache_salt: Optional[str] = None  # 租户隔离盐值


# ── 块级数据结构 (vLLM 风格) ──────────────────────────
@dataclass
class KVCacheBlock:
    block_id: int
    block_hash: Optional[str] = None
    ref_cnt: int = 0               # 被多少请求引用
    tokens: List[int] = None       # 块内 token IDs
    evict_priority: int = 35       # 默认优先级 (TensorRT-LLM 风格)
    next_free: Optional[int] = None
    prev_free: Optional[int] = None


# ── 前缀树节点 (SGLang 风格) ──────────────────────────
class RadixNode:
    def __init__(self):
        self.children: Dict[int, 'RadixNode'] = {}
        self.parent: Optional['RadixNode'] = None
        self.kv_tensors: Optional[torch.Tensor] = None  # 实际 KV Cache 张量
        self.last_access_time: float = 0
        self.hit_count: int = 0
        self.depth: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 0


# ── 前缀缓存系统核心 ─────────────────────────────────
class PrefixCacheManager:
    """管理 KV Cache 的缓存、查找与淘汰"""

    def __init__(self, config: CacheConfig):
        self.config = config
        # 块哈希 → 块 ID 的映射 (O(1) 查找)
        self.cache_blocks: Dict[str, int] = {}
        # 所有预分配的块对象池
        self.block_pool: List[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(config.max_gpu_blocks)
        ]
        # LRU 空闲队列 (双向链表)
        self.free_queue_head: Optional[int] = None
        self.free_queue_tail: Optional[int] = None
        # 请求 → 分配的块列表
        self.request_blocks: Dict[str, List[int]] = {}

    def compute_block_hash(self, parent_hash: str, tokens: List[int],
                           extra: str = "") -> str:
        """计算块的链式哈希 (SHA256)"""
        content = f"{parent_hash}:{tokens}:{extra}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get_computed_blocks(self, prompt_tokens: List[int]) -> List[int]:
        """查找已有缓存——返回缓存命中的块 ID 列表"""
        matched = []
        parent_hash = self.config.cache_salt or ""
        for i in range(0, len(prompt_tokens), self.config.block_size):
            block_tokens = prompt_tokens[i:i + self.config.block_size]
            block_hash = self.compute_block_hash(parent_hash, block_tokens)
            if block_hash in self.cache_blocks:
                matched.append(self.cache_blocks[block_hash])
                parent_hash = block_hash  # 链式传递
            else:
                break  # 一旦不匹配，后续块也无法命中
        return matched

    def allocate_slots(self, request_id: str, num_blocks: int) -> List[int]:
        """为新请求分配缓存块"""
        allocated = []
        for _ in range(num_blocks):
            # 从空闲队列头部取出一个块
            block_id = self._pop_free_block()
            if block_id is None:
                raise MemoryError("KV Cache 已满，无法分配新块")
            allocated.append(block_id)
        self.request_blocks[request_id] = allocated
        return allocated

    def free_request(self, request_id: str):
        """请求完成后释放块——逆序加入空闲队列尾部"""
        if request_id not in self.request_blocks:
            return
        blocks = self.request_blocks.pop(request_id)
        for block_id in reversed(blocks):  # 逆序释放
            self._return_to_free_queue(block_id)

    def _pop_free_block(self) -> Optional[int]:
        """LRU 淘汰——弹出队列头部最久未使用的块"""
        if self.free_queue_head is None:
            return None
        block_id = self.free_queue_head
        block = self.block_pool[block_id]
        # 如果该块有缓存，移除缓存映射
        if block.block_hash:
            del self.cache_blocks[block.block_hash]
            block.block_hash = None
        # 更新队列头部
        self.free_queue_head = block.next_free
        if self.free_queue_head is not None:
            self.block_pool[self.free_queue_head].prev_free = None
        return block_id

    def touch_block(self, block_id: int):
        """标记块为活跃——从空闲队列中移除"""
        block = self.block_pool[block_id]
        if block.prev_free is not None:
            self.block_pool[block.prev_free].next_free = block.next_free
        if block.next_free is not None:
            self.block_pool[block.next_free].prev_free = block.prev_free
        if self.free_queue_head == block_id:
            self.free_queue_head = block.next_free
        if self.free_queue_tail == block_id:
            self.free_queue_tail = block.prev_free
        block.prev_free = block.next_free = None


# ── 前缀树实现 (SGLang RadixAttention 简化版) ────────
class RadixAttentionCache:
    """基于前缀树的 KV Cache 管理"""

    def __init__(self):
        self.root = RadixNode()
        self.evict_queue: List[RadixNode] = []

    def insert(self, token_ids: List[int], kv_tensor: torch.Tensor):
        """将 token 序列及其 KV Cache 插入前缀树"""
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                new_node = RadixNode()
                new_node.parent = node
                new_node.depth = node.depth + 1
                node.children[tid] = new_node
            node = node.children[tid]
        node.kv_tensors = kv_tensor  # 叶子节点存储 KV Cache
        node.last_access_time = time.time()

    def find_longest_prefix(self, token_ids: List[int]) -> Tuple[int, RadixNode]:
        """查找最长匹配前缀——返回匹配长度和对应的节点"""
        node = self.root
        matched = 0
        for tid in token_ids:
            if tid not in node.children:
                break
            node = node.children[tid]
            matched += 1
            node.last_access_time = time.time()
            node.hit_count += 1
        return matched, node

    def evict_leaf(self):
        """LRU 淘汰——优先淘汰叶子节点"""
        self.evict_queue.sort(key=lambda n: n.last_access_time)
        for node in self.evict_queue:
            if node.is_leaf() and node.kv_tensors is not None:
                node.kv_tensors = None
                self.evict_queue.remove(node)
                return True
        return False
```

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| TTFT 降低比例 | 50-90% | 端到端基准（有/无缓存对比） | 取决于缓存命中率，高命中率可达 10× 加速 |
| 缓存命中率 | > 80% 理想 | 实际请求跟踪统计 | 受前缀分布、缓存容量、淘汰策略影响 |
| 缓存查找开销 | < 1ms | Profiling | vLLM SHA256 约 75ns/token（CPU 开销） |
| 单块匹配粒度 | 16 tokens（vLLM）/ 1 token（SGLang） | 架构设计决定 | 粒度越细缓存效率越高，但管理开销越大 |
| KV Cache 压缩比 | 3.5-4.3×（CacheGen）/ 10×（QAQ） | 压缩前后内存对比 | 压缩与质量损失的权衡 |
| 最大缓存容量 | 单 GPU ~200K tokens（Llama 70B FP16） | 理论计算 | 使用量化+卸载可扩展至数百万 tokens |
| 多级缓存命中率提升 | 3% -> 50%+（SSD 辅助） | 生产环境 A/B 测试 | SSD 极大扩展了缓存覆盖范围 |

## 1.6 扩展性与安全性

**水平扩展**：
- **Distributed KV Cache**：跨 GPU 节点共享缓存池，通过 RDMA（如 CHIME、NVIDIA NIXL）或以太网传输 KV Cache 元数据。AIBrix + InfiniStore 方案在 4.3 QPS 下将 TTFT 从 296s 降至 0.58s。
- **Cache-Aware Routing**：调度层根据各节点已有的缓存前缀分布，将请求优先路由到具备可用缓存的节点，避免随机路由导致的缓存碎片化（DigitalOcean 指出的 round-robin 之下命中率降至 ~1/N 的问题）。
- **PD Disaggregation**：将 Prefill 和 Decode 分离到不同节点，Prefill 节点写 KV Cache 到分布式存储，Decode 节点从存储读取，实现独立扩缩容。

**垂直扩展**：
- **GPU HBM 升级**：单节点可通过更大 HBM（如 H200 的 141GB / B200 的 192GB）直接提升缓存容量。
- **NVLink 域内共享**：同节点内多 GPU 可通过 NVLink 共享 KV Cache，减少冗余计算。
- **量化精度下探**：从 FP16 → FP8 → INT4 → MXFP4，每步约 2× 容量提升，但精度损失需通过量化感知训练或自适应量化补偿。

**安全考量**：
- **跨租户缓存泄漏**：多租户场景下，缓存命中可能暴露其他用户请求的前缀信息。vLLM 通过 `cache_salt` 实现租户隔离——仅相同 salt 的请求能命中同一缓存。
- **时序侧信道攻击**：缓存命中与否会导致 TTFT 差异（命中 ~2ms vs 未命中 ~200ms），攻击者可通过测量响应时间推断前缀是否被缓存过。使用随机延迟注入可缓解。
- **内容污染**：恶意构造的特殊前缀可导致缓存中毒，影响后续请求的正确性。需要缓存内容校验和版本控制。
- **数据过期**：当模型权重更新后，旧 KV Cache 在语义空间中的表示可能失效，需实现缓存版本标记和自动失效。

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | ~78,000 | PagedAttention + 自动前缀缓存(APC)，哈希块 + LRU 淘汰 | Python/C++/CUDA | 2026-05 | [GitHub](https://github.com/vllm-project/vllm) |
| **SGLang** | ~25,000 | RadixAttention 前缀树，token 级粒度缓存，结构化生成 | Python/C++/CUDA | 2026-05 | [GitHub](https://github.com/sgl-project/sglang) |
| **TensorRT-LLM** | ~15,000 | 优先级 LRU + 部分匹配复用 + 二级 CPU Offload | C++/CUDA | 2026-04 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **LMCache** | ~3,500 | 多级 KV 缓存层（GPU→DRAM→SSD→Redis）+ CacheGen 压缩 | Python/CUDA | 2026-04 | [GitHub](https://github.com/LMCache/LMCache) |
| **LMDeploy** | ~7,000 | TurboMind 引擎，支持前缀缓存（块大小 128 tokens 可配） | C++/CUDA/Python | 2026-04 | [GitHub](https://github.com/InternLM/lmdeploy) |
| **HuggingFace TGI** | ~9,000 | 前缀缓存 + 分块 Prefill，长上下文（>200K tokens）优化 | Rust/Python | 2025-12（维护模式） | [GitHub](https://github.com/huggingface/text-generation-inference) |
| **NVIDIA Dynamo** | — | KV Block Manager 管理 GPU/CPU/SSD/网络缓存卸载 | C++/CUDA | 2026-04 | [GitHub](https://github.com/ai-dynamo/dynamo) |
| **AIBrix** | ~1,200 | 多级 KV 卸载 + 前缀感知路由 + InfiniStore RDMA 缓存 | Go/Python | 2026-05 | [GitHub](https://github.com/aibrix/aibrix) |
| **FlashAttention** | ~14,000 | 高效 Attention 计算内核，为 KV Cache 优化提供底层依赖 | CUDA | 2026-04 | [GitHub](https://github.com/Dao-AILab/flash-attention) |
| **Mooncake** | ~1,800 | PD 分离推理的 KV Cache 传输层，DeepSeek 开源 | C++/Python | 2026-03 | [GitHub](https://github.com/kvcache-ai/mooncake) |

## 2.2 关键论文

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|-------|------|
| **PagedAttention** (vLLM) | Kwon et al. (UC Berkeley) | 2023 | SOSP | 提出分页 KV Cache 管理，消除内部碎片，是后续所有前缀缓存的基础 | 被引 2000+ | [arXiv](https://arxiv.org/abs/2309.06180) |
| **SGLang: RadixAttention** | Zheng et al. (UCSD) | 2024 | OSDI | 提出前缀树管理 KV Cache，支持 token 级灵活匹配和最长前缀优先调度 | 被引 500+ | [arXiv](https://arxiv.org/abs/2407.07328) |
| **CacheGen** | Liu et al. (UChicago) | 2024 | SIGCOMM | KV Cache 压缩编码（3.5-4.3×）+ 带宽自适应流式传输 | 被引 200+ | [arXiv](https://arxiv.org/abs/2310.07240) |
| **CacheBlend** | Wang et al. | 2024 | ACM TOCS | 非前缀位置 KV Cache 融合复用（选择性重算+融合），RAG 场景 TTFT 降低 2.2-3.3× | — | [arXiv](https://arxiv.org/abs/2405.16444) |
| **HotPrefix** | Li, Gu, Huan, Wang et al. (南京大学) | 2026 | **SIGMOD** | 热度感知布谷鸟过滤器 + GPU/CPU 协同调度，比 vLLM 提升 2.25× | 顶会收录 | [NJU](https://ise.nju.edu.cn/info/1017/2581.htm) |
| **PrefillShare** | Peking Univ. | 2026 | arXiv | 冻结 Prefill 模块、多模型共享 KV Cache，p95 时延降低 4.5× | 新论文 | [arXiv](https://arxiv.org/abs/2602.12029) |
| **ICaRus** | — | 2026 | **ICLR** Poster | 逻辑编/解码器分解，多模型相同 KV Cache 复用，内存节省最高 11.1× | 顶会收录 | [ICLR](https://iclr.cc/virtual/2026/poster/10007206) |
| **ContiguousKV** | — | 2026 | arXiv | 统一数据管理粒度 ContiguousChunk + 异步预取，Re-Prefill 加速 3.85× | 新论文 | [arXiv](https://arxiv.org/abs/2601.13631) |
| **PrfaaS** | Moonshot AI × 清华 | 2026 | arXiv | 跨数据中心 KV Cache 传输，以太网传输 Prefill 结果，吞吐 +54% | 新论文 | [arXiv](https://arxiv.org/abs/2604.15039) |
| **KVTC** | — | 2026 | **ICLR** | PCA 去相关 + 自适应量化 + 熵编码，压缩比最高 20× | 顶会收录 | [arXiv](https://arxiv.org/abs/2511.01815) |
| **Sequential KV Compression** | — | 2026 | arXiv | 概率前缀去重 + 预测性 Delta 编码，超越 Shannon 极限 | 新论文 | [arXiv](https://arxiv.org/abs/2604.15356) |
| **Rando-KV** (Randomization Boosts KV Caching) | — | 2026 | **ICLR** Poster | 随机化淘汰 + 学习驱动的查询路由，延迟/吞吐提升最高 6.92× | 顶会收录 | [OpenReview](https://openreview.net/forum?id=R7fv5NWfMm) |

## 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| Advanced Prompt Caching at Scale | DigitalOcean | EN | 技术博客 | 多副本缓存问题、session affinity 路由、分层缓存架构 | 2026-03 | [Link](https://www.digitalocean.com/blog/advanced-prompt-caching) |
| Cache-Aware Prefill-Decode Disaggregation | Together AI | EN | 技术博客 | CPD 三层架构，40% 吞吐提升，长时间上下文场景 | 2026-04 | [Link](https://www.together.ai/blog/cache-aware-disaggregated-inference) |
| Inside vLLM: Anatomy of a High-Throughput System | vLLM Blog | EN | 架构解析 | vLLM v1架构、APC 哈希实现、0 开销缓存 | 2025-09 | [Link](https://blog.vllm.com.cn/2025/09/05/anatomy-of-vllm.html) |
| SGLang vs vLLM: KV Cache Reuse Multi-Turn | RunPod | EN | 对比评测 | 多轮对话场景下 SGLang 前缀缓存优势（5× 吞吐提升） | 2025-10 | [Link](https://www.runpod.io/blog/sglang-vs-vllm-kv-cache) |
| How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo | NVIDIA | EN | 技术博客 | KV Block Manager、GPU Direct Storage、NIXL 传输 | 2025-09 | [Link](https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/) |
| CacheGen: Store Your KV Cache on Disk or S3 | LMCache Blog | EN | 技术教程 | CacheGen 集成 vLLM 实战，TTFT 737ms vs naive 4355ms | 2025-07 | [Link](https://blog.lmcache.ai/en/2025/07/31/cachegen-store-your-kv-cache-on-disk-or-s3-load-blazingly-fast/) |
| LMCache Joins the PyTorch Ecosystem | PyTorch Blog | EN | 官方公告 | LMCache 成为 PyTorch 生态项目，支持 vLLM/SGLang | 2025-11 | [Link](https://pytorch.org/blog/lmcache-joins-pytorch-ecosystem/) |
| vLLM 内参：深度剖新高吞吐量大模型推理系统 | Doocs 技术社区 | 中文 | 深度教程 | vLLM 内部架构、APC 实现细节、性能分析 | 2025-09 | [Link](https://docs.d.run/blogs/2025/inside-vllm) |
| vLLM、SGLang 与 TensorRT-LLM 综合对比分析报告 | 阿里云开发者 | 中文 | 对比评测 | 三大框架在 KV Cache 管理、前缀缓存、性能方面的详实对比 | 2025-11 | [Link](https://developer.aliyun.com/article/1686693) |
| 大模型推理性能优化全景：72 项技术深度解析 | AI快讯 | 中文 | 综述 | 包含前缀缓存在内的完整推理优化技术图谱 | 2026-04 | [Link](https://blockchain.news/zh/ainews/llm-inference-vs-traditional-ml-9-pillars-and-72-optimization-techniques-explained-2026-analysis-zh) |

## 2.4 技术演进时间线

```
2023 ─┬─ PagedAttention (vLLM SOSP'23) ─→ 分页 KV Cache 管理，消除碎片
      ├─ FlexGen ─→ KV Cache CPU/磁盘卸载的早期探索
      └─ 首次提出 Prefix Caching 概念 ─→ 基础思路确立

2024 ─┬─ RadixAttention (SGLang OSDI'24) ─→ 前缀树管理，token 级复用
      ├─ CacheGen (SIGCOMM'24) ─→ KV Cache 压缩编码 3.5-4.3×
      ├─ CacheBlend ─→ 非前缀位置缓存融合
      ├─ TensorRT-LLM 优先级缓存 ─→ NVIDIA 企业级前缀缓存
      └─ vLLM APC 正式发布 ─→ 哈希 APC 成为社区标准

2025 ─┬─ vLLM v1 重构 ─→ "零开销"前缀缓存，即使 0% 命中率也无性能损失
      ├─ TGI v3 前缀缓存 ─→ 长上下文（>200K）场景 13× 提速
      ├─ NVIDIA Dynamo ─→ KV Block Manager + 多级卸载
      ├─ LMCache 生态化 ─→ 加入 PyTorch 生态系统
      ├─ PD Disaggregation 普及 ─→ Prefill/Decode 分离架构成为主流
      ├─ Mooncake (DeepSeek) ─→ 开源 PD 分离 KV Cache 传输层
      └─ AIBrix v0.3.0 ─→ 多级 KV 卸载 + InfiniStore RDMA 缓存

2026 ─┬─ HotPrefix (SIGMOD'26) ─→ 热度感知调度，2.25× 优于 vLLM
      ├─ ICaRus (ICLR'26) ─→ 多模型共享同一 KV Cache，11.1× 节省
      ├─ KVTC (ICLR'26) ─→ PCA + 量化，20× 压缩比
      ├─ PrfaaS ─→ 跨数据中心 KV Cache 传输
      ├─ PrefillShare ─→ 冻结 Prefill 模块，多模型复用
      ├─ Rando-KV (ICLR'26) ─→ 随机化淘汰 + 学习路由
      ├─ Sequential KV Compression ─→ 超越 Shannon 极限
      └─ 当前状态：从小范围前缀复用 → 跨模型/跨数据中心/智能化缓存的全面演进
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
2023 ─┬─ vLLM PagedAttention ─→ 首次将操作系统分页思想引入 KV Cache 管理
2024 ─┼─ SGLang RadixAttention ─→ 前缀树替代哈希表，实现灵活缓存匹配
2024 ─┼─ TensorRT-LLM 优先级缓存 ─→ 引入企业级缓存管理和二级 offload
2025 ─┼─ 多级缓存卸载 (LMCache) ─→ GPU→DRAM→SSD→RDMA 分层存储
2025 ─┼─ PD Disaggregation ─→ Prefill/Decode 分离，跨节点缓存共享
2026 ─┼─ 跨模型/跨数据中心缓存 (PrefillShare, PrfaaS) ─→ 缓存复用边界不断扩展
2026 ─┴─ 当前状态：从"单机块级缓存"进入"分布式、多模型、智能调度"时代
```

## 3.2 5 种核心方案横向对比

### 方案 A：vLLM Automatic Prefix Caching (APC)

| 原理 | 优点 (3+) | 缺点 (3+) | 适用场景 | 成本量级 |
|------|-----------|-----------|---------|---------|
| 基于 SHA256 哈希的块级缓存（块大小 16 tokens），LRU 淘汰，O(1) 块查找 | ① 实现简单，工程成熟度高，社区最大（78K Stars）<br>② v1 架构下 0% 命中率也无性能损失<br>③ 硬件兼容性好（NVIDIA + AMD ROCm）<br>④ 丰富的生产级功能（多模态缓存、cache salt 隔离） | ① 固定块大小导致缓存粒度粗，部分匹配效率低<br>② 块对齐要求严格，不完全匹配即无法缓存<br>③ LRU 淘汰策略未考虑访问频率<br>④ 极端高并发下调度开销增加 | 批量推理、模板化 prompt、通用部署场景 | 免费开源，部署运维成本中等 |

### 方案 B：SGLang RadixAttention

| 原理 | 优点 (3+) | 缺点 (3+) | 适用场景 | 成本量级 |
|------|-----------|-----------|---------|---------|
| 基于前缀树的 KV Cache 管理，token 级粒度匹配，最长前缀优先（LPF）调度 | ① Token 级缓存粒度，零配置自动匹配<br>② 结构化生成（JSON/Regex）原生支持<br>③ 多轮对话和 Agent 场景碾压级优势（5× 吞吐）<br>④ LPF 调度减少缓存碎片 | ① 学习曲线较陡，文档不如 vLLM 完善<br>② 高并发（>100）场景性能不及 vLLM<br>③ 生态成熟度略逊（25K Stars vs 78K）<br>④ 硬件兼容性（AMD ROCm）支持较晚 | Agent 工作流、多轮对话、RAG、结构化输出 | 免费开源，部署运维成本略高 |

### 方案 C：TensorRT-LLM + NVIDIA Triton

| 原理 | 优点 (3+) | 缺点 (3+) | 适用场景 | 成本量级 |
|------|-----------|-----------|---------|---------|
| 分页 KV Cache + 优先级 radix 树 + CPU 二级 offload + 缓存事件 API + 感知路由 | ① NVIDIA 硬件上性能天花板最高<br>② 优先级 LRU（0-100）支持细粒度控制<br>③ KV Cache Event API 实现集群级缓存感知路由<br>④ 二级内存 offload 扩展缓存容量 | ① 仅支持 NVIDIA GPU，生态锁定<br>② 配置复杂，KVCacheRetentionConfig 等 API 门槛高<br>③ 社区活跃度和资源不如 vLLM/SGLang<br>④ 部分功能（partial reuse）有已知限制 | NVIDIA 独占部署、延迟敏感型企业生产环境 | 企业级，需 NVIDIA 硬件+许可，成本较高 |

### 方案 D：LMDeploy TurboMind

| 原理 | 优点 (3+) | 缺点 (3+) | 适用场景 | 成本量级 |
|------|-----------|-----------|---------|---------|
| C++/CUDA 高性能引擎，块级前缀缓存（默认 128 tokens），支持 KV Cache 卸载 | ① 吞吐量最高可比 vLLM 高 156%<br>② 量化灵活（W4A16KV8 等任意组合）<br>③ 生态集成好（MMLU、OpenCompass 同团队）<br>④ KV 卸载功能在 v0.10.0 已生产可用 | ① 前缀缓存块粒度大（128 tokens），部分缓存效率低<br>② 社区规模较小（~7K Stars）<br>③ 文档和最佳实践不如 vLLM 丰富<br>④ 功能更新节奏不如 vLLM 高频 | InternLM 模型生态、需要极致吞吐量的量化部署场景 | 免费开源，运维成本低 |

### 方案 E：TGI v3 (HuggingFace)

| 原理 | 优点 (3+) | 缺点 (3+) | 适用场景 | 成本量级 |
|------|-----------|-----------|---------|---------|
| 专用前缀缓存数据结构 + 分块 Prefill，长上下文场景优化，Rust 实现高性能 | ① 超长上下文（>200K tokens）场景性能最优（13× vs vLLM）<br>② Rust 实现，内存安全性和单请求延迟低<br>③ HuggingFace 生态深度集成 | ① **已进入维护模式（2025-12）**，无新功能开发<br>② 高并发吞吐量远低于 vLLM（~24× 差距）<br>③ 不支持多 GPU 分布式部署<br>④ Rust 技术栈二次开发门槛高 | 长上下文推理、小规模部署、HuggingFace 生态用户 | 免费开源，但社区已进入维护期 |

## 3.3 技术细节对比

| 维度 | vLLM APC | SGLang RadixAttention | TensorRT-LLM | LMDeploy TurboMind | TGI v3 |
|------|---------|---------------------|--------------|-------------------|--------|
| **缓存数据结构** | 哈希映射 (SHA256) | 前缀树 (Radix Tree) | Radix Tree + 优先级 | 分块哈希表 | 专用前缀缓存 |
| **缓存粒度** | 固定 16 tokens | 可变（token 级） | 分页 (paged) | 固定 128 tokens | 分块 + 可变 |
| **查找复杂度** | O(1)/块 | O(\|P\|) | O(\|P\|) | O(1)/块 | O(1) |
| **80% 命中 TTFT 加速** | ~3-5× | ~5-8× | ~2-5× | ~2-4× | ~5-13×（长上下文） |
| **高并发吞吐 (>100)** | 优秀（最优） | 良好 | 优秀 | 优秀 | 一般（短板） |
| **多模态缓存** | ✓（image hash） | ✓ | ✓（v1.0+） | — | — |
| **缓存隔离** | ✓（cache salt） | ✓（per-request） | ✓（优先级隔离） | — | — |
| **CPU Offload** | 外部（LMCache） | 外部（LMCache） | 原生（二级内存） | 原生（v0.10+） | — |
| **PD 分离支持** | ✓（Mooncake） | ✓ | ✓（NVIDIA Dynamo） | ✓（Mooncake等） | — |
| **硬件兼容性** | NVIDIA + AMD | NVIDIA + AMD | NVIDIA 独占 | NVIDIA + 部分国产 | NVIDIA |
| **学习曲线** | 低 | 中 | 高 | 中低 | 低 |
| **社区活跃度** | ★★★★★ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆（维护期） |

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证** | vLLM APC | 社区最大（78K Stars），文档最完善，零成本上手，APC 开箱即用 | $0（单卡 A100，按需云实例 ~$3,000/月） |
| **多轮对话 Chatbot** | SGLang RadixAttention | 5× 吞吐提升，token 级前缀自动匹配，多轮对话场景优势明显 | $8,000-15,000（2-4× H100，含缓存 SSD） |
| **Agent 工作流/ToT** | SGLang + LMCache | RadixAttention + CacheBlend 非前缀复用 + 多级缓存卸载，支撑复杂 Agent 拓扑 | $15,000-30,000（4-8× H100，分层缓存） |
| **RAG 长上下文服务** | SGLang 或 TGI v3 + LMCache | 长 prompt 场景（>100K），TGI 专长，但 SGLang + LMCache 更可持续发展 | $10,000-20,000 |
| **大规模生产部署 (>200 concurrency)** | vLLM v1 + NVIDIA Triton | vLLM 高并发吞吐量最优，Triton 提供稳定生产级 serving | $50,000-150,000（多节点 H200/B200） |
| **NVIDIA 独占企业环境** | TensorRT-LLM | NVIDIA 硬件性能天花板最高，优先级 LRU + Event API 提供最精细控制 | $80,000-200,000+（含 NVIDIA 企业支持） |
| **量化+吞吐优先部署** | LMDeploy TurboMind | W4A16KV8 灵活量化，吞吐比 vLLM 高 58-156% | $5,000-12,000（单/双卡，量化后可用更低端硬件） |
| **跨数据中心部署** | PrfaaS + Mooncake | 跨数据中心 KV Cache 传输，Hybrid Attention 缩减体积，商品以太网即可 | $100,000+（多数据中心分布式部署） |
| **多模型共享服务** | PrefillShare / ICaRus | 冻结 Prefill 模块，多模型共享同一 KV Cache，内存节省最高 11.1× | $30,000-80,000（显著减少 GPU 需求） |

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{Prefix Caching} = \underbrace{\text{KV Cache Reuse}}_{\text{避免重复计算}} + \underbrace{\text{Multi-tier Storage}}_{\text{扩展缓存容量}} - \underbrace{\text{Cache Redundancy \& Consistency}}_{\text{淘汰开销 + 版本失效}}
$$

> 前缀缓存的核心本质是"空间换时间"：用存储空间换取 Prefill 计算时间。公式表明，有效缓存 = 复用的计算量 + 分层的存储量 - 管理和一致性损耗。优秀的缓存系统（如 SGLang + LMCache）通过细粒度前缀树和多级存储最大化前两项，同时通过 LRU 和热度感知最小化第三项。

## 4.2 一句话解释

**"就像你写论文时不需要每次打开都从绪论读到方法部分——大模型推理时，如果多个用户发送相似的前面部分（如系统指令、知识库上下文），系统会把这些已计算好的中间结果缓存起来，后面的请求直接从'共享段落'之后开始算，而不是每次都从头算一遍。"**

## 4.3 核心架构图

```
请求1: [System Prompt | 用户问题 A]  →  从头计算，缓存 System Prompt
请求2: [System Prompt | 用户问题 B]  →  命中缓存！跳过 System Prompt，仅增量计算
请求3: [System Prompt | 用户问题 C]  →  命中缓存！跳过 System Prompt，仅增量计算

缓存时间线:
  t0: Prefill System Prompt (计算)    ───→ KV Cache [System Prompt]
  t1: Prefill 用户问题 A (计算)        ───→ KV Cache [Sys. + QA]
  t2: 命中 System Prompt (复用)        ───→ 仅计算用户问题 B 的 KV
  t3: 命中 System Prompt (复用)        ───→ 仅计算用户问题 C 的 KV
       ↓ TTFT 从 ~200ms → ~20ms
       ↓ GPU 计算量节省 ~80-90%

  [GPU HBM] ← 热数据 (当前活跃请求)
      ↓ offload
  [CPU DRAM] ← 温数据 (最近访问过的)
      ↓ offload
  [NVMe SSD] ← 冷数据 (LRU 淘汰，但保留以备后续)
      ↓ 压缩 (CacheGen: 3.5-4.3×)
  [分布式存储] ← 跨节点共享 (PrfaaS: 跨数据中心)
```

## 4.4 STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation（背景+痛点）** | 大模型推理中，Prefill 阶段需要计算全部输入 token 的 KV Cache，计算量随序列长度线性增长。在多轮对话、RAG 和 Agent 场景下，大量请求共享相同的 system prompt、few-shot 示例或检索上下文，但传统推理引擎对每个请求都"从头计算"，导致 GPU 计算严重浪费（50-90% 的 token 在不同请求间重复），首 Token 时延（TTFT）居高不下（通常 200ms-数秒）。 |
| **Task（核心问题）** | 如何设计一种缓存机制，使不同请求间能够高效地共享和复用 KV Cache，在确保计算正确性和精度一致性的前提下，最大化缓存命中率、最小化缓存管理开销，并支持跨请求、跨模型、甚至跨数据中心的缓存共享？核心约束包括：缓存粒度（块 vs token）、查找效率、淘汰策略、多租户隔离和存储层级设计。 |
| **Action（主流方案）** | 行业经历了四个阶段的演进：① **PagedAttention 时代**（2023）：vLLM 引入分页 KV Cache 管理，奠定块级缓存基础；② **RadixAttention 时代**（2024）：SGLang 用前缀树替代哈希表，实现 token 级灵活匹配；③ **多层次分载时代**（2025）：LMCache/CacheGen 将缓存扩展至 CPU DRAM、SSD 和分布式存储，缓存命中率从 3% 提升至 50%+；④ **智能调度+跨模型时代**（2026）：HotPrefix 热度感知、ICaRus 多模型共享、PrfaaS 跨数据中心传输等前沿研究突破传统边界。 |
| **Result（效果+建议）** | 当前最优实践可实现 80-90% 的缓存命中率、TTFT 降低 5-10 倍、吞吐量提升 3-5 倍。**建议**：中小规模选 vLLM APC（生态成熟）；多轮对话和 Agent 选 SGLang RadixAttention（缓存效率最高）；生产级 NVIDIA 环境选 TensorRT-LLM（性能天花板）；超大场景叠加 LMCache 多级卸载。但所有方案都面临"冷启动"（首次请求无缓存）和数据一致性（模型更新后缓存失效）两大固有局限，跨数据中心缓存仍处于早期阶段。 |

## 4.5 理解确认问题

**问题**：假设你的大模型服务面临这样的工作负载——100 个并发用户，每个人都使用了相同的 2000-token System Prompt，但后面的用户输入完全不同（每个 50 tokens）。请计算：
1. 如果使用 SGLang（token 级前缀匹配），第一轮全部 100 个请求的**总 Prefill 计算量**（以 token 为单位）是多少？
2. 相比"完全不使用前缀缓存"的方案，节省了多少计算量？
3. 如果使用 vLLM APC（16-token 块），缓存命中率是否会降低？为什么？

**参考答案**：
1. 第一轮第一个请求需要计算全部 2050 tokens（2000+50）。后续 99 个请求各需计算增量 50 tokens（2000 tokens 全部命中缓存）。总计算量 = 2050 + 99 × 50 = 2050 + 4950 = **7000 tokens**。
2. 无缓存时需要 100 × 2050 = 205,000 tokens。节省比例 = (205000 - 7000) / 205000 = **96.6%**。
3. 会略有降低。因为 vLLM 的块大小为 16 tokens，如果 2000-token 的 system prompt 恰好是 16 的整数倍（125 块），则命中率相同。但如果 system prompt 长度不是 16 的整数倍，最后一块不完整将无法缓存，导致每个后续请求需要额外 Prefill 1-15 个"尾部"token。同时，如果请求在 2000 之后续接的内容有细微差异（如开头有特殊格式），可能破坏块对齐，进一步降低命中率。这就是 SGLang 的 token 级粒度在"精确匹配"场景的优势。

---

# 附录：关键术语表

| 术语 | 英文 | 定义 |
|------|------|------|
| KV Cache | Key-Value Cache | Transformer 解码过程中各层 Attention 计算的 Key、Value 矩阵缓存 |
| Prefix Caching | 前缀缓存 | 跨请求共享相同前缀的 KV Cache 以避免重复 Prefill 计算 |
| Prefill | 预填充阶段 | 对输入 prompt 进行并行 Attention 计算，生成首个输出 token 的过程 |
| Decode | 解码阶段 | 逐个 token 自回归生成的过程，每次迭代计算一个 token |
| TTFT | Time to First Token | 从请求到达到收到第一个输出 token 的时间，是最重要的用户体验指标 |
| RadixAttention | 基数树注意力 | SGLang 提出的基于前缀树的 KV Cache 管理方法 |
| APC | Automatic Prefix Caching | vLLM 的自动前缀缓存机制，基于 SHA256 哈希 |
| PD Disaggregation | Prefill/Decode 分离 | 将 Prefill 和 Decode 部署在不同 GPU 节点上的架构模式 |
| LPF | Longest Prefix First | 最长前缀优先的请求调度策略 |
| CacheGen | — | KV Cache 压缩编码和流式传输系统（SIGCOMM'24） |

---

> **本报告调研日期：2026-05-03**
> 所有 GitHub Stars 数据、论文接受情况和基准测试结果均基于当时公开信息。
> 技术演进速度极快，建议每季度更新一次数据。