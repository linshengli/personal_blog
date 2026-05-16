# 大模型推理服务冷启动延迟优化技术 — 深度调研报告

> 调研日期：2026-05-16 | 领域：大模型框架

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

大模型推理服务的**冷启动延迟**（Cold Start Latency）指从推理实例启动（或收到首个请求）到生成首个令牌（First Token）之间的时间间隔。在无服务器（Serverless）和弹性扩缩容部署场景中，冷启动延迟是影响服务质量（SLO/SLA）的核心瓶颈，典型表现为首令牌时间（TTFT）显著劣化。

### 常见误解

| # | 误解 | 正解 |
|---|------|------|
| 1 | 冷启动 = 模型权重加载时间 | 权重加载仅占冷启动的一部分，还包括 CUDA Graph 捕获、KV 缓存初始化、JIT 编译等阶段，后者合计可达 50%+ |
| 2 | 冷启动只影响 Serverless 部署 | 冷启动同样影响蓝绿部署、模型热更新、A/B 测试等场景；每次实例重建都面临冷启动问题 |
| 3 | 冷启动可以通过纯硬件升级解决 | 虽然 H100→B200→GB300 带宽不断提升，但 CUDA Graph 捕获、KV 缓存分配等软件开销并不随硬件线性改善，仍需系统级优化 |

### 边界辨析

| 易混淆概念 | 核心区别 |
|-----------|---------|
| **冷启动 vs. 热启动** | 冷启动需加载权重+初始化运行时；热启动时模型已驻留 GPU，只需 KV 缓存分配 |
| **TTFT vs. TPOT** | TTFT（首令牌时间）受冷启动影响大；TPOT（每令牌输出时间）更多受推理引擎调度影响 |
| **冷启动 vs. P99 尾延迟** | 冷启动是首次请求的延迟，尾延迟是稳态下的高百分位延迟，两者优化手段不同 |
| **KV 缓存预热 vs. 模型加载** | KV 缓存预热指提前计算公共前缀的 KV 状态；模型加载指将权重从存储传输到 GPU 显存 |

## 1.2 核心架构

以下是大模型推理服务冷启动延迟的完整分解架构图：

```
┌─────────────────────────────────────────────────────────────┐
│               LLM 推理服务冷启动延迟分解                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌─────────┐  │
│  │ 权重加载  │ → │ 运行时初始化│ → │ KV 缓存  │ → │ CUDA    │  │
│  │ (Disk→GPU)│   │ (Python    │   │ 分配     │   │ Graph   │  │
│  │          │   │  Context)  │   │ (显存)   │   │ 捕获    │  │
│  └──────────┘   └───────────┘   └──────────┘   └─────────┘  │
│       ↓               ↓              ↓              ↓        │
│  优化方向：         优化方向：      优化方向：       优化方向： │
│  并行加载          JIT 缓存      按需分配        模板化/    │
│  存储层次          Snapshot      零开销前缀       惰性捕获   │
│  (SSD/DRAM/GPU)    持久化        缓存复用                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              核心瓶颈分解（典型占比）                    │    │
│  │  ┌────────────────┐  ┌──────────────────┐            │    │
│  │  │ 权重加载：~30%  │  │ CUDA Graph：~32% │            │    │
│  │  ├────────────────┤  ├──────────────────┤            │    │
│  │  │ KV 缓存：~18%  │  │ 运行时初始化：~22%│            │    │
│  │  └────────────────┘  └──────────────────┘            │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

各阶段说明：

| 阶段 | 职责说明 | 典型耗时（70B 模型） |
|------|---------|-------------------|
| **权重加载** | 将模型权重从持久化存储（SSD/云盘）加载到 GPU 显存 | 30-60s |
| **运行时初始化** | Python 环境配置、模型对象创建、推理引擎启动 | 10-20s |
| **KV 缓存分配** | 为预填充和解码阶段分配 KV 缓存显存空间 | 5-15s |
| **CUDA Graph 捕获** | 捕获计算图以消除 Kernel Launch 开销 | 10-65s |

## 1.3 数学形式化

### 公式 1：冷启动延迟分解

$$
T_{\text{cold\_start}} = T_{\text{load}} + T_{\text{init}} + T_{\text{kv\_alloc}} + T_{\text{cuda\_graph}}
$$

冷启动总延迟 = 权重加载时间 + 运行时初始化时间 + KV 缓存分配时间 + CUDA Graph 捕获时间

### 公式 2：权重加载时间模型（受 Amdahl 定律约束）

$$
T_{\text{load}} = \frac{M_{\text{model}}}{\min(B_{\text{storage}}, B_{\text{PCIe}}, B_{\text{mem}})} + T_{\text{overlap\_penalty}}
$$

其中 $M_{\text{model}}$ 为模型大小，$B_{\text{storage}}$ 为存储带宽，$B_{\text{PCIe}}$ 为 PCIe 带宽，$B_{\text{mem}}$ 为显存带宽。加载时间受三者中最慢的环节限制。

### 公式 3：KV 缓存内存占用

$$
M_{\text{KV}} = 2 \times L \times H \times S \times B \times d
$$

其中 $L$ 为层数，$H$ 为注意力头数，$S$ 为序列长度，$B$ 为批处理大小，$d$ 为每个元素字节数（FP16 为 2 字节）。KV 缓存在大序列和长上下文场景下可占用数百 GB 显存。

### 公式 4：CUDA Graph 捕获开销

$$
T_{\text{cuda\_graph}} = \sum_{i=1}^{N_{\text{bs}}} \sum_{j=1}^{M} t_{\text{capture}}(bs_i, op_j)
$$

CUDA Graph 捕获需要为每个批量大小（batch size）和每种算子组合进行独立的图捕获，总时间与批量大小种类数 $N_{\text{bs}}$ 和算子种类数 $M$ 成正比。vLLM 默认捕获 67 个图。

### 公式 5：整体系统效率

$$
E = \frac{T_{\text{compute}}}{T_{\text{compute}} + T_{\text{cold\_start}} + T_{\text{idle}}}
$$

系统效率 = 有效计算时间 /（有效计算时间 + 冷启动时间 + 空闲时间）。在 Serverless 场景中，频繁冷启动可将效率降至 50% 以下。

## 1.4 实现逻辑

以下伪代码展示了冷启动流程和各类优化策略的抽象：

```python
class LLMColdStartOptimizer:
    """大模型推理冷启动优化系统"""

    def __init__(self, config: ColdStartConfig):
        self.model_store = ModelStore(config.storage)       # 权重存储层
        self.kv_cache_manager = KVCacheManager(config.gpu)  # KV 缓存管理器
        self.graph_engine = CUDAGraphEngine(config.cuda)    # CUDA Graph 引擎
        self.runtime_cache = RuntimeCache(config.cache_dir) # 运行时缓存

    def cold_start(self, request: InferenceRequest) -> float:
        """执行带优化的冷启动，返回首令牌时间"""
        start = time.now()

        # 阶段 1：模型加载（可并行化）
        model = self._load_model_optimized(request.model_id)

        # 阶段 2：CUDA Graph 恢复 vs 捕获
        if self.graph_engine.has_cached(request.model_id, request.max_bs):
            self.graph_engine.restore(request.model_id)  # O(ms) 级恢复
        else:
            self.graph_engine.capture(                    # O(s) ~ O(min) 级捕获
                model, request.max_bs
            )

        # 阶段 3：KV 缓存分配（按需/惰性分配）
        kv_cache = self.kv_cache_manager.allocate(
            request, strategy="lazy"  # 仅分配当前批次所需
        )

        # 阶段 4：预填充 (Prefill)
        first_token = model.prefill(
            request.prompt,
            kv_cache=kv_cache,
            graph=self.graph_engine.get()
        )

        return time.now() - start

    def _load_model_optimized(self, model_id: str):
        """带缓存的优化模型加载"""
        # 尝试从 GPU 内存池复用（Tangram 方案）
        if model := self.model_store.reuse_from_gpu_pool(model_id):
            return model

        # 尝试从 CPU DRAM 加载
        if model := self.model_store.load_from_dram(model_id):
            return model

        # 最后从 SSD/网络加载（最慢路径）
        return self.model_store.load_from_disk(model_id)
```

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **TTFT（首令牌时间）** | < 500ms | 端到端请求计时 | 冷启动影响最直接的指标 |
| **冷启动总时长** | < 10s (Serverless) | 实例启动到就绪 | 70B 模型当前约 60s-28min |
| **CUDA Graph 捕获时间** | < 1s | 单次 Graph 捕获计时 | 当前 vLLM 默认需 10-65s |
| **模型加载吞吐** | > 5 GB/s | 加载速度测量 | Safetensors 约 1-3 GB/s |
| **KV 缓存命中率** | > 80% | 缓存命中统计 | 直接影响前缀请求的 TTFT |
| **实例扩缩容响应** | < 30s | 从触发到就绪 | 弹性场景的关键指标 |

## 1.6 扩展性与安全性

### 水平扩展

- **Prefill/Decode 分离架构**：将预填充和解码分配到不同 GPU，支持独立扩缩容。Together AI 的 CPD 方案通过三层节点（预预填充、标准预填充、解码）实现 35-40% 吞吐提升。
- **分布式 KV 缓存**：llm-d 项目利用 Kubernetes-native 调度实现 KV 缓存的全局感知路由，避免缓存颠簸。
- **流水线并行加载**：ParaServe 通过流水线并行将模型参数分布在多个 GPU 服务器上，利用聚合网络带宽并行获取，冷启动降低 4.7 倍。

### 垂直扩展

- **单节点上限**：显存容量（H100 80GB → B200 192GB）和互联带宽（NVLink 900 GB/s）持续提升，但 CUDA Graph 捕获时间不随硬件线性改善。
- **JIT 缓存持久化**：将编译后的 CUDA kernel 缓存保存到持久卷，避免重复编译。

### 安全考量

- **模型权重安全**：冷启动过程中的权重传输环节存在中间人攻击风险，需加密传输通道。
- **KV 缓存隔离**：多租户场景下，KV 缓存在 GPU 内存中的复用可能导致数据泄露，需实施严格的内存隔离。
- **拒绝服务攻击**：攻击者可通过发送大量冷启动请求耗尽推理集群资源，增加冷启动配额和速率限制是必要防御手段。

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | 35,000+ | PagedAttention 分页注意力，连续批处理，推测解码 | Python/C++/CUDA | 2026-05 | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| **SGLang** | 12,000+ | RadixAttention 前缀树缓存，结构化输出，推测解码 | Python/CUDA/Triton | 2026-05 | [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) |
| **TensorRT-LLM** | 10,000+ | NVIDIA 官方推理引擎，FP8/INT4 量化，内核融合 | C++/CUDA/TensorRT | 2026-05 | [github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |
| **Mooncake** | 2,500+ | 以 KVCache 为中心的解耦架构，RDMA 传输引擎 | C++/Python | 2026-04 | [github.com/kvcache-ai/Mooncake](https://github.com/kvcache-ai/Mooncake) |
| **LMCache** | 1,200+ | 分层 KV 缓存（GPU→DRAM→Disk），vLLM 集成 | Python/C++ | 2026-04 | [github.com/lmcache/lmcache](https://github.com/lmcache/lmcache) |
| **ServerlessLLM** | 410+ | 无服务器 LLM 部署，快速 Checkpoint 加载 | Python | 2025-02 | [github.com/serverlessllm/serverlessllm](https://github.com/serverlessllm/serverlessllm) |
| **llm-d** | 800+ | Kubernetes-native 分布式推理，Cache 感知路由 | Go/Python | 2026-05 | [github.com/llm-d/llm-d](https://github.com/llm-d/llm-d) |
| **PiKV** | 300+ | MoE 模型的多层 KV 缓存管理与调度 | Python/C++ | 2026-03 | [github.com/NoakLiu/PiKV](https://github.com/NoakLiu/PiKV) |
| **InferCache** | 200+ | 三层固定预算 KV 缓存（L0/L1/L2） | Python | 2026-02 | [github.com/ravirajb/infercache](https://github.com/ravirajb/infercache) |
| **llm-d-kv-cache** | 150+ | 分布式 KV 缓存调度与索引 | Python | 2026-04 | [github.com/llm-d/llm-d-kv-cache](https://github.com/llm-d/llm-d-kv-cache) |
| **KV Marketplace** | 100+ | 跨 GPU KV 缓存 RDMA/NVLink 共享 | C++/CUDA | 2026-01 | [github.com/neelsomani/kv-marketplace](https://github.com/neelsomani/kv-marketplace) |
| **Modal SnapStart** | 500+ | GPU 内存快照，加速 Serverless 推理启动 | Python | 2026-04 | [modal.com/docs/examples/sglang_snapshot](https://modal.com/docs/examples/sglang_snapshot) |
| **P/D-Serve** | N/A | 华为 Prefill/Decode 分离系统，10,000+ NPU 部署 | C++/Ascend | 2025-12 | 论文项目 |
| **EAGLE3** | 2,000+ | 推测解码第三代，80% 接受率 | Python/CUDA | 2026-02 | [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) |
| **FlashInfer** | 1,500+ | GPU Kernel 库，SGLang/vLLM 核心依赖 | CUDA/Triton | 2026-05 | [github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer) |

## 2.2 关键论文

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|-------|------|
| **Foundry: Template-Based CUDA Graph Materialization** | 学术团队 | 2026.04 | arXiv | 模板化 CUDA Graph 离线物化，冷启动降低 99%（10min→3.9s） | 开创性工作 | [2604.06664](https://arxiv.org/abs/2604.06664) |
| **Medusa: Accelerating Serverless LLM Inference with Materialization** | PCFC Lab | 2025.03 | ASPLOS'25 | 状态物化技术，加载延迟降 42.5%，TTFT 尾延迟降 53% | ASPLOS 顶会 | [10.1145/3669940.3707285](https://dl.acm.org/doi/10.1145/3669940.3707285) |
| **ParaServe: Swift Serverless LLM Cold Starts** | 学术团队 | 2025.02 | arXiv | 流水线并行加载，冷启动降低 4.7 倍 | 高被引预印本 | [2502.15524](https://arxiv.org/abs/2502.15524) |
| **Tangram: Accelerating Serverless LLM Loading** | 学术团队 | 2025.12 | arXiv | GPU 内存池复用 + 亲和性调度，加载快 6.2 倍 | 最新成果 | [2512.01357](https://arxiv.org/abs/2512.01357) |
| **Predictive-LoRA (P-LoRA)** | 学术团队 | 2025.12 | arXiv | LSTM 预测 + 主动预取适配器，冷启动降 68% | 创新方法 | [2512.20210](https://arxiv.org/abs/2512.20210) |
| **EdgeFlow: Fast Cold Starts on Mobile** | 学术团队 | 2026.04 | arXiv | NPU 自适应量化 + SIMD 打包，移动端冷启动降 4 倍 | 边缘计算 | [2604.09083](https://arxiv.org/abs/2604.09083) |
| **Stream2LLM: Overlap Context and Prefill** | 学术团队 | 2026.04 | arXiv | 流式预填充（检索与推理重叠），TTFT 改善 11 倍 | 架构创新 | [2604.16395](https://arxiv.org/abs/2604.16395) |
| **Layered Prefill: From Tokens to Layers** | 学术团队 | 2025.10 | arXiv | 逐层调度替代逐 token 调度，TTFT 降 70% | 调度革新 | [2510.08055](https://arxiv.org/abs/2510.08055) |
| **Hybrid JIT-CUDA Graph Optimization** | 学术团队 | 2026.04 | arXiv | JIT+CUDA Graph 混合优化，TTFT 降 66% vs TRT-LLM | 混合方法 | [2604.23467](https://arxiv.org/abs/2604.23467) |
| **Cache-Aware Prefill-Decode Disaggregation** | Together AI | 2026.02 | 技术报告 | 三层节点 + 三级 KV 缓存层次，吞吐提升 35-40% | 工业级实践 | [Together Blog](https://www.together.ai/blog/cache-aware-disaggregated-inference) |
| **P/D-Serve: Disaggregated LLM Serving** | 华为 | 2025 | 系统论文 | 10,000+ NPU 部署，连续无块 KV 传输，吞吐提升 6.7 倍 | 大规模实践 | - |
| **HyperFlexis: Unified LLM Serving** | 学术团队 | 2025.08 | arXiv | D2D 权重传输替代硬盘加载，冷启动降低 19.39 倍 | 创新传输 | - |

## 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| Cache-Aware Prefill-Decode Disaggregation (CPD) | Together AI Blog | 英文 | 架构解析 | 冷热分离推理架构设计，40% 吞吐提升 | 2026.02 | [together.ai](https://www.together.ai/blog/cache-aware-disaggregated-inference) |
| Anatomy of vLLM: High-Throughput LLM Inference | vLLM Blog | 英文 | 深度技术 | vLLM V1 架构详解，零开销前缀缓存 | 2025.09 | [blog.vllm.com.cn](https://blog.vllm.com.cn/2025/09/05/anatomy-of-vllm.html) |
| The Great LLM Inference Engine Showdown | Ultradune Blog | 英文 | 对比评测 | vLLM vs TGI vs TRT-LLM vs SGLang 全面对比 | 2025.11 | [buttondown.com](https://buttondown.com/ultradune/archive/eval-001-the-great-llm-inference-engine-showdown) |
| KV-Cache Wins: From Prefix Caching to Distributed Scheduling | llm-d Blog | 英文 | 技术实践 | 精确缓存感知调度 vs 近似策略，P90 TTFT 降低 57 倍 | 2025.08 | [llm-d.ai/blog](https://llm-d.ai/blog/kvcache-wins-you-can-see) |
| Deep Dive into llm-d and Distributed Inference | Solo.io | 英文 | 深度技术 | Kubernetes 原生分布式推理方案详解 | 2025.05 | [solo.io](https://www.solo.io/blog/deep-dive-into-llm-d-and-distributed-inference) |
| Speculative Decoding: 2-3x LLM Speedup | Introl Blog | 英文 | 技术教程 | 推测解码原理与工程实现，EAGLE3 详解 | 2025.12 | [introl.com](https://introl.com/blog/speculative-decoding-llm-inference-speedup-guide-2025) |
| Unlocking 25x Inference with SGLang on GB300 | LMSYS Blog | 英文 | 基准测试 | GB300 NVL72 上的推理性能突破 | 2026.02 | [lmsys.org](https://www.lmsys.org/blog/2026-02-20-gb300-inferencex/) |
| vLLM、SGLang 与 TensorRT-LLM 综合对比分析 | 阿里云开发者 | 中文 | 对比评测 | 三大推理引擎冷启动和吞吐对比 | 2025 | [aliyun.com](https://developer.aliyun.com/article/1686693) |
| CUDA Graph 在 LLM 推理中的战略价值 | CSDN 技术博客 | 中文 | 深度技术 | CUDA Graph 原理与冷启动优化技术 | 2025 | [blog.csdn.net](https://blog.csdn.net/plusplus168/article/details/151179930) |
| 大模型推理框架对比：SGLang 与行业主流方案 | 百度开发者 | 中文 | 对比评测 | SGLang 冷启动、TTFT、吞吐对比分析 | 2026 | [developer.baidu.com](https://developer.baidu.com/article/detail.html?id=6899371) |

## 2.4 技术演进时间线

```
2023 ─┬─ PagedAttention（vLLM）推出 → KV 缓存内存效率大幅提升
       ├─ 连续批处理（Continuous Batching）成为标准 → 推理吞吐提升 2-10 倍
       └─ 冷启动问题初现：Serverless LLM 部署兴起，冷启动成为瓶颈

2024 ─┬─ TensorRT-LLM 开源 → NVIDIA 官方推理引擎，高吞吐但编译期长达 28 分钟
       ├─ SGLang RadixAttention → 前缀缓存大幅降低重复请求的 TTFT
       ├─ Mooncake（Moonshot AI）→ KVCache 中心化解耦架构，吞吐提升 525%
       └─ ServerlessLLM → 首个专注冷启动的无服务器推理框架

2025 ─┬─ Medusa（ASPLOS'25）→ 状态物化技术，CUDA Graph 捕获时间降低 42.5%
       ├─ ParaServe → 流水线并行加载，冷启动降低 4.7 倍
       ├─ vLLM V1 + 零开销前缀缓存 → 前缀缓存无性能退化
       ├─ Together AI CPD → 工业级冷热分离推理架构
       ├─ P/D-Serve（华为）→ 10,000+ NPU 生产部署验证
       ├─ llm-d → KV 缓存感知的全局分布式调度
       └─ vLLM 商业化（估值 $800M）→ 资本加速推理基础设施发展

2026 ─┬─ Foundry → 模板化 CUDA Graph 物化，冷启动降低 99%（10min→3.9s）
       ├─ EdgeFlow → 移动端 NPU 推理冷启动优化
       ├─ Stream2LLM → 流式上下文与预填充重叠，TTFT 提升 11 倍
       ├─ HyperFlexis → D2D 权重传输，冷启动降低 19.39 倍
       ├─ SGLang v0.5.10 → JIT 编译缓存 + 句柄缓存优化
       └─ 当前状态：方案百花齐放，CUDA Graph 物化 + 解耦架构 + 缓存感知调度成为三大主流方向
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
2023 ─┬─ 问题发现期：Serverless LLM 推理场景涌现，冷启动延迟成为突出瓶颈
       └─ 典型表现：70B 模型冷启动需 3-5 分钟，远无法满足实时交互需求

2024 ─┬─ 第一波优化（系统架构层）：
       ├─ P/D 分离架构 → 消除预填充与解码的资源争抢
       ├─ 前缀缓存 → 复用公共前缀，大幅降低重复请求延迟
       ├─ P2P KV 缓存共享 → Mooncake 通过 RDMA 实现跨节点缓存共享
       └─ 典型效果：TTFT 从分钟级降至秒级

2025 ─┬─ 第二波优化（底层引擎层）：
       ├─ 状态物化 → Medusa 将 KV 缓存/CUDA Graph 离线预计算
       ├─ 流水线并行加载 → ParaServe 多机并行加载模型
       ├─ 预测预取 → P-LoRA 基于 LSTM 预测适配器需求
       └─ 典型效果：TTFT 从秒级降至百毫秒级

2026 ─┬─ 第三波优化（极速+超大规模）：
       ├─ 模板化 CUDA Graph → Foundry 将 10 分钟冷启动压缩至 3.9 秒
       ├─ D2D 权重传输 → HyperFlexis 降低 19 倍冷启动延迟
       ├─ 流式预填充 → Stream2LLM 检索与推理并行
       └─ 当前状态：8B 模型冷启动 < 5s，70B 模型 < 30s，向 < 1s 目标迈进
```

## 3.2 六种方案横向对比

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **CUDA Graph 物化**（Foundry/Medusa） | 离线预计算 Graph 拓扑和执行上下文，在线快速恢复 | ① 降低 CUDA Graph 捕获时间 90-99% ② 无运行时开销 ③ 兼容现有推理引擎 | ① 需额外离线存储空间 ② 模型结构变化需重新物化 ③ 对动态形状支持有限 | 生产环境固定模型部署、Serverless | 低（存储成本） |
| **P/D 分离 + 缓存感知调度**（Together CPD/llm-d） | 分离 Prefill/Decode 到不同节点，三级 KV 缓存层次，缓存感知路由 | ① 吞吐提升 35-40% ② 支持超大上下文 ③ 冷热请求分离 | ① 架构复杂，运维成本高 ② 需要 RDMA 网络 ③ 小规模部署收益不高 | 大规模生产集群、长上下文场景 | 高（GPU 集群+RDNA） |
| **模型加载优化**（Tangram/ParaServe/HyperFlexis） | GPU 内存池复用、流水线并行加载、D2D 直接传输 | ① 冷启动降低 4-19 倍 ② 直接减少加载时间 ③ 与引擎无关 | ① 需要额外 GPU 内存池 ② 流水线并行增加网络开销 ③ D2D 传输需要特殊硬件支持 | 多模型共享 GPU、模型频繁切换 | 中-高 |
| **前缀/KV 缓存复用**（RadixAttention/LMCache/PiKV） | 基于前缀树/非连续块复用的 KV 缓存系统，多层存储 | ① TTFT 降低 1.3-9.5 倍 ② 显著降低算力消耗 ③ 支持长上下文 | ① 缓存命中率依赖请求模式 ② 需要大容量 CPU/SSD 存储 ③ 缓存一致性管理复杂 | 多轮对话、Agent、RAG 场景 | 中 |
| **推测解码**（EAGLE3/Medusa） | 小型 draft model 并行生成候选 token，大模型验证 | ① 吞吐提升 2-3.6 倍 ② 单请求延迟降低 ③ 无需特殊硬件 | ① draft model 训练成本 ② 对生成质量有影响 ③ 并非直接解决冷启动问题 | 高吞吐需求、批量推理 | 中（额外 draft model） |
| **内存快照/Checkpoint优化**（Modal SnapStart/ServerlessLLM） | 保存 GPU 内存快照，恢复时直接加载到 GPU | ① 冷启动降低 10 倍 ② 简单易用 ③ 兼容多种引擎 | ① 快照文件大（GB 级） ② 加载快照仍需带宽 ③ 跨代 GPU 不兼容 | Serverless 推理、自动扩缩容 | 低-中 |

## 3.3 技术细节对比

| 维度 | CUDA Graph 物化 | P/D 分离+缓存调度 | 模型加载优化 | 前缀/KV 缓存复用 | 推测解码 | 内存快照 |
|------|----------------|-----------------|-------------|----------------|---------|---------|
| **性能（TTFT 降低）** | 53-99% | 35-40%（吞吐） | 75-95% | 50-90% | 50-65% | 80-90% |
| **易用性** | 中（需离线流程） | 低（架构复杂） | 高（透明替换） | 高（引擎原生支持） | 中（需额外模型） | 高（一键快照） |
| **生态成熟度** | 学术（原型） | 工业（早期） | 工业（较成熟） | 工业（成熟） | 工业（较成熟） | 工业（早期） |
| **社区活跃度** | 低 | 中 | 中 | 高 | 高 | 中 |
| **学习曲线** | 中 | 高 | 低 | 低 | 中 | 低 |
| **硬件依赖** | NVIDIA GPU | NVIDIA GPU + RDMA | 通用 | GPU + CPU/SSD | NVIDIA GPU | NVIDIA GPU |
| **维护复杂度** | 中 | 高 | 低 | 中 | 中 | 低 |

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证**（1-4 GPU） | vLLM/SGLang + 前缀缓存 | 即开即用，无额外基础设施；SGLang 的 RadixAttention 在中小规模下效果显著 | $1,000-5,000（GPU 实例） |
| **中型生产环境**（8-32 GPU） | SGLang/vLLM + LMCache + CUDA Graph 缓存 | 前缀缓存 + 多层 KV 缓存可降低 50% 以上的 TTFT；无需修改代码即可集成 | $10,000-50,000 |
| **大型分布式系统**（32+ GPU, 跨节点） | P/D 分离架构 + Cache-Aware 调度（Together CPD/llm-d/Mooncake） | 解耦架构可独立扩缩容 Prefill/Decode；缓存感知调度可避免缓存颠簸；RDMA 传输是基础设施标配 | $50,000-500,000+ |
| **Serverless/自动扩缩容** | Modal SnapStart + SGLang + Foundry CUDA Graph 物化 | 内存快照 + 模板化 CUDA Graph 可将冷启动压缩到 10s 以内；弹性扩缩容场景下 TCO 最优 | $5,000-30,000 |
| **长上下文/多轮对话** | SGLang（RadixAttention）+ LMCache + 推测解码 | 前缀树 + 多层缓存 + 推测解码的组合在长上下文中效果最佳 | $10,000-80,000 |
| **移动端/边缘计算** | EdgeFlow + 量化 + ONNX Runtime | 小模型 + 自适应量化 + NPU 流水线加载，冷启动 < 5s | $100-1,000（边缘设备） |

### 选型决策树

```
冷启动延迟要求 < 1s?
├── 是 → 是否有 RDMA 网络?
│   ├── 是 → P/D 分离 + Cache 感知调度（推荐 Together CPD/llm-d）
│   └── 否 → CUDA Graph 物化（推荐 Foundry/Medusa）+ 内存快照
├── 否 → 是否 Serverless 场景?
│   ├── 是 → SGLang + Modal SnapStart + 前缀缓存
│   └── 否 → 请求模式是否高度重复?
│       ├── 是 → SGLang RadixAttention + LMCache
│       └── 否 → vLLM + CUDA Graph 缓存 + 模型加载优化
```

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{LLM 冷启动优化} = \underbrace{\text{CUDA Graph 物化}}_{\text{消除捕获开销}} + \underbrace{\text{KV 缓存复用}}_{\text{避免重复计算}} - \underbrace{\text{模型加载延迟}}_{\text{存储/传输瓶颈}}
$$

## 4.2 一句话解释

大模型推理的"冷启动"就像每次打开一个超大文件都需要从头加载——优化技术的关键在于让系统"记住"之前加载好的状态（缓存计算图、缓存 KV 状态、预加载模型权重），从而把每次"冷"启动变成"温"或"热"启动，让用户几乎感觉不到等待。

## 4.3 核心架构图

```
冷启动请求 ──→ [缓存感知路由器] ──→ 是否命中缓存？
                   │                        │
               命中(热)                   未命中(冷)
                   │                        │
                   ↓                        ↓
              [复用 KV 缓存]          [快速模型加载]
                   │                        │
                   ↓                        ├──→ [CUDA Graph 恢复]（模板化/物化）
              立即生成                      ├──→ [KV 缓存分配]（按需/惰性）
                   │                        └──→ [前缀预填充]
                   │                        │
                   ↓                        ↓
              TTFT < 100ms              TTFT < 5s（优化后）
```

## 4.4 STAR 总结

### Situation（背景+痛点）

随着 ChatGPT/GPT-4 等大语言模型的普及，LLM 推理服务的部署规模急剧扩张。Serverless 和自动弹性扩缩容成为降低运营成本的首选架构。然而，LLM 推理实例的冷启动涉及权重加载（数十 GB）、CUDA Graph 捕获（10-65 秒）、KV 缓存分配（5-15 秒）等多个耗时阶段，总延迟可达数分钟。在生产环境中，这意味着用户面临数十秒乃至数分钟的首个响应等待，直接违反 SLO。根据 2025-2026 年的行业数据，冷启动导致的服务质量劣化已成为 LLM 推理基础设施最突出的瓶颈。

### Task（核心问题）

核心问题是：**如何在保持推理吞吐和精度的前提下，将 LLM 推理服务的冷启动延迟从分钟级降低到秒级甚至毫秒级？** 关键约束包括：(1) 模型规模持续增长（70B→671B→1T+），权重加载量不断加大；(2) 弹性场景下实例频繁启停，冷启动成本被重复支付；(3) 多租户环境需要严格隔离；(4) 成本敏感场景下无法无限制增加 GPU 资源。

### Action（主流方案）

技术演进经历了三个阶段。**第一波（2024，系统架构层）**：P/D 分离（Prefill/Decode 分离）消除资源争抢，前缀缓存（RadixAttention）复用公共前缀，Mooncake 实现跨节点 KV 缓存共享。**第二波（2025，底层引擎层）**：Medusa 提出状态物化技术离线预计算 CUDA Graph，ParaServe 通过流水线并行将加载降低 4.7 倍，P-LoRA 利用 LSTM 预测实现主动预取。**第三波（2026，极速阶段）**：Foundry 的模板化 CUDA Graph 物化将 10 分钟冷启动压缩至 3.9 秒，HyperFlexis 的 D2D 权重传输降低 19 倍延迟，Stream2LLM 实现检索与预填充的流水线并行。

### Result（效果+建议）

当前最佳实践下，8B 模型冷启动 < 5 秒，70B 模型 < 30 秒（对比 2023 年的 3-5 分钟）。对于中小型团队，推荐 **SGLang + 前缀缓存 + CUDA Graph 缓存**的组合方案，零代码改动即可获得 50-80% 的 TTFT 改善。大规模生产环境需投资 **P/D 分离 + 缓存感知调度**架构，虽然初期成本较高（需要 RDMA 网络和额外 GPU），但长期可降低 35-40% 的总推理成本。值得警惕的是，当前优化方案多依赖 NVIDIA 生态，开源社区（尤其是 AMD ROCm）的支持仍显不足。预计 2027 年冷启动优化的重点关注方向将包括：非连续块缓存（Beyond-Prefix Caching）、端到端 1 秒冷启动、跨厂商硬件统一抽象层。

## 4.5 理解确认问题

**问题**：大模型推理冷启动中，CUDA Graph 捕获为何会成为瓶颈？为什么不能简单地"不捕获 CUDA Graph"来解决？

**参考答案**：
CUDA Graph 捕获之所以是瓶颈，是因为 LLM 推理通常需要为 67 种（vLLM 默认配置）不同的批量大小分别执行一次完整的图捕获操作，每次耗时 100ms-1s，合计可达 10-65 秒。之所以不能简单地禁用 CUDA Graph（`--enforce-eager` 模式），是因为 CUDA Graph 消除了每次 Kernel 调用时 CPU→GPU 的指令发射开销——在 LLM 推理中，单个 Decode 步骤可能触发数百个小型 CUDA Kernel，每个 Kernel 的启动开销（约 5-20μs）累积起来会导致 2-5 倍的吞吐下降。因此冷启动优化的正确方向不是"禁用"，而是通过物化（离线预计算）、模板化（Foundry）或惰性捕获（按需捕获）来降低捕获阶段的时间开销，同时保留运行时 CUDA Graph 的性能优势。

---

> 本报告基于公开可获取的 GitHub 数据、学术论文预印本和技术博客编写，调研日期为 2026 年 5 月 16 日。Star 数据为近似值，可能随时间变化。学术论文引用标注了 arXiv ID 或 DOI 链接。
