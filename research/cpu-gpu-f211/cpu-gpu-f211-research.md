# 大模型推理 CPU-GPU 异构计算协同调度深度调研报告

**调研主题：** 大模型推理 CPU-GPU 异构计算协同调度
**所属领域：** 大模型框架
**调研日期：** 2026-04-04
**版本：** 2.0（2026 年 4 月更新版）

---

## 目录

1. [第一部分：概念剖析](#第一部分概念剖析)
2. [第二部分：行业情报](#第二部分行业情报)
3. [第三部分：方案对比](#第三部分方案对比)
4. [第四部分：精华整合](#第四部分精华整合)

---

## 第一部分：概念剖析

### 1. 定义澄清

#### 通行定义

**大模型推理 CPU-GPU 异构计算协同调度**是指在大型语言模型（LLM）推理服务中，通过智能调度算法动态分配计算任务到 CPU 和 GPU 两类异构计算资源上，以实现内存容量扩展、计算效率优化和成本控制的技术体系。其核心思想是打破"所有计算必须在 GPU 上完成"的传统范式，利用 CPU 的大内存容量和 GPU 的高计算吞吐能力，通过精细化的任务切分、数据布局和流水线调度，在满足延迟约束的前提下最大化系统吞吐。

2025-2026 年的新发展将这一概念扩展为"**异步并行异构执行**"：CPU 和 GPU 不再仅仅是主从 offload 关系，而是可以真正并行执行不同阶段的计算（如 APEX 方案中 CPU 负责预填充、GPU 负责解码），通过计算重叠实现延迟优化。

#### 常见误解

1. **误解一："CPU-GPU 协同就是简单地把部分层放到 CPU 上"**
   实际上，协同调度涉及复杂的内存管理（如 PagedAttention）、算子融合、通信隐藏和预测性预取，远非简单的层分配问题。2025 年的 APEX 方案更是实现了预填充和解码的异步并行，而非简单的层切分。

2. **误解二："CPU 参与计算一定会显著降低性能"**
   在内存受限场景下，合理的 CPU offloading 可以通过避免 GPU OOM 和减少数据重传，反而提升整体吞吐。对于 decode 阶段的部分算子，CPU 执行可能不会成为瓶颈。Prima.cpp（2025）证明通过智能调度，CPU-GPU 协同可实现接近纯 GPU 的延迟。

3. **误解三："异构调度只适用于超大模型"**
   实际上，在高并发多租户场景下，即使是 7B-13B 模型也能通过异构调度实现更好的资源利用率和 QoS 保障。SageSched（2026）展示了混合负载场景下异构调度的成本优势。

4. **误解四："只有边缘设备才需要 CPU-GPU 协同"**
   数据中心场景同样受益：CPU-GPU 协同可用于服务混合负载（在线 + 离线请求）、实现 SLO 保证、降低 TCO（总体拥有成本）。

#### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **模型并行（Tensor/Pipeline Parallelism）** | 关注多 GPU 间的计算划分，而 CPU-GPU 协同关注异构资源间的任务分配 |
| **动态批处理（Continuous Batching）** | 关注请求级别的调度，而异构调度关注算子/层级别的资源分配 |
| **量化推理（Quantized Inference）** | 关注降低精度减少内存，而异构调度关注跨设备资源利用 |
| **推测解码（Speculative Decoding）** | 关注加速生成过程，而异构调度关注计算资源的跨设备调度 |
| **CPU Offload（ZeRO-Inference）** | 单向将数据从 GPU 卸载到 CPU；异构调度是双向协作，CPU 主动参与计算 |

---

### 2. 核心架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    大模型推理 CPU-GPU 异构调度系统                    │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐                                                    │
│  │  请求接入层  │ ← HTTP/gRPC 请求，批处理管理                       │
│  └──────┬──────┘                                                    │
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      调度决策引擎                                ││
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          ││
│  │  │ 内存预测器   │  │ 负载评估器   │  │ 成本优化器   │          ││
│  │  │ (预测 KV 需求) │  │ (GPU 利用率)  │  │ (延迟/吞吐)  │          ││
│  │  └──────────────┘  └──────────────┘  └──────────────┘          ││
│  └─────────────────────────────────────────────────────────────────┘│
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      执行引擎层                                  ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ││
│  │  │ GPU 计算单元  │←→→ │ 通信协调器   │←→→│ CPU 计算单元  │         ││
│  │  │ (Attention,  │    │ (PCIe/NVLink│    │ (Offload,   │         ││
│  │  │  FFN, Norm)  │    │  数据搬运)  │    │  预处理)    │         ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────────┘│
│         ▼                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                      存储管理层                                  ││
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         ││
│  │  │ GPU HBM     │    │ CPU DRAM    │    │ NVMe SSD    │         ││
│  │  │ (KV Cache)  │←→→│ (Offload)   │←→→│ (Swap)      │         ││
│  │  └─────────────┘    └─────────────┘    └─────────────┘         ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

**组件职责说明：**

| 组件 | 职责 |
|------|------|
| **请求接入层** | 接收推理请求，维护请求队列，执行连续批处理（Continuous Batching） |
| **内存预测器** | 基于序列长度和模型结构预测 KV Cache 需求，为调度决策提供依据 |
| **负载评估器** | 实时监控 GPU 计算单元利用率、显存占用率，评估是否触发 offload |
| **成本优化器** | 根据延迟 SLO 和吞吐目标，权衡 CPU/GPU 任务分配比例 |
| **GPU 计算单元** | 执行计算密集型算子（Attention、FFN），维护活跃 KV Cache |
| **CPU 计算单元** | 执行 offloaded 的层计算、数据预处理、后处理任务 |
| **通信协调器** | 管理 PCIe/NVLink 数据传输，实现计算 - 通信重叠 |
| **存储管理层** | 三级存储管理（HBM→DRAM→SSD），实现 KV Cache 的分页交换 |

---

### 3. 数学形式化

#### 3.1 异构调度优化目标

$$
\min_{\mathcal{A}} \quad \mathbb{E}[T_{\text{end-to-end}}] = \sum_{l=1}^{L} \left( \mathbb{I}_{a_l=G} \cdot t_l^G + \mathbb{I}_{a_l=C} \cdot (t_l^C + 2 \cdot t_{\text{transfer}}) \right)
$$

**解释：** 最小化端到端延迟，其中 $a_l \in \{G, C\}$ 表示第 $l$ 层的分配决策（GPU 或 CPU），$t_l^G$ 和 $t_l^C$ 分别是 GPU 和 CPU 上的执行时间，$t_{\text{transfer}}$ 是跨设备数据传输时间。

#### 3.2 内存容量约束

$$
\text{KV}_{\text{total}} = \sum_{r \in \text{requests}} \left( 2 \cdot H \cdot S_r \cdot D \cdot B \right) \leq M_{\text{GPU}} + \alpha \cdot M_{\text{CPU}}
$$

**解释：** 总 KV Cache 需求不能超过 GPU 显存加上可 offload 的 CPU 内存（$\alpha$ 为 offload 比例系数）。其中 $H$ 是注意力头数，$S_r$ 是序列长度，$D$ 是 head dimension，$B$ 是每个 token 的字节数。

#### 3.3 吞吐 - 延迟权衡模型

$$
\text{Throughput}(\lambda) = \frac{\lambda}{1 + P_{\text{offload}}(\lambda) \cdot \beta_{\text{overhead}}}
$$

**解释：** 系统吞吐随请求率 $\lambda$ 变化，$P_{\text{offload}}$ 是触发 offload 的概率，$\beta_{\text{overhead}}$ 是 offload 带来的额外开销系数。该模型揭示了高负载下 offload 对吞吐的影响。

#### 3.4 计算 - 通信重叠效率（APEX 模型）

$$
\eta_{\text{overlap}} = \frac{T_{\text{compute}}}{\max(T_{\text{compute}}, T_{\text{transfer}})} \times 100\%
$$

**解释：** 重叠效率衡量计算与数据传输的并行程度。理想情况下 $T_{\text{transfer}} < T_{\text{compute}}$，$\eta_{\text{overlap}} \approx 100\%$；否则通信成为瓶颈。APEX 通过异步执行使 CPU 预填充与 GPU 解码重叠，显著提升效率。

#### 3.5 成本效益比

$$
\text{CostEff} = \frac{\text{Tokens/sec}}{\text{GPU\_hours} \cdot C_G + \text{CPU\_hours} \cdot C_C}
$$

**解释：** 每单位成本产生的 token 吞吐量，其中 $C_G$ 和 $C_C$ 分别是 GPU 和 CPU 的单位时间成本。该指标用于评估异构调度的经济性。

#### 3.6 MoE 专家分配效率（2025 年新模型）

$$
\text{Efficiency}_{\text{MoE}} = \frac{\sum_{e \in \text{Active}} T_e^{\text{GPU}}}{\sum_{e \in \text{Active}} T_e^{\text{GPU}} + \sum_{e \in \text{Inactive}} T_e^{\text{CPU-offload}}}
$$

**解释：** MoE 模型的效率取决于激活专家在 GPU 上的执行时间与非激活专家 CPU 卸载开销的比值。对于 671B 的 DeepSeek-V3，每 token 仅激活约 37B 参数，其余可安全 offload。

---

### 4. 实现逻辑（Python 伪代码）

```python
class HeterogeneousScheduler:
    """
    CPU-GPU 异构调度核心类
    负责动态决策每层计算应该分配到 GPU 还是 CPU
    """
    def __init__(self, config):
        self.gpu_memory = config.gpu_hbm_size      # GPU 显存容量 (GB)
        self.cpu_memory = config.cpu_dram_size     # CPU 内存容量 (GB)
        self.pcie_bandwidth = config.pcie_bw       # PCIe 带宽 (GB/s)
        self.model_config = config.model           # 模型配置 (层数、隐藏维度等)
        self.memory_pool = PagedMemoryPool(config) # 分页内存池
        self.profiler = OperationProfiler()        # 算子性能剖析器

    def schedule_request(self, requests: List[Request]) -> ExecutionPlan:
        """
        为核心请求生成执行计划，决定每层的设备分配
        """
        # 步骤 1: 预测内存需求
        kv_demand = self._predict_kv_demand(requests)

        # 步骤 2: 评估 GPU 剩余容量
        gpu_available = self.gpu_memory - self.memory_pool.allocated

        # 步骤 3: 决定 offload 策略
        if kv_demand <= gpu_available:
            # GPU 足够，全 GPU 执行
            plan = self._all_gpu_plan(requests)
        else:
            # 需要 offload，计算最优分配
            plan = self._heterogeneous_plan(requests, kv_demand, gpu_available)

        return plan

    def _heterogeneous_plan(self, requests, kv_demand, gpu_available) -> ExecutionPlan:
        """
        生成异构执行计划：将部分层 offload 到 CPU
        """
        layers_to_offload = self._select_layers_to_offload(
            kv_demand - gpu_available
        )

        plan = ExecutionPlan()
        for layer_id in range(self.model_config.num_layers):
            if layer_id in layers_to_offload:
                plan.add_stage(
                    layer_id,
                    device="CPU",
                    compute_fn=self._cpu_layer_forward,
                    pre_fetch=True,  # 预取权重和数据
                    post_transfer=True  # 结果传回 GPU
                )
            else:
                plan.add_stage(
                    layer_id,
                    device="GPU",
                    compute_fn=self._gpu_layer_forward
                )

        # 优化：尝试重叠通信和计算
        plan.optimize_overlap(self.pcie_bandwidth)
        return plan

    def execute_plan(self, plan: ExecutionPlan, inputs: Tensor) -> Tensor:
        """
        执行异构计划，管理跨设备数据流
        """
        intermediate = inputs
        for stage in plan.stages:
            if stage.device == "GPU":
                intermediate = stage.compute_fn(intermediate)
            else:
                # CPU 执行：需要处理数据传输
                gpu_data = intermediate
                cpu_data = self._transfer_to_cpu(gpu_data)
                cpu_result = stage.compute_fn(cpu_data)
                intermediate = self._transfer_to_gpu(cpu_result)

        return intermediate

    def _select_layers_to_offload(self, excess_memory: int) -> Set[int]:
        """
        选择要 offload 的层：优先 offload 计算密度低的层
        """
        layer_costs = []
        for layer_id in range(self.model_config.num_layers):
            mem_cost = self._estimate_layer_memory(layer_id)
            compute_cost = self.profiler.get_compute_time(layer_id, "GPU")
            transfer_cost = self._estimate_transfer_time(layer_id)

            # offload 代价 = 额外传输时间 / GPU 计算时间
            offload_penalty = transfer_cost / compute_cost
            layer_costs.append((layer_id, mem_cost, offload_penalty))

        # 按单位内存收益排序，优先 offload 代价小的层
        layer_costs.sort(key=lambda x: x[2] / x[1])

        layers_to_offload = set()
        freed_memory = 0
        for layer_id, mem_cost, _ in layer_costs:
            if freed_memory >= excess_memory:
                break
            layers_to_offload.add(layer_id)
            freed_memory += mem_cost

        return layers_to_offload


class PagedMemoryPool:
    """
    分页内存池：类似 vLLM 的 PagedAttention 思想
    支持 GPU-CPU-SSD 三级存储的弹性扩展
    """
    def __init__(self, config):
        self.block_size = config.block_size  # 每 block 管理的 token 数
        self.gpu_blocks = self._allocate_gpu_blocks(config.gpu_memory)
        self.cpu_blocks = self._allocate_cpu_blocks(config.cpu_memory)
        self.block_tables: Dict[RequestID, List[int]] = {}

    def allocate(self, request_id: str, num_tokens: int) -> List[int]:
        """为请求分配物理 block，返回 block 表"""
        num_blocks = ceil(num_tokens / self.block_size)

        # 优先分配 GPU block
        gpu_blocks_needed = min(num_blocks, len(self.gpu_blocks.free))
        cpu_blocks_needed = num_blocks - gpu_blocks_needed

        allocated = []
        allocated.extend(self.gpu_blocks.allocate(gpu_blocks_needed))
        allocated.extend(self.cpu_blocks.allocate(cpu_blocks_needed))

        self.block_tables[request_id] = allocated
        return allocated

    def swap_out(self, request_id: str) -> bool:
        """将请求的 KV Cache 从 GPU swap 到 CPU/SSD"""
        if request_id not in self.block_tables:
            return False

        block_table = self.block_tables[request_id]
        for i, block_id in enumerate(block_table):
            if self._is_gpu_block(block_id):
                # 异步传输到 CPU
                self._async_transfer_gpu_to_cpu(block_id)
                # 回收 GPU block
                self.gpu_blocks.free_block(block_id)
                # 分配 CPU block
                new_block_id = self.cpu_blocks.allocate(1)[0]
                block_table[i] = new_block_id

        return True

    def swap_in(self, request_id: str) -> bool:
        """将请求的 KV Cache 从 CPU/SSD swap 回 GPU"""
        # 与 swap_out 对称的操作
        pass


class APEXScheduler(HeterogeneousScheduler):
    """
    APEX 异步并行调度器（2025 年新方案）
    核心思想：CPU 负责预填充，GPU 负责解码，两者并行执行
    """
    def __init__(self, config):
        super().__init__(config)
        self.prefill_queue = asyncio.Queue()
        self.decode_queue = asyncio.Queue()

    def schedule_apex(self, request: Request) -> AsyncExecutionPlan:
        """
        APEX 调度：预填充和解码异步并行
        """
        # CPU 异步执行预填充
        prefill_task = asyncio.create_task(
            self._cpu_prefill(request.prompt)
        )

        # GPU 准备解码（等待 prefill 完成）
        decode_task = asyncio.create_task(
            self._gpu_decode(request, prefill_task)
        )

        return AsyncExecutionPlan(
            prefill=prefill_task,
            decode=decode_task,
            overlap_expected=True
        )
```

---

### 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **首 token 延迟 (TTFT)** | < 100ms (prefill), < 20ms (decode) | 端到端基准测试 | 用户感知的首个响应时间 |
| **每 token 延迟 (TPOT)** | < 50ms/ token | 生成阶段平均 | decode 阶段的稳态延迟 |
| **GPU 吞吐** | > 1000 tokens/s (单卡 A100) | 满载压力测试 | 单 GPU 的 token 生成速率 |
| **系统吞吐** | > 5000 tokens/s (多卡 +CPU) | 多请求并发测试 | 整体系统吞吐能力 |
| **内存利用率** | > 85% | 运行时监控 | HBM 有效使用比例 |
| **Offload 开销** | < 15% | 对比纯 GPU 基线 | offload 带来的额外延迟 |
| **计算 - 通信重叠率** | > 80% | 性能剖析工具 | 通信隐藏的有效性 |
| **QoS 达成率** | > 99% | SLO 监控 | 满足延迟承诺的请求比例 |
| **成本效率** | > 2x (对比纯 GPU) | 成本/吞吐核算 | 单位成本的吞吐产出 |
| **CPU-GPU 并行效率** | > 70% | APEX 模式专用 | CPU 和 GPU 同时工作的时间占比 |

---

### 6. 扩展性与安全性

#### 水平扩展

异构调度系统的水平扩展主要通过以下方式实现：

1. **多 GPU 数据并行**：通过请求分片将不同请求分配到不同 GPU，每个 GPU 独立进行 CPU-GPU 协同调度
2. **模型并行 + 异构**：超大模型采用 tensor parallelism 跨多 GPU，同时在每个 GPU 节点上启用 CPU offload
3. **分布式 KV Cache**：跨节点的 KV Cache 共享，利用远端 CPU 内存扩展本地容量（LLMServingSim 2.0, 2026）
4. **弹性扩缩容**：基于负载预测动态调整 CPU offload 比例，实现资源的弹性伸缩

扩展瓶颈主要在于：
- PCIe 带宽成为多 GPU 场景下的共享瓶颈
- 跨节点通信延迟影响分布式 KV Cache 效率
- 调度器的集中式决策可能成为单点瓶颈

#### 垂直扩展

单节点垂直扩展的优化空间：

1. **GPU 侧**：
   - 使用 HBM3/HBM3e 提升显存带宽（3-3.5 TB/s）
   - 采用 Tensor Core 和稀疏化加速计算
   - 多实例 GPU (MIG) 实现细粒度资源隔离

2. **CPU 侧**：
   - 利用 AVX-512/AMX 指令集加速量化推理
   - 大页内存 (Huge Page) 减少 TLB miss
   - NUMA 感知的内存分配降低访问延迟

3. **互联侧**：
   - PCIe 5.0/6.0 提升带宽（64-128 GB/s）
   - CXL 3.0 实现内存池化和相干访问
   - NVLink 桥接多 GPU 减少 CPU 中转

#### 安全考量

异构调度系统特有的安全风险：

| 风险类型 | 描述 | 防护措施 |
|---------|------|---------|
| **数据泄露** | CPU 内存中的 KV Cache 可能被未授权访问 | 加密 offload 数据、内存隔离、安全擦除 |
| **侧信道攻击** | 通过访问模式推断敏感信息 | 恒定时间算法、访问模式模糊化 |
| **资源耗尽** | 恶意请求占用大量 offload 内存 | 请求配额限制、优先级调度、快速驱逐 |
| **模型窃取** | 通过时序分析推断模型结构 | 添加噪声延迟、批处理混淆 |
| **供应链风险** | 第三方 offload 库存在漏洞 | 代码审计、最小权限原则、签名验证 |

---

## 第二部分：行业情报

### 1. GitHub 热门项目（16 个）

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | 70,000+ | PagedAttention、Continuous Batching、CPU offload 支持 | Python/CUDA | 2026-04 | [GitHub](https://github.com/vllm-project/vllm) |
| **llama.cpp** | 65,000+ | 纯 CPU 推理优化、Metal/Vulkan 后端、GPU offload | C/CUDA | 2026-04 | [GitHub](https://github.com/ggml-org/llama.cpp) |
| **Ollama** | 55,000+ | 本地 LLM 运行、自动 CPU/GPU 调度 | Go/CUDA | 2026-04 | [GitHub](https://github.com/ollama/ollama) |
| **DeepSpeed** | 40,000+ | ZeRO-Offload、MoE 支持、训练/推理优化 | Python/CUDA | 2026-04 | [GitHub](https://github.com/microsoft/DeepSpeed) |
| **TensorRT-LLM** | 25,000+ | NVIDIA 高性能推理、In-flight Batching、多 GPU | C++/CUDA | 2026-04 | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **TGI** | 14,600+ | HuggingFace 官方、维护模式、推荐转向 vLLM/SGLang | Rust/Python | 2025-12 | [GitHub](https://github.com/huggingface/text-generation-inference) |
| **SGLang** | 15,000+ | 结构化生成、RadixAttention、Agentic 工作流标准 | Python/CUDA | 2026-04 | [GitHub](https://github.com/sgl-project/sglang) |
| **Petals** | 12,000+ | 分布式推理、BitTorrent 式模型并行 | Python/PyTorch | 2026-03 | [GitHub](https://github.com/bigscience-workshop/petals) |
| **MLC LLM** | 12,000+ | 跨平台编译、WebGPU 支持、移动端优化 | Python/TVM | 2026-04 | [GitHub](https://github.com/mlc-ai/mlc-llm) |
| **WebLLM** | 8,000+ | 浏览器内推理、WebGPU 加速、离线运行 | TypeScript | 2026-04 | [GitHub](https://github.com/mlc-ai/web-llm) |
| **IPEX-LLM** | 8,000+ | Intel XPU 加速、CPU-GPU 协同、MoE 优化（FlashMoE） | Python/C++ | 2026-04 | [GitHub](https://github.com/intel/ipex-llm) |
| **LMDeploy** | 6,000+ | 量化推理、多后端支持、高性能服务 | Python/C++ | 2026-04 | [GitHub](https://github.com/InternLM/lmdeploy) |
| **FasterTransformer** | 7,000+ | NVIDIA 优化、Transformer 加速、多 GPU | C++/CUDA | 2025-10 | [GitHub](https://github.com/NVIDIA/FasterTransformer) |
| **InferLLM** | 3,000+ | 轻量级 CPU 推理、嵌入式优化 | C/C++ | 2026-02 | [GitHub](https://github.com/MegEngine/InferLLM) |
| **ik_llama.cpp** | 2,000+ | llama.cpp 优化分支、CPU 性能提升、新量化 | C/C++ | 2026-04 | [GitHub](https://github.com/ikawrakow/ik_llama.cpp) |
| **Hetu** | 2,000+ | 万亿参数模型、分布式训练/推理 | Python/C++ | 2026-03 | [GitHub](https://github.com/pku-dair/hetu) |

**数据更新日期：** 2026-04-04
**数据来源：** GitHub API、WebSearch 搜索结果

---

### 2. 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|---------|------|
| **vLLM: Easy, Fast, and Cost-Effective LLM Serving** | Kwon et al., Stanford | 2023 | SOSP'23 | PagedAttention 内存管理，连续批处理 | 奠基性工作 | [arXiv:2309.06180](https://arxiv.org/abs/2309.06180) |
| **HeteGen: Heterogeneous Parallel Inference for LLMs** | Zhao et al. | 2024 | arXiv | 首个系统性 CPU-GPU 异构推理框架 | 奠基性工作 | arXiv:2403.01164 |
| **Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous Devices** | Zhang et al. | 2025 | ICLR'26 | 单设备 30-70B 模型异构推理、CPU-GPU 协同 | SOTA | [arXiv:2504.08791](https://arxiv.org/html/2504.08791v2) |
| **APEX: Asynchronous Parallel CPU-GPU Execution for LLM Inference** | Liu et al. | 2025 | arXiv | 异步并行执行、预填充 (CPU)+ 解码 (GPU) | SOTA | [arXiv:2506.03296](https://arxiv.org/html/2506.03296v4) |
| **Online Scheduling for LLM Inference with KV Cache Constraints** | MIT et al. | 2026 | arXiv | KV Cache 约束下的在线调度算法、理论分析 | 理论突破 | [arXiv:2502.07115](https://arxiv.org/html/2502.07115v5) |
| **SageSched: Efficient LLM Scheduling Confronting Demand Uncertainty** | Stanford et al. | 2026 | arXiv | 需求不确定性下的混合负载调度、SLO 保证 | 生产导向 | [arXiv:2603.07917](https://arxiv.org/html/2603.07917) |
| **Serving Hybrid LLM Loads with SLO Guarantees** | UC Berkeley | 2026 | arXiv | Llumnix 扩展、优先级调度、混合负载 SLO | 系统创新 | [arXiv:2603.12831](https://arxiv.org/html/2603.12831v2) |
| **HGCA: Hybrid GPU-CPU Attention for Long Context Inference** | Tsinghua | 2025 | arXiv | 长上下文混合注意力、CPU-GPU 协同计算 | 架构创新 | [arXiv:2507.03153](https://arxiv.org/html/2507.03153v1) |
| **Efficient CPU-GPU Collaborative Inference for MoE-based LLMs** | Alibaba | 2025 | arXiv | MoE 模型专家级卸载、消费级硬件推理 | 应用导向 | [arXiv:2512.16473](https://arxiv.org/html/2512.16473v1) |
| **Characterizing Mobile SoC for Heterogeneous LLM Inference** | Yonsei | 2025 | SOSP'25 | 移动 SoC 异构计算特性分析、NPU/GPU/CPU 协同 | SOSP 接收 | [arXiv:2501.14794](https://arxiv.org/html/2501.14794v2) |
| **LLMServingSim 2.0: Unified Simulator for Heterogeneous Serving** | CMU et al. | 2026 | arXiv | 异构和分离式 LLM 服务统一仿真器 | 工具创新 | [arXiv:2602.23036](https://arxiv.org/html/2602.23036v2) |
| **HillInfer: Efficient Long-Context LLM Inference on the Edge** | Edge AI Lab | 2026 | arXiv | 边缘长上下文推理、层预取、KV Cache 优化 | 边缘优化 | [arXiv:2602.18750](https://arxiv.org/html/2602.18750v2) |

**论文选择策略：**
- **经典高影响力（40%）：** vLLM、HeteGen 等奠基性研究
- **最新 SOTA（60%）：** 2025-2026 年的 Prima.cpp、APEX、SageSched 等前沿进展

---

### 3. 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| **LLM Inference Optimization: Tutorial & Best Practices** | LaunchDarkly | 英文 | 教程 | 量化、批处理、调度策略全解析 | 2026-01 | [链接](https://launchdarkly.com/blog/llm-inference-optimization/) |
| **Reducing LLM Inference Cost: A Practical Guide** | Medium/@vyaswanth965 | 英文 | 实践指南 | 推理成本优化、异构部署实战 | 2026-03 | [链接](https://medium.com/@vyaswanth965/reducing-llm-inference-cost) |
| **LLM Inference Optimization Techniques** | Clarifai | 英文 | 技术分析 | Batching、KV Cache、Speculative Decoding | 2025-09 | [链接](https://www.clarifai.com/blog/llm-inference-optimization/) |
| **The Engineering Guide to Efficient LLM Inference** | Towards AI | 英文 | 工程指南 | 指标、内存、数学原理深度解析 | 2025-11 | [链接](https://pub.towardsai.net/llm-inference-metrics) |
| **LLM Optimization Techniques and Guide** | Mirantis | 英文 | 部署指南 | 生产环境优化技术清单 | 2026-02 | [链接](https://www.mirantis.com/blog/llm-optimization-techniques/) |
| **掌握 LLM 技术：推理优化** | NVIDIA 中文博客 | 中文 | 官方教程 | Attention 优化、量化、批处理 | 2025-12 | [链接](https://developer.nvidia.cn/blog/mastering-llm-techniques) |
| **vLLM 源码解析之 continuous batch** | 知乎专栏 | 中文 | 源码分析 | Continuous Batching 实现细节 | 2025-10 | [链接](https://zhuanlan.zhihu.com/p/1914965829864362801) |
| **LLM 推理优化 Continuous Batching 及其实现** | 知乎专栏 | 中文 | 技术解析 | Continuous Batching 原理与实战 | 2025-11 | [链接](https://zhuanlan.zhihu.com/p/19696795081) |
| **2025 年大模型推理框架全解析** | CSDN | 中文 | 综述 | 主流推理框架横向对比 | 2025-12 | [链接](https://adg.csdn.net/696f3db9437a6b403369c108.html) |
| **SOSP 2025 论文评述：LLM Inference** | 知乎专栏 | 中文 | 论文解读 | SOSP 2025 异构推理论文深度分析 | 2025-11 | [链接](https://zhuanlan.zhihu.com/p/1961891859036086346) |

**博客选择标准：**
- 内容深度：优先系列文章、深度教程、架构解析
- 作者权威：官方团队博客、知名专家、一线工程师实践
- 语言平衡：英文约 70%，中文约 30%

---

### 4. 技术演进时间线

```
2022 年 ─┬─ Orca (Microsoft) → 提出迭代式调度和动态批处理概念
         │
2023 年 ─┼─ vLLM (Stanford) → PagedAttention 革命性内存管理，开启高效推理时代
         │
2023 年 ─┼─ DeepSpeed-ZeRO Offload → 训练时 offload 技术迁移到推理场景
         │
2024 年 ─┼─ HeteGen → 首个系统性 CPU-GPU 异构推理框架，开创协同调度研究方向
         │
2024 年 ─┼─ DistServe (PKU) → Prefill/Decode 解耦，异构部署新范式
         │
2024 年 ─┼─ Llumnix → 多实例动态调度、请求迁移，提升集群级资源利用率
         │
2024 年 ─┼─ TGI 进入维护模式 → HuggingFace 推荐用户转向 vLLM 和 SGLang
         │
2025 年 ─┼─ SGLang 崛起 → 结构化生成、RadixAttention，成为 Agentic 工作流事实标准
         │
2025 年 ─┼─ Prima.cpp (ICLR'26) → 单设备 30-70B 模型异构推理，推动边缘部署
         │
2025 年 ─┼─ APEX → 异步并行 CPU-GPU 执行，实现计算重叠优化
         │
2025 年 ─┼─ HGCA → 混合 GPU-CPU 注意力机制，支持长上下文高效推理
         │
2026 年 ─┼─ SageSched → 需求不确定性下的混合负载调度、SLO 保证
         │
2026 年 ─┼─ LLMServingSim 2.0 → 异构和分离式 LLM 服务统一仿真框架
         │
2026 年 ─┴─ 当前状态：CPU-GPU 异构调度从边缘场景扩展到数据中心，成为降本增效的核心技术
```

---

## 第三部分：方案对比

### 1. 历史发展时间线

```
2020 年 ─┬─ 早期探索 → 模型并行训练技术开始应用于推理场景
         │
2021 年 ─┼─ DeepSpeed ZeRO → 内存优化从训练延伸到推理
         │
2022 年 ─┼─ Orca 系统 → 首次系统化提出迭代式调度框架
         │
2023 年 ─┼─ vLLM PagedAttention → GPU 利用率提升至 90%+
         │
2024 年 ─┼─ HeteGen → CPU 主动参与计算，而非仅用于 offload
         │
2025 年 ─┼─ APEX 异步并行 → CPU 和 GPU 真正并行执行，而非简单的串行 offload
         │
2025 年 ─┼─ Prima.cpp → 面向资源受限设备的 30-70B 模型推理
         │
2026 年 ─┼─ SageSched → 在线 + 离线请求统一调度、SLO 保证、需求不确定性处理
         │
2026 年 ─┴─ 当前状态：异构调度从"能运行大模型"演进为"高效服务混合负载"
```

---

### 2. 六种方案横向对比

| 方案 | 原理 | 优点（3+） | 缺点（3+） | 适用场景 | 成本量级 |
|------|------|-----------|-----------|---------|---------|
| **纯 GPU 推理**<br>(vLLM/TensorRT-LLM) | 所有计算在 GPU 执行，CPU 仅负责调度和 I/O | • 延迟最低<br>• 生态最成熟<br>• 优化最充分 | • 显存受限<br>• 大模型无法运行<br>• 成本高 | 在线低延迟服务<br>中小模型生产部署 | $$$$<br>(高端 GPU) |
| **CPU Offload**<br>(ZeRO-Inference) | KV Cache 和/或模型参数卸载到 CPU 内存 | • 支持更大模型<br>• 利用廉价 CPU 内存<br>• 实现简单 | • PCIe 带宽瓶颈<br>• 延迟增加 2-5 倍<br>• CPU 计算闲置 | 内存受限场景<br>离线批处理 | $$<br>(中端 GPU+ 大内存) |
| **HeteGen 异构协同**<br>(Zhao et al. 2024) | CPU 主动参与计算，与 GPU 并行执行不同层 | • 计算资源充分利用<br>• 支持 30B+ 模型单卡运行<br>• 吞吐提升 1.5-2 倍 | • 调度复杂度高<br>• 需要细粒度算子支持<br>• 同步开销需优化 | 边缘设备<br>消费级 GPU 推理 | $$<br>(消费级硬件) |
| **APEX 异步并行**<br>(Liu et al. 2025) | 预填充 (CPU) 和解码 (GPU) 异步并行执行 | • 延迟降低 30-50%<br>• 计算重叠率高<br>• 在线推理优化 | • 实现复杂<br>• 需要请求级流水线<br>• 内存管理开销 | 在线交互式服务<br>低延迟要求场景 | $$$<br>(中高端 GPU) |
| **MoE 专家级调度**<br>(Alibaba 2025) | 仅激活专家在 GPU，非激活专家卸载到 CPU | • 671B 模型可单卡运行<br>• 显存需求降低 80%+<br>• 计算高效 | • 仅适用于 MoE 模型<br>• 专家路由开销<br>• 负载不均衡风险 | MoE 模型服务<br>超大模型部署 | $$<br>(消费级/中端 GPU) |
| **混合负载调度**<br>(SageSched 2026) | 在线请求优先，离线请求填充空闲资源 | • 资源利用率最大化<br>• SLO 有保证<br>• 成本最优 | • 调度算法复杂<br>• 需要负载预测<br>• 实现难度大 | 多租户云平台<br>混合业务场景 | $$$<br>(数据中心级) |

---

### 3. 技术细节对比

| 维度 | 纯 GPU | CPU Offload | HeteGen | APEX | MoE 专家调度 | 混合负载调度 |
|------|--------|-------------|---------|------|-------------|-------------|
| **性能** | 最优 | 中等 | 良好 | 优秀 (在线) | 良好 (MoE) | 良好 |
| **易用性** | 高 | 中 | 低 | 低 | 中 | 低 |
| **生态成熟度** | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★☆☆☆ |
| **社区活跃度** | 极高 | 中等 | 低 | 中等 | 中等 | 中等 |
| **学习曲线** | 平缓 | 中等 | 陡峭 | 陡峭 | 中等 | 陡峭 |
| **PCIe 依赖** | 低 | 高 | 中 | 中 | 中 | 中 |
| **量化支持** | 完善 | 完善 | 部分 | 部分 | 完善 | 部分 |
| **长上下文** | 受显存限制 | 支持 | 支持 | 支持 | 支持 | 支持 |

---

### 4. 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证** | llama.cpp + ik_llama.cpp 分支 | • 开箱即用<br>• CPU 可运行<br>• 社区资源丰富<br>• 零部署成本 | $0-50<br>(本地/免费云) |
| **在线 API 服务 (<100 QPS)** | vLLM (纯 GPU) | • 延迟最优<br>• 生态成熟<br>• 运维简单<br>• 文档完善 | $500-2000<br>(单卡 A10/A100) |
| **中型生产环境 (100-1000 QPS)** | vLLM + CPU Offload 混合 | • 平衡延迟和成本<br>• 支持中等模型 (13-30B)<br>• 可扩展性强 | $2000-10000<br>(多卡 A100/H100) |
| **边缘设备部署** | HeteGen / Prima.cpp | • 单设备运行大模型<br>• 低功耗优化<br>• 离线可用 | $200-500<br>(消费级硬件) |
| **超大模型服务 (70B+)** | MoE 专家调度 + Intel IPEX-LLM | • 671B 模型可运行<br>• 显存需求低<br>• 利用 Intel 生态 | $5000-20000<br>(多卡 A100/Intel GPU) |
| **多租户云平台** | SageSched 混合负载调度 | • 资源利用率最大化<br>• SLO 保证<br>• 成本最优 | 按用量计费<br>(TCO 降低 30-50%) |
| **MoE 模型专用** | Alibaba MoE 调度方案 | • 专家级优化<br>• 激活参数仅占小部分<br>• 高效推理 | $1000-5000<br>(中端 GPU 集群) |
| **长上下文推理** | HGCA 混合注意力 | • CPU-GPU 协同计算注意力<br>• 支持 100K+ 上下文<br>• 内存效率高 | $3000-15000<br>(大显存 GPU) |

**成本说明：**
- 成本估算基于 2026 年云平台价格（AWS/GCP/Azure）
- 包含 GPU 实例、内存、存储和网络费用
- 未计入人力运维成本

---

## 第四部分：精华整合

### 1. The One 公式

$$\text{CPU-GPU 异构推理} = \underbrace{\text{GPU 稠密计算}}_{\text{高吞吐}} + \underbrace{\text{CPU 弹性扩展}}_{\text{大容量}} - \underbrace{\text{PCIe 传输 + 同步开销}}_{\text{协调成本}}$$

**心智模型：** 异构推理的本质是用 GPU 的"快"和 CPU 的"大"进行互补，代价是两者之间的协调成本。成功的调度就是让收益远大于开销。

---

### 2. 一句话解释

> CPU-GPU 异构推理就像让一位速算专家（GPU）和一位记忆力超群但计算较慢的助手（CPU）合作解题：专家负责核心计算，助手负责准备数据和存储中间结果，两人并行工作时效率最高。

---

### 3. 核心架构图

```
        推理请求
           │
           ▼
    ┌──────────────┐
    │  调度决策器   │ ← 分析请求特征、硬件状态、SLO 要求
    └──────┬───────┘
           │
    ┌──────┴───────┬────────────────┐
    │              │                │
    ▼              ▼                ▼
┌─────────┐  ┌───────────┐   ┌────────────┐
│ 预填充   │  │  KV Cache  │   │  解码生成   │
│  (CPU)  │  │  管理层    │   │  (GPU)     │
└────┬────┘  └─────┬─────┘   └─────┬──────┘
     │            │               │
     │   异步并行  │   分页管理     │  Continuous
     │   (APEX)   │  (Paged)      │  Batching
     ▼            ▼               ▼
    ┌─────────────────────────────────┐
    │          结果合并与输出          │
    └─────────────────────────────────┘
           │
           ▼
        生成文本

关键指标：TTFT < 100ms | 吞吐 > 1000 tokens/s | 显存利用率 > 85%
```

---

### 4. STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation（背景 + 痛点）** | 大模型推理面临两大核心挑战：一是显存墙问题——70B+ 模型的 KV Cache 和参数远超单卡显存容量；二是成本压力——纯 GPU 方案在高并发场景下 TCO 过高。传统 offload 方案将 CPU 视为被动存储，未能充分利用其计算能力。同时，在线服务的低延迟要求与离线批处理的高吞吐需求难以在同一系统中共存。 |
| **Task（核心问题）** | 如何在单节点内实现 CPU-GPU 的真正协同计算，而非简单的数据卸载？关键约束包括：PCIe 带宽有限（32-64 GB/s）、调度决策延迟需低于 1ms、SLO 保证需达到 99% 以上、支持从 7B 到 671B 的全尺寸模型。 |
| **Action（主流方案）** | 技术演进经历三个阶段：第一阶段（2023）以 vLLM 为代表，通过 PagedAttention 和 Continuous Batching 优化纯 GPU 执行；第二阶段（2024）HeteGen 开创 CPU 主动参与计算的范式；第三阶段（2025-2026）APEX 实现异步并行、SageSched 支持混合负载、Prima.cpp 优化边缘部署。核心突破包括：计算重叠（CPU 预填充与 GPU 解码并行）、专家级调度（MoE 模型激活专家在 GPU）、动态优先级调度（在线请求优先）。 |
| **Result（效果 + 建议）** | 当前技术可实现：671B MoE 模型单卡运行（Intel IPEX-LLM）、在线推理延迟降低 30-50%（APEX）、混合负载资源利用率提升 40%（SageSched）。局限性在于：调度算法复杂度高、PCIe 仍是瓶颈、生态碎片化。实操建议：小型项目用 llama.cpp，在线服务首选 vLLM，超大模型采用 MoE 专家调度，多租户平台引入混合负载调度。 |

---

### 5. 理解确认问题

**问题：** 为什么在 MoE（Mixture of Experts）模型中，CPU-GPU 异构调度相比稠密模型能获得更高的收益？请从计算特性和内存访问模式两个角度分析。

**参考答案：**

MoE 模型的核心特性是**稀疏激活**——每层仅激活 Top-K 个专家（通常 K=2），而模型可能包含数十甚至上百个专家。这带来两个优势：

1. **计算特性角度：**
   - 稠密模型所有参数都需计算，CPU 参与会拖慢整体速度
   - MoE 模型仅激活专家需要 GPU 计算，非激活专家可卸载到 CPU 异步处理
   - 对于 671B 的 DeepSeek-V3，每 token 仅激活约 37B 参数，其余 634B 可安全 offload

2. **内存访问模式角度：**
   - MoE 的专家路由具有**可预测性**——在计算前即可确定哪些专家被激活
   - 这允许提前将激活专家加载到 GPU，非激活专家保留在 CPU 内存
   - 相比之下，稠密模型的逐层计算难以预测哪些部分可 offload

因此，MoE 模型的异构调度可实现**显存需求降低 80%+** 而性能损失极小，这是稠密模型无法达到的收益水平。

---

## 附录：关键术语表

| 术语 | 定义 |
|------|------|
| **TTFT** | Time To First Token，首字延迟 |
| **TPOT** | Time Per Output Token，每输出令牌延迟 |
| **Continuous Batching** | 迭代级动态批处理，替代传统请求级批处理 |
| **PagedAttention** | 分页 KV Cache 管理，解决内存碎片化 |
| **KV Cache** | 注意力机制的键值缓存，存储历史 token 信息 |
| **MoE** | Mixture of Experts，稀疏激活的专家混合模型 |
| **Offload** | 将数据或计算从 GPU 卸载到 CPU |
| **SLO** | Service Level Objective，服务等级目标 |
| **TCO** | Total Cost of Ownership，总体拥有成本 |
| **APEX** | Asynchronous Parallel CPU-GPU Execution，异步并行执行方案 |

---

## 参考文献

### 核心论文
1. Kwon et al. "vLLM: Easy, Fast, and Cost-Effective LLM Serving." SOSP 2023.
2. Zhao et al. "HeteGen: Heterogeneous Parallel Inference for LLMs." arXiv 2024.
3. Zhang et al. "Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous Devices." ICLR 2026.
4. Liu et al. "APEX: Asynchronous Parallel CPU-GPU Execution for LLM Inference." arXiv 2025.
5. Stanford et al. "SageSched: Efficient LLM Scheduling Confronting Demand Uncertainty." arXiv 2026.

### 开源项目
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [SGLang](https://github.com/sgl-project/sglang)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [IPEX-LLM](https://github.com/intel/ipex-llm)

### 技术博客
- [vLLM Blog](https://blog.vllm.ai/)
- [NVIDIA Developer Blog](https://developer.nvidia.com/)
- [LaunchDarkly - LLM Inference Optimization](https://launchdarkly.com/blog/llm-inference-optimization/)

---

**报告版本：** 2.0
**调研完成日期：** 2026-04-04
**总字数：** 约 9,000 字
**数据来源：** GitHub、arXiv、技术博客、会议论文（2024-2026）
