# 大模型推理框架算子自动调优技术 — 深度调研报告

> 调研日期：2026-05-19 | 调研域：大模型框架

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

**大模型推理框架算子自动调优技术**，是指在大语言模型（LLM）推理过程中，对计算图中的算子（Kernel/Operator）自动进行硬件感知的最优实现选择、参数配置、融合策略和代码生成的一系列技术。其核心目标是：在给定硬件平台、模型架构和推理场景（prefill/decode/batch size）的条件下，自动搜索出延迟最低或吞吐最高的算子执行方案，无需人工手工调优。

### 常见误解

1. **"自动调优=自动选择 tile size"**：实际上自动调优的搜索空间远大于 tile size，还涵盖 kernel fusion 策略、warp 数量、流水线 stage 数、共享内存分配策略、向量化宽度、算子拆分方式等多个维度。
2. **"算子自动调优就是 AutoTVM/Ansor 那一套"**：传统 TVM 体系的自动调优面向通用神经网络，而 LLM 推理场景有其特殊性——自回归解码的 sequence length 动态变化、KV cache 的内存管理模式、attention 算子的特殊计算模式，这些都需要专门设计。
3. **"自动调优一次后结果可以永久复用"**：调优结果强依赖硬件架构（SM version、shared memory 大小）、模型结构（hidden size、num_heads、intermediate size）和推理配置（batch size、sequence length），任何参数变化都可能导致最优策略失效。
4. **"LLM 推理的算子自动调优已基本成熟，无需持续投入"**：事实上，随着 MoE 架构、长上下文推理、Blackwell 等新硬件的快速演进，算子自动调优仍是一个高度活跃的开放问题。

### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **算子自动调优 vs. 模型量化** | 量化改变数值精度（INT8/FP4 等）以降低计算和存储开销；自动调优在不改变精度的前提下优化算子实现方式。二者可协同使用。 |
| **算子自动调优 vs. 图优化** | 图优化在计算图层面做算子重排、常量折叠、dead code elimination；自动调优在图优化的基础上，针对每个算子的具体实现做硬件级别的搜索。 |
| **算子自动调优 vs. 编译优化** | 编译优化（如 LLVM pass）处理底层指令选择和寄存器分配；算子自动调优处理更高层的 kernel 实现选择（如不同 tile 策略、融合策略）。 |
| **算子自动调优 vs. 推理框架调度** | 调度负责请求排队、batching、内存管理等运行时决策；自动调优关注单个算子的最优实现，通常在编译期或部署前完成。 |

## 1.2 核心架构

下面展示 LLM 推理算子自动调优系统的通用架构：

```
┌──────────────────────────────────────────────────────────────────────┐
│                    LLM 推理算子自动调优系统架构                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  [模型计算图]                                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                 调优引擎 (Tuning Engine)                  │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │       │
│  │  │ 搜索策略 │→ │ 代码生成 │→ │ 编译执行 │→ │ 性能评估 │  │       │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │       │
│  └──────────────────────────────────────────────────────────┘       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │                 调优数据库 (Tuning Database)               │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │       │
│  │  │ 硬件特征 │  │ 算子特征 │  │ 最优配置 │  │ 性能历史 │  │       │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │       │
│  └──────────────────────────────────────────────────────────┘       │
│       │                                                             │
│       ▼                                                             │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │          推理运行时 (Inference Runtime)                    │       │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐  │       │
│  │  │ 图优化   │→ │ 算子选择 │→ │ kernel 执│→ │ 性能监控│  │       │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────┘  │       │
│  └──────────────────────────────────────────────────────────┘       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

**各组件职责说明：**

| 组件 | 功能 |
|------|------|
| **调优引擎** | 核心搜索组件，负责在庞大的配置空间中寻找最优算子实现 |
| **搜索策略** | 决定搜索方向，支持网格搜索/贝叶斯优化/强化学习/LLM Agent 等方法 |
| **代码生成** | 根据配置生成特定硬件平台的 kernel 代码（CUDA/Triton/AscendC 等） |
| **编译执行** | 调用底层编译器（nvcc/triton-C/MLIR）编译并执行 kernel |
| **性能评估** | 在目标硬件上 profiling 获取延迟/吞吐/共享内存用量等指标 |
| **调优数据库** | 存储算子-硬件-配置的映射关系，支持增量调优和缓存复用 |
| **推理运行时** | 基于调优结果实际执行推理，包含图优化和算子选择模块 |

## 1.3 数学形式化

### 公式 1：算子调优的搜索问题定义

$$
\min_{c \in \mathcal{C}(o, h)} \mathcal{L}(o, c, h, w)
$$

其中 $o$ 为算子类型，$h$ 为硬件特征，$w$ 为推理负载特征（batch size, seq len），$\mathcal{C}(o, h)$ 为给定算子-硬件下的配置空间，$\mathcal{L}$ 为延迟函数。目标是找到使延迟最小的配置 $c^*$。

### 公式 2：Operator Fusion 的收益模型

$$
\Delta T_{\text{fusion}} = T_{\text{launch\_overhead}} + T_{\text{mem\_savings}} - T_{\text{compute\_overhead}}
$$

其中 $T_{\text{launch\_overhead}}$ 为消除的 kernel launch 开销，$T_{\text{mem\_savings}}$ 为减少的 HBM 读写时间，$T_{\text{compute\_overhead}}$ 为融合后可能引入的额外计算（如更复杂的索引逻辑）。融合决策即判断 $\Delta T_{\text{fusion}} > 0$。

### 公式 3：MegaKernel 的共享内存约束模型

$$
\sum_{k=1}^{K} S_k(\text{tile}_k) \leq S_{\text{max}}(h)
$$

其中 $K$ 为融合到单个 MegaKernel 中的算子数，$S_k$ 为第 $k$ 个算子所需共享内存（依赖于 tile 参数），$S_{\text{max}}(h)$ 为硬件 $h$ 的共享内存上限（如 H100 为 227 KB，Ada L20 为 128 KB）。Ada-MK 通过 K-dimension 拆分将峰值内存占用降低 50%。

### 公式 4：Autotuning 的搜索空间复杂度

$$
|\mathcal{C}| = \prod_{d=1}^{D} N_d(o, h)
$$

其中 $D$ 为可调优维度数（tile size、warp count、pipeline stage、vectorize width 等），$N_d$ 为每个维度的候选值数量。典型 LLM attention 算子的搜索空间可达 $10^6$ 量级，穷举搜索不可行。

### 公式 5：基于 LLM Agent 的算子生成质量指标

$$
\text{fast}_p = \frac{|\{k \in \mathcal{K}_{\text{gen}} : \text{correct}(k) \land \text{speedup}(k) > p\}|}{|\mathcal{K}_{\text{gen}}|}
$$

KernelBench 提出的指标，$\mathcal{K}_{\text{gen}}$ 为 LLM 生成的 kernel 集合，$\text{correct}(k)$ 验证数值正确性，$\text{speedup}(k)$ 衡量相对 PyTorch 基线加速比，$p$ 为加速比阈值。当前 SOTA 模型在 Level-1（单算子）的 $\text{fast}_1$ 约为 40-60%。

## 1.4 实现逻辑（Python 伪代码）

```python
class OperatorAutoTuner:
    """算子自动调优系统的核心抽象"""

    def __init__(self, hardware_profile: dict, db_path: str = None):
        self.hardware = hardware_profile       # GPU型号、SM版本、共享内存大小等
        self.db = TuningDatabase(db_path)      # 调优结果持久化存储
        self.search_strategies = {
            "grid": GridSearch(),
            "bayesian": BayesianOptimization(),
            "rl": RLSearch(),
            "llm_agent": LLMAgentSearch()
        }

    def tune(self, graph: ComputationGraph,
             strategy: str = "bayesian",
             max_iters: int = 100) -> TuningResult:
        """对计算图中的所有算子执行自动调优"""
        results = {}
        for op_node in graph.operators():
            # 先检查缓存
            cached = self.db.lookup(op_node.signature(), self.hardware)
            if cached and cached.is_valid():
                results[op_node.id] = cached
                continue

            # 定义搜索空间
            search_space = self._build_search_space(op_node)

            # 执行搜索
            best_config = self.search_strategies[strategy].search(
                objective=lambda config: self._benchmark_kernel(op_node, config),
                space=search_space,
                max_iters=max_iters
            )

            # 缓存结果
            results[op_node.id] = best_config
            self.db.save(op_node.signature(), self.hardware, best_config)

        return TuningResult(results)

    def _build_search_space(self, op_node) -> SearchSpace:
        """根据算子类型和硬件特征构建搜索空间"""
        if op_node.type == OperatorType.MATMUL:
            return SearchSpace({
                "BLOCK_M": [16, 32, 64, 128],
                "BLOCK_N": [16, 32, 64, 128],
                "BLOCK_K": [16, 32, 64],
                "num_warps": [4, 8, 16],
                "num_stages": [2, 3, 4],
            })
        elif op_node.type == OperatorType.ATTENTION:
            return SearchSpace({
                "BLOCK_Q": [16, 32, 64],
                "BLOCK_KV": [32, 64, 128],
                "num_warps": [4, 8],
                "use_3d_grid": [True, False],
            })
        # ... 更多算子类型

    def _benchmark_kernel(self, op_node, config) -> float:
        """编译并运行 kernel，返回延迟（ms）"""
        source = CodeGenerator.generate(op_node, config)
        kernel = Compiler.compile(source, self.hardware)
        latencies = Profiler.benchmark(kernel, warmup=10, iter=100)
        return statistics.median(latencies)


class CodeGenerator:
    """从算子描述和配置生成硬件 kernel 代码"""

    @staticmethod
    def generate(op_node: OpNode, config: dict) -> str:
        """根据算子类型选择代码模板并填充配置"""
        template = TemplateRegistry.lookup(op_node.type)
        if template.dsl == "triton":
            return TritonTemplate.render(template, config)
        elif template.dsl == "cuda":
            return CUDATemplate.render(template, config)
        elif template.dsl == "ascendc":
            return AscendCTemplate.render(template, config)
```

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **Kernel 延迟** | < 0.5 ms (decode) | NVIDIA Nsight Compute / Triton do_bench | 单个算子的端到端执行时间，decode 场景对延迟敏感 |
| **推理吞吐** | > 3000 tok/s (7B) | 压力测试，固定 batch | 衡量推理框架处理请求的速率，受算子调优直接影响 |
| **TTFT（首Token延迟）** | < 200 ms | 端到端基准测试 | Prefill 阶段算子调优的结果，影响用户感知 |
| **ITL（Token间延迟）** | < 20 ms/token | 端到端基准测试 | Decode 阶段算子调优结果，影响流式响应体验 |
| **Shared Memory 占用** | < 128 KB (Ada) / < 227 KB (Hopper) | 编译期静态分析 / profiling | 决定 MegaKernel 融合策略可行性 |
| **调优收敛时间** | < 30 min | 端到端计时 | 影响部署效率，LLM Agent 方法可将调优耗时从小时级降至分钟级 |
| **fast_p 得分** | > 60% (Level-1) | KernelBench 基准测试 | LLM 生成的 kernel 中加速比超过 p 的比例 |

## 1.6 扩展性与安全性

### 水平扩展

- **多 GPU 并行调优**：TensorRT-LLM 的 AutoTuner 支持 PARALLEL 分布式策略，不同 GPU 分别 profiling 不同的 tactic 子集，收敛后合并最优结果。
- **离线调优 + 在线部署分离**：FlashInfer 提出的 Config Caching 机制支持在强大的集群上完成调优后，将结果 JSON 分发到生产节点复用。
- **分批渐进调优**：IBM triton-dejavu 支持增量调优，每次 session 增量 profiling 新 shape，逐步完善覆盖度。

### 垂直扩展

- **ML 辅助搜索空间剪枝**：Triton RFC #9308 提出使用决策树/随机森林预测最优配置区域，大幅减少需要实际 profiling 的候选数。
- **性能模型替代实际 profiling**：AIConfigurator 使用基于分析建模的方法，将推理分解为 GEMM、Attention、通信等可建模原语，无需 GPU profiling 即可预测性能。
- **持久化缓存去重**：通过算子签名哈希匹配，避免重复调优相同配置（TensorRT-LLM ProfilingCache、FlashInfer ConfigCache）。

### 安全考量

- **代码生成安全性**：LLM Agent 生成的 kernel 代码可能存在数值错误、死循环或越界访问，需运行时验证（如 KernelBench 的 correctness checker）。
- **调优缓存污染**：当模型/硬件配置变化时，过期的调优缓存可能导致性能回退，需要缓存失效机制（基于硬件 + Triton + PyTorch 版本的复合哈希）。
- **Profiling 数据泄露**：生产环境的 profiling 数据可能暴露模型结构信息，需要脱敏或访问控制。
- **编译时后门风险**：自动生成的 kernel 代码若被注入恶意代码，可能绕过传统安全审查，需要沙箱编译和静态分析。

---

# 第二部分：行业情报

> 数据采集时间：2026-05-19 | 所有数据均通过 WebSearch/WebFetch 实时获取

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **triton-lang/triton** | ~19,000 | LLM 推理/训练 kernel 语言与编译器，提供 `triton.autotune` 自动调优装饰器 | Python/C++/LLVM | 2026-05 (v3.5.1) | [GitHub](https://github.com/triton-lang/triton) |
| **vllm-project/vllm** | ~42,000 | 高吞吐 LLM 推理引擎，支持 -O0 到 -O3 四级优化，内置 FlashInfer/PagedAttention 等定制 kernel | Python/CUDA/Triton | 2026-04 | [GitHub](https://github.com/vllm-project/vllm) |
| **NVIDIA/TensorRT-LLM** | ~13,600 | NVIDIA 官方 LLM 推理框架，内建 AutoTuner 插件系统和 TunableRunner 架构 | C++/CUDA/TRT | 2026-05 (v1.2.0rc) | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| **flashinfer-ai/flashinfer** | ~5,600 | LLM 推理专用 kernel 库，支持 Paged/Ragged Attention 和 MoE 算子的 autotuning | C++/CUDA/Triton | 2026-05 (v0.6.11) | [GitHub](https://github.com/flashinfer-ai/flashinfer) |
| **sgl-project/sglang** | ~8,500 | 前端语言+运行时系统，深度集成 FlashInfer autotuning，与 NVIDIA 联合规划算子融合 roadmap | Python/CUDA/Triton | 2026-04 | [GitHub](https://github.com/sgl-project/sglang) |
| **kvcache-ai/ktransformers** | ~3,200 | CPU-GPU 异构推理框架，含 kt-kernel 自动调优子系统和 MoE 专家调度优化 | Python/C++/CUDA/AMX | 2026-05 (v0.5.1) | [GitHub](https://github.com/kvcache-ai/KTransformers) |
| **mirage-project/mirage** | ~2,100 | 自动将多 GPU LLM 推理编译成单个 MegaKernel，1.2-6.7x 延迟降低 | C++/Triton | 2026-02 (OSDI'25) | [GitHub](https://github.com/jamestiotio/mpk) |
| **IBM/triton-dejavu** | ~450 | Triton autotuner 持久化缓存框架，支持 Bayes 优化加速搜索 | Python | 2025-09 | [GitHub](https://github.com/IBM/triton-dejavu) |
| **pytorch/helion** | ~762 | PyTorch 的 Python 嵌入式 DSL，包含 ML 驱动的 autotuner（神经网络/决策树） | Python/C++ | 2026-01 | [GitHub](https://github.com/pytorch/helion) |
| **ScalingIntelligence/KernelBench** | ~800 | LLM kernel 生成能力的标准化评测基准（250 个 workload，3 级难度） | Python/CUDA | 2025-12 (v0.1) | [GitHub](https://github.com/orgs/ScalingIntelligence/repositories) |
| **Facebook/KernelLLM** | ~1,200 | Meta 开源的 8B 参数 Triton kernel 生成模型，单发超越 GPT-4o 和 DeepSeek V3 | Python | 2025-06 | [HuggingFace](https://huggingface.co/facebook/KernelLLM) |
| **RLsys-Foundation/TritonForge** | ~350 | SFT+RL 两阶段 Triton kernel 生成框架，支持多轮编译反馈迭代 | Python | 2025 | [GitHub](https://github.com/RLsys-Foundation/TritonForge) |
| **flagos-ai/awesome-LLM-driven-kernel-generation** | ~280 | LLM 驱动的 kernel 自动生成综述论文的配套资源列表 | - | 2026-01 | [GitHub](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation) |
| **flashinfer-ai/flashinfer-bench-starter-kit** | ~111 | MLSys 2026 AI kernel 生成竞赛的官方 starter kit | Python | 2026 | [GitHub](https://github.com/flashinfer-ai/flashinfer-bench-starter-kit) |
| **apollosenvy/kernel-anvil** | ~80 | 面向 AMD RDNA3 的 profile-guided GPU kernel 自调优器，2.25x decode 加速 | Python/C++ | 2026-03 | [GitHub](https://github.com/apollosenvy/kernel-anvil) |

## 2.2 关键论文（16 篇）

### 经典高影响力（奠基性工作，约 40%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 |
|------|----------|------|----------|---------|--------|
| **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** | Dao et al. (Stanford) | 2022 | NeurIPS | 提出了 IO-aware 的 attention 算子实现，tiling 和 recompute 策略的调优成为后续工作的基础 | 8000+ 引用 |
| **PagedAttention: Virtual Memory for LLM Serving** | Kwon et al. (Stanford) | 2023 | SOSP | 将虚拟内存思想引入 KV cache 管理，定制 GPU kernel 实现非连续块访问 | 1500+ 引用 |
| **Ansor: Generating High-Performance Tensor Programs for Deep Learning** | Zheng et al. (MIT) | 2020 | OSDI | 基于 hierarchical 搜索的自动调优框架，LLM 推理场景的调优可复用其分层搜索思想 | 800+ 引用 |
| **FlashDecoding: Fast and Efficient LLM Inference** | Dao et al. (Stanford) | 2023 | arXiv | 在 context 维度增加并行性，为 decode 阶段的长上下文 attention 提供高效的 3D 并行 kernel | 500+ 引用 |
| **TASO: Optimizing Deep Learning Computation with Automatic Graph Optimizations** | Jia et al. (Stanford) | 2019 | OSDI | 基于图等价性搜索的自动算子融合框架，启发了后续的 DAG 级融合搜索 | 400+ 引用 |
| **Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations** | Tillet et al. (OpenAI) | 2019 | PLDI | 提出了基于 tile 的 kernel 编程模型，`triton.autotune` 成为 LLM 推理算子调优的事实标准 | 600+ 引用 |

### 最新 SOTA 论文（前沿进展，约 60%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 链接 |
|------|----------|------|----------|---------|------|
| **Ada-MK: Adaptive MegaKernel Optimization via Automated DAG-based Search for LLM Inference** | Dong et al. (百度) | 2026.05 | arXiv | 基于 MLIR 细粒度 DAG 离线搜索，首个 MegaKernel 工业部署，单 batch 吞吐比 TRT-LLM 高 23.6% | [arXiv](https://arxiv.org/abs/2605.11581) |
| **Making LLMs Optimize Multi-Scenario CUDA Kernels Like Experts (CUDAMaster)** | Han et al. (清华大学) | 2026.03 | arXiv | 多 Agent + 硬件感知 CUDA 调优系统，MSKernelBench 基准，超越 Astra ~35% | [arXiv](https://arxiv.org/abs/2603.07169) |
| **AdaFuse: Accelerating Dynamic Adapter Inference via Token-Level Pre-Gating and Fused Kernel Optimization** | - | 2026.03 | AAAI 2026 | Token 级预门控 + 融合 CUDA kernel，LoRA 适配器推理解码延迟降低 2.4x | [arXiv](https://arxiv.org/abs/2603.11873) |
| **AIConfigurator: Lightning-Fast Configuration Optimization for Multi-Framework LLM Serving** | - | 2026.01 | arXiv | 框架无关的性能建模，无需 GPU profiling 的配置搜索，稠密模型提速 40%，MoE 提速 50% | [arXiv](https://arxiv.org/abs/2601.06288) |
| **Towards Automated Kernel Generation in the Era of LLMs: A Survey** | Yu et al. | 2026.01 | arXiv | 首个 LLM 驱动的 kernel 自动生成领域综述，覆盖 SFT/RL/多 Agent/硬件 profiling 等方向 | [arXiv](https://arxiv.org/abs/2601.15727) |
| **KernelBench: Can LLMs Write Efficient GPU Kernels?** | Ouyang et al. (Stanford) | 2025.02 | arXiv | 250 个 workload 的 LLM kernel 生成基准，定义 fast_p 评估指标，推动 LLM-driven 调优评估标准化 | [arXiv](https://arxiv.org/abs/2502.10517) |
| **STARK: Strategic Team of Agents for Refining Kernels** | - | 2025.10 | arXiv | Plan-Code-Debug 多 Agent GPU kernel 优化，KernelBench 上最高 16x 加速 | [arXiv](https://arxiv.org/abs/2510.16996) |
| **PRAGMA: A Profiling-Reasoned Multi-Agent Framework for Automatic Kernel Optimization** | - | 2025.11 | arXiv | Profile-guided 多 Agent 框架，CPU 2.81x / GPU 2.30x 加速 | [arXiv](https://arxiv.org/abs/2511.06345) |
| **AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization** | Zhang et al. (AWS) | 2025.11 | arXiv | 自改进 LLM Agent 优化 AWS Trainium kernel，beam search + 优化记忆，26x 低成本的 Sonnet 4 级效果 | [arXiv](https://arxiv.org/abs/2511.15915) |
| **Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs** | Cheng et al. | 2025.12 | OSDI 2025 / arXiv | 端到端多 GPU LLM 推理 MegaKernel 编译器，1.2-6.7x 延迟降低 | [arXiv](https://arxiv.org/abs/2512.22219) |

## 2.3 系统化技术博客（12 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|---------|----------|------|------|---------|------|
| **VLLM-X 高性能推理引擎：深度适配与优化实践** | 百度开发者平台 | 中文 | 工程实践 | 动态算子融合三阶段策略，基于强化学习的自动调优，混合精度 2.8-3.5x 提升 | 2026.05 |
| **AKG Agent：大模型驱动的算子自动生成技术探索** | 昇思 MindSpore 技术博客 | 中文 | 架构解析 | LLM 多 Agent（Designer/Coder/Verifier）协作框架，Skill/Trace/Evolve 系统 | 2026.04 |
| **SGLang NVIDIA Collaboration Roadmap (2026 Q1)** | SGLang GitHub Issue | 英文 | 路线图 | FlashInfer MoE autotuning、算子融合（RoPE+Q+Cache）、Blackwell SM10x 支持 | 2026.01 |
| **GEAK: Introducing Triton Kernel AI Agent & Evaluation Benchmarks** | AMD ROCm Blog | 英文 | 工程实践 | AMD Instinct GPU 的 AI Agent kernel 优化，GEAK-OptimAgent v2 平均 3.32x 加速 | 2025 |
| **Enabling vLLM V1 on AMD GPUs With Triton** | PyTorch Blog | 英文 | 深度教程 | Triton 自定义统一 attention kernel，warp 缩减 3-5x 加速，GQA 打包 25% 吞吐提升 | 2025 |
| **vLLM Optimization Levels Documentation** | vLLM Docs | 英文 | 官方文档 | -O0 到 -O3 四级优化级别定义，CUDAGraph 模式，算子融合配置 | 2025-2026 |
| **TensorRT-LLM AutoTuner Architecture** | NVIDIA/TensorRT-LLM DeepWiki | 英文 | 架构解析 | TunableRunner 体系、ProfilingCache、分布式调优策略（BROADCAST/INDEPENDENT/PARALLEL/MERGE） | 2025-2026 |
| **KernelBench v0.1 Blog** | Stanford Scaling Intelligence Lab | 英文 | 基准介绍 | 250 workload 评测结果，anti-reward-hacking 设计，rand_mix 分布测试 | 2025.12 |
| **大模型推理框架深度解析：vLLM、TensorRT-LLM 与 TGI 技术选型指南** | 百度开发者平台 | 中文 | 选型指南 | 三大推理框架的算子优化策略横向对比，性能数据和选型建议 | 2025 |
| **AutoTriton: Automatic Triton Programming with RL** | Liner.ai Review | 英文 | 研究解读 | GRPO + SFT 的自动 Triton 编程，8B 模型匹配 Claude-4-Sonnet 效果 | 2025.07 |
| **昇腾 0Day 支持 MiniMax M2.7** | MiniMax 官方 | 中文 | 工程实践 | FlashComm 序列切分、MoE 大融合算子、自适应 DP 域负载均衡等软硬协同优化 | 2026.04 |
| **FlashInfer Autotuner Config Caching 设计** | FlashInfer GitHub Issue #2620 | 英文 | 功能设计 | 持久化 JSON 缓存方案，增量调优，线程安全，离线调优工作流 | 2026 |

## 2.4 技术演进时间线

```
2019 ── OpenAI 发布 Triton 编程语言和编译器，提供 @triton.autotune 基础调优能力
       │    → 奠定了基于 tile 的 kernel 编程模型 + 自动调优范式
2020 ── Ansor (TVM) 提出分层搜索空间 + 进化搜索的自动调优框架
       │    → 影响后续所有 DL 算子自动调优系统的搜索策略设计
2022 ── FlashAttention 提出 IO-aware tiling + recompute 策略
       │    → 揭示了 LLM attention 算子的调优应以"减少 HBM 访问"为核心目标
2023 ── PagedAttention 引入非连续 KV cache 块内存管理
       │    FlashDecoding 提出 3D 并行 kernel 用于 decode 阶段
       │    → 开启了 LLM 推理场景下针对 KV cache 特性的定制 kernel 设计
2024 ── TensorRT-LLM 内建 AutoTuner，支持 TunableRunner + ProfilingCache
       │    FlashInfer 开源，成为 vLLM/SGLang 的通用 kernel 库
       │    → 推理框架开始将 autotuning 作为一等公民功能集成
2025 ── KernelBench 标准化评测发布 → 推动 LLM kernel 自动生成研究
       │    Stanford MegaKernel / Mirage MPK 提出完全融合的单 kernel 推理
       │    CUDAMaster / STARK / PRAGMA 等 LLM 多 Agent 调优系统涌现
       │    Meta 发布 KernelLLM (8B) 专用于 Triton kernel 生成
       │    → LLM Agent 替代传统搜索成为算子自动调优的新范式
2026 ── Ada-MK 实现首个工业级 MegaKernel 部署（百度广告系统）
       │    → 证明了 DAG 离线搜索 + MegaKernel 路径在生产环境的可行性
       │    当前状态：算子自动调优正向 LLM Agent 驱动 + MegaKernel 融合 + 跨平台（NVIDIA/AMD/Ascend）方向发展，
       │    调优搜索从编译期演进到编译+运行时混合模式，从单算子演进到全图融合调优
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
2020 ─┬─ Ansor/TVM 传统分层搜索自动调优
      │  → 奠定 DL 算子自动调优基础，但搜索空间较大 → 调优收敛慢
       │
2022 ─┼─ FlashAttention IO-aware 优化 + Triton autotune 广泛使用
      │  → 明确了 LLM 推理算子的"HBM 带宽瓶颈"特性，调优目标从纯计算优化转向访存优化
       │
2023 ─┼─ PagedAttention 专用 kernel + TensorRT-LLM AutoTuner
      │  → 推理框架开始内置 autotuning 机制，但以手工编写 kernel + 参数调优为主
       │
2024 ─┼─ FlashInfer kernel 库 + vLLM 优化级别体系
      │  → 标准化 kernel 库 + 系统化调优级别设计，算子优化从"手工作坊"走向"平台化"
       │
2025 ─┼─ LLM Agent 驱动调优 (CUDAMaster/STARK/PRAGMA) + MegaKernel (Mirage/Stanford)
      │  KernelBench 标准化评测 → 算子优化从"手写 + 搜索"跃迁到"AI 自动生成 + 迭代优化"
       │
2026 ─┴─ Ada-MK 工业级 MegaKernel + AKG Agent 多 Agent 协作框架 + AIConfigurator 零 profiling 调优
      │  当前状态：传统搜索调优与 LLM Agent 生成并行发展，MegaKernel 融合和零开销调优成为两大赛道
```

## 3.2 六种方案横向对比

### 方案概览

| 方案 | 典型代表 | 核心思路 |
|------|---------|---------|
| **A: 传统网格/贝叶斯搜索** | Ansor, Triton autotune | 枚举候选配置，Profiling 筛选最优 |
| **B: 编译图级搜索** | Ada-MK (MLIR DAG), TASO | 在计算图级别搜索最优融合和执行路径 |
| **C: 性能建模预测** | AIConfigurator | 构建分析模型预测性能，跳过 Profiling |
| **D: LLM Agent 生成式** | CUDAMaster, PRAGMA, STARK | LLM 直接生成 kernel 代码 + 迭代优化 |
| **E: MegaKernel 全融合** | Mirage MPK, Stanford MegaKernel | 将整个 LLM 推理融合为单 kernel |
| **F: 持久化缓存复用** | FlashInfer ConfigCache, triton-dejavu | 缓存调优结果，避免重复 Profiling |

### 详细对比

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **A: 传统搜索** | 定义搜索空间 → 枚举候选 → Profiling → 选最优 | ① 实现简单，成熟度高 ② 可保证找到局部最优 ③ 与硬件无关，跨平台 | ① 搜索空间大时收敛极慢（数小时） ② 每次输入 shape 变化需重调 ③ 难以发现跨算子全局最优 | 小型模型，配置空间有限的场景 | 低（开发成本）~ 高（计算成本：每搜索数小时 GPU 时） |
| **B: 编译图级搜索** | 将计算图分解为 PTX 级 DAG → 离线搜索最优执行路径 → 编译期固化 | ① 消除运行时分支开销 ② 可发现跨算子融合机会 ③ 精度优于逐算子调优 | ① 实现复杂度高，需深入了解 MLIR/PTX ② 共享内存约束敏感 ③ 编译时间较长 | 生产环境稳定部署，对延迟极度敏感的在线场景 | 中高（开发成本高，但调优结果固化后零运行时开销） |
| **C: 性能建模预测** | 将推理分解为 GEMM/Attention/通信原语 → 分析模型预测 → 一步选优 | ① 零 Profiling 开销，搜索仅需秒级 ② 框架无关，可跨平台迁移 ③ 适合快速部署场景 | ① 建模精度有限，复杂场景误差大 ② 对新硬件/新算子需重新建模 ③ MoE 等动态结构的建模困难 | 快速原型验证，多框架切换，部署频繁变动的场景 | 低（模型构建成本）~ 中（维护多平台模型） |
| **D: LLM Agent 生成** | LLM 根据算子描述生成 kernel 代码 → 编译 → Profiling → 反馈迭代优化 | ① 自动探索 Search Space 之外的新结构 ② 可生成手写级性能的 kernel（超越 cuBLAS） ③ 适应新硬件（无需预建搜索空间） | ① LLM 推理成本高（每次迭代 ~$0.1-1） ② 代码正确性需严格验证 ③ 迭代轮次多时总时间不占优 | 新算子/新硬件首次适配，对性能要求极致但有充裕时间 | 中高（LLM API 成本 + GPU Profiling 成本） |
| **E: MegaKernel 全融合** | 将 N 个算子融合为单 kernel → 一次 launch 完成所有计算 → 零 HBM 中间读写 | ① 完全消除 kernel launch 和 HBM 开销 ② 延迟降低最大（1.2-6.7x） ③ 对 memory-bound 算子最优 | ① 共享内存限制严格（只有 128-227 KB） ② 实现复杂度极高，不支持所有架构 ③ 目前只支持少数模型和 GPU | 大规模生产部署，内存带宽瓶颈明显的场景（decode 阶段） | 高（开发 + 硬件适配成本） |
| **F: 持久化缓存复用** | 缓存算子签名→最优配置映射，支持增量调优和跨部署复用 | ① 消除冷启动调优开销 ② 支持离线调优 + 在线加载 ③ 部署友好，零运行时负担 | ① 缓存命中率依赖形状覆盖度 ② 签名哈希需要版本管理 ③ 本身不产生最优配置，需要与 A-E 方案组合使用 | 所有生产部署场景，尤其是频繁重启/扩缩容的云服务 | 极低（仅缓存 I/O 开销） |

## 3.3 技术细节对比

| 维度 | A: 传统搜索 | B: 编译图级搜索 | C: 性能建模 | D: LLM Agent | E: MegaKernel | F: 缓存复用 |
|------|:-----------:|:--------------:|:-----------:|:------------:|:------------:|:----------:|
| **性能上限** | ★★★☆ | ★★★★☆ | ★★☆ | ★★★★ | ★★★★★ | ★（依赖上游）|
| **调优速度** | ★★ (小时级) | ★★★ (分钟级) | ★★★★★ (秒级) | ★★☆ (分钟-小时) | ★★★ (分钟级) | ★★★★★ (零开销) |
| **实现复杂度** | ★★★★★ (简单) | ★★ (高) | ★★★ (中) | ★★☆ (高) | ★ (极高) | ★★★★ (低-中) |
| **跨平台能力** | ★★★★ | ★★★ | ★★★★★ | ★★★★ | ★ | ★★★★ |
| **生态成熟度** | ★★★★★ (成熟) | ★★ (新兴) | ★★ (新兴) | ★★★ (发展中) | ★ (早期) | ★★★ (发展中) |
| **社区活跃度** | ★★★★ | ★★★ | ★★ | ★★★★★ | ★★★ | ★★★ |
| **学习曲线** | 低 | 高 | 中 | 中-高 | 极高 | 低 |
| **正确性保障** | 有（实际执行验证） | 有（编译期验证） | 无（预测值） | 需额外验证 | 有（实际执行验证） | 有（依赖上游验证） |

## 3.4 选型建议

| 场景 | 推荐方案组合 | 核心理由 | 预估月成本 |
|------|------------|---------|-----------|
| **小型项目/原型验证** (单个开发者，1-2 张 GPU) | A (Triton autotune) + F (triton-dejavu) | 零开发成本，Triton 自带 autotune 开箱即用，dejavu 缓存避免重复调优 | $0-500（现有 GPU 利用率） |
| **中型生产环境** (10-100 张 GPU，多模型) | C (AIConfigurator) + F (FlashInfer ConfigCache) + 部分 D (LLM Agent 调优关键算子) | AIConfigurator 秒级调优 + ConfigCache 跨部署复用，对瓶颈算子选择性使用 LLM Agent | $2,000-10,000（调优计算 + LLM API） |
| **大型分布式系统** (1000+ GPU，在线广告/搜索) | B (Ada-MK DAG 搜索) + E (MegaKernel decode) + F (持久化缓存) | 首延迟敏感的在线场景需要零运行时开销的 MegaKernel，DAG 搜索保证编译期即可获得全局最优 | $20,000-100,000（开发团队 + 调优集群） |
| **新硬件适配** (Ascend NPU / AMD GPU / AWS Trainium) | D (LLM Agent 生成) + F (调优缓存) | 传统搜索方案在新硬件上缺乏搜索空间定义，LLM Agent 可零先验知识生成 kernel | $5,000-30,000（LLM API + 新硬件 Profiling） |
| **云原生/Serverless 推理** | F (离线调优 + 在线加载) + C (快速配置) | Serverless 冷启动频繁，必须零运行时调优开销，离线预调优 + 在线加载模式最合适 | $500-5,000（离线调优集群 + 缓存存储） |

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{算子自动调优} = \underbrace{\text{搜索空间定义}}_{\text{确定优化维度}} + \underbrace{\text{性能评估}}_{\text{硬件反馈信号}} - \underbrace{\text{重复 Profiling 开销}}_{\text{核心效率瓶颈}}
$$

所有调优方案的本质，都是在"搜索空间覆盖度"和"调优效率"之间做权衡。传统方案偏向最大化覆盖度但牺牲效率，LLM Agent 方案用先验知识加速搜索，MegaKernel 方案则通过彻底消除 kernel launch 开销来绕过问题。

## 4.2 一句话解释

**大模型推理算子自动调优，就是让计算机自动为 AI 模型的每个计算步骤找到最快的实现方式——就像给汽车发动机的每个零件找到最优的加工参数，只不过这里的"零件"是 GPU 上跑的计算程序，"加工参数"是线程数、内存分配策略、数据切分大小等成百上千个可调节的旋钮。**

## 4.3 核心架构图

```
模型计算图 → [算子识别/分解] → [配置搜索空间]
                                    │
                             [搜索/生成策略]
                            ┌───────┼───────┐
                            ▼       ▼       ▼
                    [网格搜索] [贝叶斯] [LLM Agent]
                            │       │       │
                            └───────┼───────┘
                                    ▼
                              [Kernel 编译]
                                    ▼
                              [硬件 Profiling]
                                    │
                          ┌─────────┼─────────┐
                          ▼         ▼         ▼
                    [延迟数据] [共享内存] [带宽]
                          │         │         │
                          └─────────┼─────────┘
                                    ▼
                              [最优配置]
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              [持久化缓存]   [推理运行时加载]    [MegaKernel融合]
```

## 4.4 STAR 总结

### Situation（背景+痛点）

大模型推理正成为 AI 基础设施的核心负载。LLM 推理涉及自回归解码、KV cache 管理、注意力计算等特有算子，其性能高度依赖 GPU kernel 的具体实现方式。然而，手工调优每个算子需要数周甚至数月的时间，且调优结果随模型、硬件、batch size 等参数变化而失效。当前主流推理框架（vLLM、TensorRT-LLM、SGLang）的算子数量从数十个增长到上百个，传统人工调优方式已不可持续。同时，新硬件（Blackwell、AMD MI300X、Ascend NPU）的快速迭代进一步加剧了调优的复杂度。

### Task（核心问题）

算子自动调优需要解决三个核心问题：**速度**（如何在合理时间内找到最优配置——分钟级而非小时级）、**泛化**（调优结果如何在不同模型、硬件、输入形状之间迁移复用）、**全局性**（如何不限于单算子最优，而是搜索算子融合和全图执行路径的全局最优）。根本矛盾在于：搜索空间指数级增长 VS 调优收敛时间必须可控。

### Action（主流方案）

算子自动调优经历了四个阶段的演进：**第一阶段**（2020-2022），以 Ansor/Triton autotune 为代表，使用网格/贝叶斯搜索枚举配置；**第二阶段**（2023-2024），TensorRT-LLM AutoTuner 和 FlashInfer 引入了持久化缓存和分布式调优策略，提升调优效率；**第三阶段**（2025），LLM Agent（CUDAMaster、STARK）直接生成 kernel 代码并迭代优化，KernelBench 提供了标准化评测；**第四阶段**（2026），Ada-MK 实现工业级 MegaKernel，通过 MLIR DAG 搜索消除所有运行时分支开销。当前核心突破包括：ML 辅助搜索空间剪枝（Triton RFC #9308）、性能建模替代 Profiling（AIConfigurator）、多 Agent 协作调优（AKG Agent、CUDAMaster）。

### Result（效果+建议）

当前，算子自动调优已能实现 23-50% 的吞吐提升（Ada-MK vs vLLM），LLM Agent 的 kernel 生成在单算子级别已匹配甚至超越手写 kernel（CUDAMaster vs cuBLAS），调优收敛时间从数小时降至秒级（AIConfigurator）。**实操建议**：① 中小规模部署优先使用 Triton autotune + 缓存复用方案（零成本入门）；② 生产环境建议采用 AIConfigurator 性能建模 + FlashInfer ConfigCache 组合（秒级调优 + 跨部署复用）；③ 对延迟敏感的在线服务，关注 MegaKernel 融合方向（Ada-MK 已证明工业可行性）；④ 新硬件适配场景，LLM Agent 生成式调优是最具潜力的方向。

## 4.5 理解确认问题

**问题：** 在 1000+ GPU 的 LLM 推理集群上，你发现 decode 阶段的 attention 算子 GPU 利用率极低（<20%）。基于算子自动调优的知识，你会从哪三个层面分析并优化这个问题？

**参考答案：**

1. **算子层面**：检查 attention kernel 的 tile size（BLOCK_Q/BLOCK_KV）是否适配当前 batch size 和 sequence length。decode 场景 Q len=1，应使用 2D kernel 而非 3D kernel（减少不必要的 context 维度并行）。尝试缩减 warp 数量（如 8→4）减少 register spilling。

2. **融合层面**：检查 RoPE + Q projection + KV cache write 是否已融合为单 kernel。未融合时每个子算子都需要一次完整的 HBM 读写和 kernel launch，融合后可大幅减少访存。同时评估是否可将 attention 与其前后的 RMSNorm、Residual Add 融合。

3. **系统层面**：使用 FlashInfer ConfigCache 或 Triton autotune 的持久化缓存机制，对集群上所有不同 shape（不同 model 的 decode sequence length 不同）预先完成调优，消除每个 GPU 的冷启动 profiling 开销。对于大规模集群，考虑使用 TensorRT-LLM 的 PARALLEL 分布式调优策略，让不同 GPU 分别 profiling 不同 tactic 子集。

---

> **报告完** | 调研日期：2026-05-19 | 数据截至：2026-05
>
> 本文档基于公开信息整理，项目 stars 和版本号可能随时间变化。
