# 大模型训练并行策略自动搜索与优化 —— 深度调研报告

> 调研日期：2026-05-16 | 技术域：大模型训练

---

## 目录

1. [概念剖析](#一概念剖析)
2. [行业情报](#二行业情报)
3. [方案对比](#三方案对比)
4. [精华整合](#四精华整合)

---

## 一、概念剖析

### 1.1 定义澄清

**通行定义**：大模型训练并行策略自动搜索是指通过算法在离散的策略空间中自动寻找最优的分布式并行配置组合（包括数据并行、张量并行、流水线并行、序列并行、ZeRO 分片策略、激活检查点策略等），以最小化端到端训练时间或最大化吞吐量，同时满足显存约束的一种自动化系统技术。

**常见误解**：

1. **"自动搜索等同于 AutoML/NAS"**—— 混淆了并行策略搜索与神经网络架构搜索（NAS）。前者固定模型架构、搜索分布式执行方案；后者搜索模型结构本身。
2. **"并行策略搜索只需一次即可"** —— 实际上最优策略随模型规模、硬件拓扑、批次大小变化而变化，增量搜索和动态调整往往更实用。
3. **"搜索空间越大效果越好"** —— 实际上过大的搜索空间会导致组合爆炸，大多数系统通过剪枝、决策树或动态规划将空间压缩到可处理范围。
4. **"自动搜索出的策略一定优于人工调优"** —— 当前 SOTA 自动搜索与专家手工配置的结果互有胜负，优势更多体现在异构集群和快速原型场景。

**边界辨析**：

| 相邻概念 | 核心区别 |
|---------|---------|
| **AutoML/NAS** | 搜索模型架构；并行策略搜索搜索分布式执行方案 |
| **超参数优化（HPO）** | 搜索学习率、批次大小等训练超参；并行策略搜索关注硬件资源划分 |
| **编译器自动并行化（如XLA）** | 在算子级别自动融合/拆分；并行策略搜索在通信组和策略维度寻优 |
| **调度系统（如Slurm）** | 调度系统分配节点资源；并行策略搜索决定如何使用已分配的节点 |

### 1.2 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│              大模型训练并行策略自动搜索系统架构                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  模型定义 + 硬件拓扑                                                │
│       │                                                            │
│       ▼                                                            │
│  ┌──────────────┐    ┌──────────────────┐                         │
│  │  Profiler     │───▶│  成本模型/性能预估 │                         │
│  │  (显存+耗时)   │    │  (simulator/ML)  │                         │
│  └──────────────┘    └────────┬─────────┘                         │
│       │                       │                                    │
│       ▼                       ▼                                    │
│  ┌────────────────────────────────────────────────┐               │
│  │           搜索引擎 (Search Engine)                │               │
│  │  ┌─────────┐ ┌──────────┐ ┌────────────────┐   │               │
│  │  │ MCTS    │ │ DP/ILP   │ │ Genetic/EA     │   │               │
│  │  │ 树搜索   │ │ 动态规划  │ │ 遗传算法       │   │               │
│  │  └─────────┘ └──────────┘ └────────────────┘   │               │
│  │  ┌─────────┐ ┌──────────┐ ┌────────────────┐   │               │
│  │  │ RL      │ │ 多智能体  │ │ 贝叶斯优化     │   │               │
│  │  └─────────┘ └──────────┘ └────────────────┘   │               │
│  └──────────────────────┬─────────────────────────┘               │
│                         │                                          │
│                         ▼                                          │
│  ┌────────────────────────────────────────────────┐               │
│  │       最优并行策略 (最优并行配置)                   │               │
│  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ │               │
│  │  │DP配置 │ │TP配置 │ │PP配置 │ │SP配置 │ │CKPT  │ │               │
│  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ │               │
│  └──────────────────────┬─────────────────────────┘               │
│                         │                                          │
│                         ▼                                          │
│  ┌────────────────────────────────────────────────┐               │
│  │          运行时执行引擎 (Runtime Engine)           │               │
│  │  ┌──────────┐ ┌──────────┐ ┌────────────────┐   │               │
│  │  │通信调度   │ │显存管理   │ │动态调整        │   │               │
│  │  └──────────┘ └──────────┘ └────────────────┘   │               │
│  └────────────────────────────────────────────────┘               │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**各组件职责**：

| 组件 | 功能 |
|------|------|
| **Profiler（性能分析器）** | 对每个算子的计算时间、显存消耗、通信开销进行微基准测试 |
| **成本模型（Cost Model）** | 根据 profile 数据预测特定策略下的端到端训练时间和显存占用；可用模拟器或 ML 模型 |
| **搜索引擎（Search Engine）** | 在策略空间中高效搜索，找到满足显存约束、训练时间最短的并行配置 |
| **运行时引擎（Runtime Engine）** | 执行搜索出的并行策略，包括通信组初始化、梯度同步调度、动态策略调整 |

### 1.3 数学形式化

#### (1) 并行策略搜索的目标函数

$$
\min_{s \in \mathcal{S}} T_{\text{train}}(s) \quad \text{s.t.} \quad M_{\text{peak}}(s) \leq M_{\text{avail}}
$$

其中 $\mathcal{S}$ 是所有可行并行策略的离散空间，$T_{\text{train}}$ 是端到端训练时间，$M_{\text{peak}}$ 是峰值显存占用，$M_{\text{avail}}$ 是可用显存总量。

#### (2) 流水线并行的气泡率

$$
\text{Bubble}_{\text{PP}} = \frac{(p-1) \cdot (t_{\text{fwd}} + t_{\text{bwd}})}{p \cdot t_{\text{fwd}} + (p-1) \cdot t_{\text{bwd}} + (m-1) \cdot (t_{\text{fwd}} + t_{\text{bwd}})}
$$

其中 $p$ 为流水线阶段数，$m$ 为微批次数量，$t_{\text{fwd}}/t_{\text{bwd}}$ 为前/反向传播时间。气泡率衡量流水线并行引入的空转时间占比。

#### (3) 混合并行通信成本模型

$$
T_{\text{comm}} = \alpha \cdot n_{\text{msg}} + \beta \cdot \frac{D_{\text{size}}}{N_{\text{link}}}
$$

其中 $\alpha$ 是通信延迟（latency），$\beta$ 是带宽倒数（bytes/s），$D_{\text{size}}$ 是通信数据量，$N_{\text{link}}$ 是并行链路数。该公式是评估 TP/DP 通信开销的基础。

#### (4) 显存占用分解

$$
M_{\text{peak}} = M_{\text{params}} + M_{\text{opt_states}} + M_{\text{grad}} + M_{\text{activations}}(s) + M_{\text{reserved}}
$$

激活显存 $M_{\text{activations}}$ 是并行策略 $s$ 的决定性变量——增加 TP 或开启激活检查点可减少它，但会增加通信或计算开销。

#### (5) 自动搜索效率比

$$
R_{\text{auto}} = \frac{T_{\text{manual\_best}} - T_{\text{auto\_found}}}{T_{\text{search}}} \quad \text{(每单位搜索时间的收益)}
$$

该指标量化自动搜索的价值——如果搜索耗时远大于收益，则直接使用默认策略可能更好。

### 1.4 实现逻辑（Python 伪代码）

```python
class AutoParallelSearchSystem:
    """并行策略自动搜索系统的核心抽象"""

    def __init__(self, model_config, hardware_topology):
        self.model_config = model_config          # 模型架构信息
        self.hardware_topology = hardware_topology  # GPU拓扑、带宽
        self.profiler = Profiler()                 # 性能分析器
        self.cost_model = CostModel()              # 策略成本预估模型
        self.search_engine = SearchEngine()        # 策略搜索引擎

    def profile_model(self):
        """对模型的每个算子进行微基准测试"""
        for op in self.model_config.operators:
            compute_time = self.profiler.bench_compute(op)
            memory_usage = self.profiler.measure_memory(op)
            self.cost_model.register_op_profile(op, compute_time, memory_usage)

    def search_optimal_strategy(self, memory_budget):
        """在策略空间中搜索满足显存约束的最优并行配置"""
        # Step 1: 构建策略搜索空间（压缩剪枝后）
        strategy_space = self.build_pruned_space()

        # Step 2: 使用 DP/MCTS 搜索最优策略
        best_strategy = self.search_engine.search(
            space=strategy_space,
            cost_fn=lambda s: self.cost_model.estimate_time(s),
            constraint_fn=lambda s: self.cost_model.estimate_memory(s) <= memory_budget
        )

        # Step 3: 可选的实际运行时验证
        if self.runtime_verification:
            actual_time = self.dry_run(best_strategy)
            best_strategy.actual_cost = actual_time

        return best_strategy

    def execute_training(self, strategy):
        """按照搜索出的策略执行分布式训练"""
        # 初始化通信组
        comm_groups = self.initialize_comm_groups(strategy)
        # 部署模型并行
        self.deploy_model_parallel(strategy, comm_groups)
        # 启动训练循环
        for epoch in range(self.model_config.num_epochs):
            self.train_one_epoch(strategy)

        return strategy.summary()

    def build_pruned_space(self):
        """构建经过剪枝的策略搜索空间（降低组合复杂度）"""
        # 利用决策树剪枝：排除显式不可行的策略组合
        return self.search_engine.prune_by_memory_and_topology(
            self.model_config, self.hardware_topology
        )
```

### 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 模型吞吐量（TFLOP/s/GPU） | > 140（H100） | 稳定训练期间的平均计算吞吐 | 衡量 GPU 利用效率 |
| 端到端加速比 | 1.2×~1.5× vs 默认策略 | 相同模型/集群下对比 | 自动搜索相对于专家配置或默认配置的提升 |
| 搜索时间 | < 30 分钟 | 从开始 profiler 到输出最优策略 | 影响方案的实用性；越长收益越小 |
| 搜索空间压缩率 | > 90% | 剪枝后/剪枝前的空间大小比 | 衡量搜索效率 |
| 成本模型预测精度 | > 95% | 预测时间 vs 实际运行时间 | 影响搜索结果的可靠性 |
| 吞吐加速比（异构集群） | 1.3×~9.2× | 与 SOTA 系统对比 | 异构场景下自动搜索优势更明显 |

### 1.6 扩展性与安全性

**水平扩展**：
- 支持跨节点、跨集群的联合搜索（如 HetAuto 支持多达 736 块异构设备）
- 搜索过程本身可并行化——通过并行profile和多worker协作搜索加速
- 支持 GPU 集群从 8 卡到万卡级别的线性扩展

**垂直扩展**：
- 单节点优化上限取决于 NVLink 域内带宽和 GPU 显存容量
- 通过 FlashAttention、算子融合等提高单 GPU 利用率
- Profiler 和搜索的并行度受 CPU 线程数和内存限制

**安全考量**：
- 策略搜索涉及硬件拓扑信息泄露（在跨集群场景中需注意数据隐私）
- 动态策略调整可能导致通信中断或数据不一致（需设计安全切换机制）
- 成本模型可能被"投毒"（如果使用在线学习更新模型），需引入校验机制
- 搜索过程长时间占用 GPU 资源做 profiling，可能影响同集群中的其他任务

---

## 二、行业情报

### 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **Colossal-AI** | ~41,400 | 自动并行 + MoE + FlashAttention 的一站式大模型训练框架 | Python/PyTorch/CUDA | 2026年活跃 | [hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI) |
| **FlexFlow** | ~1,900 | MCMC驱动的自动并行化策略搜索 | Python/C++/Legion | 2025年1月 | [flexflow/flexflow-train](https://github.com/flexflow/flexflow-train) |
| **Alpa** | ~3,120 | 自动 inter/intra-operator 并行（JAX+XLA） | Python/JAX | 已归档 | [alpa-projects/alpa](https://github.com/alpa-projects/alpa) |
| **Hetu-Galvatron** | ~183 | 逐层混合并行自动搜索（DP+TP+PP+SP+ZeRO+CKPT） | Python/PyTorch/CUDA | 2025-2026活跃 | [PKU-DAIR/Hetu-Galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron) |
| **AMP** | ~300+ | 异构感知自动模型并行（NeurIPS 2022） | Python/PyTorch | 2023年 | [DachengLi1/AMP](https://github.com/DachengLi1/AMP) |
| **Megatron-LM** | ~7,000+ | NVIDIA官方大模型训练框架（核心TP/PP实现） | Python/PyTorch/CUDA | 2026年活跃 | [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) |
| **DeepSpeed** | ~37,000+ | ZeRO内存优化 + 自动并行（微软） | Python/PyTorch/CUDA | 2026年活跃 | [microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed) |
| **Fast-LLM** | ~500+ | ServiceNow开源的LLM训练框架，内置自动并行 | Python/PyTorch | 2025年活跃 | [ServiceNow/Fast-LLM](https://github.com/ServiceNow/Fast-LLM) |
| **SimuMax** | 新项目 | 国产分布式训练仿真工具，支持并行策略自动搜索 | Python | 2026年1月v1.1 | 摩尔线程开源 |
| **SkyPilot AutoSearch** | ~6,000+ | Agent驱动的自动搜索+GPU集群调度 | Python | 2026年活跃 | [skypilot-org/skypilot](https://github.com/skypilot-org/skypilot) |

### 2.2 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 |
|------|----------|------|----------|---------|--------|
| Alpa: Automating Inter- and Intra-Operator Parallelism | Zheng et al. / UC Berkeley + Google | 2022 | **OSDI** | 首次统一 inter/intra-operator 自动并行搜索框架 | 高被引，奠基性工作 |
| Unity: Accelerating DNN Training Through Joint Optimization | Unger et al. / CMU + Meta | 2022 | **OSDI** | 代数变换+并行化的联合自动调优（FlexFlow内核） | 高影响力 |
| Galvatron: Efficient Transformer Training with Auto Hybrid Parallelism | Li et al. / PKU | 2025 | **arXiv** | 逐层混合并行自动搜索，1.26-1.47× 加速 | 企业采用（华为、字节） |
| Galvatron-BMW: Bi-Objective Memory-Workload Optimization | Li et al. / PKU | 2025 | **arXiv** | 引入流水线阶段间显存-吞吐双目标优化 | 持续扩展系列 |
| HetAuto: Cross-Cluster Auto-Parallelism | PKU | 2026 | **EuroSys** | 跨异构集群的MCTS+随机森林成本模型搜索 | 最具创新性 |
| Koala: Automated Pipeline Schedule Search via DSL | 多机构 | 2025 | **ACM TACO** | DSL建模+遗传算法的流水线调度自动搜索 | 流水线专用搜索 |
| PROMPTS: Multi-Agent Planning for LLM Training | MLSys团队 | 2026 | **MLSys** | 多智能体+RAG的并行策略生成，87.5%匹配专家级配置 | 开辟新范式 |
| HetRL: Heterogeneous RL Training via Multi-Level Search | 多机构 | 2026 | **arXiv** | 多级搜索+Successive Halving，9.17×加速 | 异构场景最优 |
| Mist: Memory-Parallelism Co-Optimization | 多机构 | 2025 | **EuroSys** | 全量内存减少技术与并行的联合优化，1.28-1.73×加速 | 全面性突出 |
| TAPAS: Fast Tensor Parallel Strategy Derivation | 多机构 | 2023(更新至2025) | **arXiv** | 利用重复子结构使搜索比Alpa快160× | 搜索效率大幅提升 |
| Astra: Money-saving Auto Parallel Search | 多机构 | 2025 | **arXiv** | 首个引入"省钱"模式的异构GPU并行搜索 | 成本感知创新 |
| Colossal-Auto: Unified Automation of Parallelization and CKPT | Li et al. / 潞晨 | 2023 | **AAAI** | 统一自动并行与激活检查点搜索 | 工程实践参考 |
| AutoSP: Compiler-Based Auto Sequence Parallelism | 多机构 | 2026 | **arXiv** | 编译器驱动自动序列并行，长上下文训练2.7×提升 | 长上下文场景突破 |

### 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|---------|----------|------|------|---------|------|
| Scaling Karpathy's AutoResearch: Agent Gets a GPU Cluster | SkyPilot Blog | EN | 实践报告 | Agent 自动管理 16 块 H100/H200 进行超参+并行搜索 | 2026.04 |
| Fixstars AIBooster 自动调优 Megatron Core 并行策略 | Fixstars | EN | 产品技术 | 两种搜索算法实现 1/16 搜索时间 + 1.79×加速 | 2026.04 |
| 深度学习自动并行技术：突破计算瓶颈的智能调度艺术 | CSDN 博客 | ZH | 深度教程 | 全自动并行技术体系综述（2025版） | 2025.12 |
| Survey on Auto-Parallelism of Large-Scale DL Training | OpenReview | EN | 综述 | 全面调查自动并行的挑战、基础和搜索方法 | 2025 |
| Parallelim Strategies for GenAI Deployments | BudEcosystem | EN | 技术博客 | 2023-2025年各并行策略全面梳理 | 2025 |
| 分布式训练全解析：PyTorch/DeepSpeed/Megatron对比 | CSDN | ZH | 对比教程 | 三大框架并行策略对比与实战 | 2025 |
| Galvatron: Auto Distributed Training Introduction | Hetu-Galvatron Docs | EN | 官方文档 | 系统架构、API使用、实验复现 | 2025-2026 |
| Maestro: Accelerating Compound LLM Training | arXiv Blog | EN | 研究简报 | Wavefront调度优化复合训练负载 | 2026.05 |
| 大模型自动并行训练技术综述 | 知乎/PaperWeekly | ZH | 综述 | 中文社区最完整的自动并行技术汇总 | 2025 |
| Distributed Training of LLMs: A Survey | ScienceDirect | EN | 学术综述 | DeepSpeed/Megatron/Colossal-AI系统性对比 | 2025 |

### 2.4 技术演进时间线

| 时间 | 事件 | 发起方 | 影响 |
|------|------|-------|------|
| **2019** | FlexFlow 提出"超越数据和模型并行" | CMU/Meta |首次在多维度上自动搜索并行策略 |
| **2021** | DeepSpeed ZeRO-3 + Megatron-LM TP 成为主流 | 微软/NVIDIA | 奠定了"3D并行"的事实标准 |
| **2022.07** | Alpa (OSDI'22) 提出统一 inter/intra-operator 自动并行 | UC Berkeley | 首次用 ILP 建模两级并行搜索 |
| **2022.09** | Unity (OSDI'22) 代数变换+并行化联合搜索 | CMU/Meta | 扩展了搜索的维度（不仅是并行策略） |
| **2023.02** | Colossal-Auto (AAAI'23) 统一自动并行+激活检查点 | 潞晨科技 | 将自动并行推向工程化产品（1行代码） |
| **2023.06** | TAPAS 利用重复子结构加速搜索 | 多机构 | 搜索效率提升160×，逼近实时 |
| **2024-2025** | Galvatron 系列论文发布 | 北京大学 | 逐层DP+动态调整，被华为/字节采用 |
| **2025.01** | Astra 引入成本感知的并行搜索 | 多机构 | 首次将"省钱"作为优化目标 |
| **2025.07** | Koala 用 DSL+遗传算法搜索流水线调度 | 多机构 | 流水线调度自动化的新范式 |
| **2025-2026** | PROMPTS 提出多智能体+RAG 生成并行策略 | MLSys | 开辟知识驱动而非穷举搜索的新路径 |
| **2026.04** | HetAuto (EuroSys'26) 跨集群自动并行 | 北京大学 | 扩展至736异构设备的跨集群场景 |
| **2026.05** | SkyPilot Agent 驱动 + 16 GPU 自动搜索 | SkyPilot | Agent 自动化搜索扩展至端到端 |
| **2026 当前** | **走向全自动、异构感知、成本感知、知识驱动** | 全行业 | **自动并行策略搜索从学术走向生产** |

---

## 三、方案对比

### 3.1 历史发展时间线

```
2019 ── FlexFlow: 首次多维度自动并行搜索 → 证明"自动比手动更好"
      ── GPipe/1F1B: 流水线并行标准化 → 降低PP使用门槛
2020 ── ZeRO: 显存分片打破DP显存瓶颈 → 让自动搜索需要覆盖ZeRO策略
      ── Megatron-LM TP: 张量并行在NVLink上高效 → 自动搜索需感知硬件拓扑
2021 ── 3D并行(DP+TP+PP)成为标准范式 → 搜索空间膨胀到组合爆炸
2022 ── Alpa (ILP搜索) / Unity (代数变换+并行) → 学术上证明自动搜索可行
      ── Colossal-Auto: 1行代码自动并行 → 工程上证明可用
2023 ── TAPAS 160×加速 → 搜索从小时级降至分钟级
      ── ZeRO++ 通信压缩 → 自动搜索需考虑通信优化多层次
2024 ── Galvatron: 逐层并行决策 — 更细粒度、更优
      ── 异构集群成为焦点（Astra, HAPT）
2025 ── PROMPTS: 多智能体+RAG → 知识驱动替代穷举
      ── Koala: DSL+遗传算法 → 流水线专用优化
2026 ── HetAuto: 跨736设备异构集群 → 极大规模自动搜索
      ── 当前状态: 多种范式并存，知识驱动+算法搜索融合趋势明显
```

### 3.2 七种方案横向对比

#### 方案概述

| 方案 | 代表系统 | 核心原理 |
|------|---------|---------|
| **A - ILP/DP精确求解** | Alpa, Galvatron | 将并行决策建模为整数线性规划或动态规划，求全局最优解 |
| **B - 启发式搜索** | HetAuto (MCTS), FlexFlow (MCMC) | 使用蒙特卡洛树搜索或马尔可夫链蒙特卡洛采样搜索策略空间 |
| **C - 遗传/进化算法** | Koala, HetRL (EA) | 策略空间中的候选解通过交叉、变异、选择进化迭代 |
| **D - 成本模型+剪枝** | Colossal-Auto, TAPAS | 建立高精度成本模型预测性能，通过剪枝压缩空间 |
| **E - 多智能体/RAG** | PROMPTS | 利用大模型+知识库+多智能体协作生成和优化策略 |
| **F - 编译器自动并行** | AutoSP, XLA/MLIR | 在编译器层自动分析计算图，生成分布式执行方案 |
| **G - 仿真驱动搜索** | SimuMax | 通过高精度仿真代替真实运行来评估策略 |

#### 横向对比表

| 方案 | 原理 | 优点（3+） | 缺点（3+） | 适用场景 | 成本量级 |
|------|------|-----------|-----------|---------|---------|
| **A - ILP/DP** | 建模为优化问题求全局最优 | ① 解的质量高，理论最优性保证<br>② 可严格处理显存约束<br>③ 数学模型透明可解释 | ① 搜索空间大时组合爆炸<br>② 需要准确profile数据<br>③ 对模型架构变化敏感 | 中小规模（<100B）模型，同构集群 | 搜索成本低（O(10³)秒） |
| **B - 启发式搜索** | MCTS/MCMC采样搜索 | ① 可处理极大规模空间<br>② 对不精确成本模型鲁棒<br>③ 可并行加速 | ① 无最优性保证<br>② 调参（MCTS超参数）敏感<br>③ 收敛速度不确定 | 大规模异构集群探索 | 搜索成本中（O(10⁴)秒） |
| **C - 遗传/进化** | 种群迭代进化 | ① 天然并行化<br>② 适合非凸/离散空间<br>③ 可同时维护多个候选解 | ① 收敛慢，需大量迭代<br>② 遗传算子设计依赖专家<br>③ 易早熟收敛 | 流水线调度等结构化空间 | 搜索成本中高 |
| **D - 成本模型+剪枝** | ML模型预测+空间压缩 | ① 搜索速度极快（TAPAS快160×）<br>② 可泛化到未见配置<br>③ 空间压缩减少冗余评估 | ① 成本模型精度是关键瓶颈<br>② 需要大量训练数据构建模型<br>③ 迁移到新硬件需要重新训练 | 快速原型、频繁调优场景 | 搜索成本低（秒级） |
| **E - 多智能体/RAG** | LLM+RAG生成策略 | ① 可借助历史经验，不需每次都搜索<br>② 87.5%匹配专家配置<br>③ 自然语言界面可解释 | ① 无法保证最优性<br>② 依赖知识库质量和覆盖度<br>③ LLM推理开销 | 生产环境，已有大量调优经验积累 | 搜索成本低（LLM调用） |
| **F - 编译器自动并行** | 编译器IR层自动分析 | ① 与框架无关，可跨平台<br>② 可在IR层做全局优化<br>③ 一次编译多次受益 | ① 开发门槛高<br>② 对动态图支持有限<br>③ 调试复杂 | 静态计算图，长期固定模型 | 开发成本高，运行成本低 |
| **G - 仿真驱动** | 高精度模拟 | ① 不占用真实GPU资源<br>② 可并行评估海量策略<br>③ 支持what-if分析 | ① 仿真精度有限<br>② 需校准大量参数<br>③ 无法捕捉真实运行时抖动 | 快速探索、资源受限环境 | 无需GPU，仿真成本低 |

### 3.3 技术细节对比

| 维度 | A - ILP/DP | B - 启发式 | C - 遗传/进化 | D - 成本模型+剪枝 | E - 多智能体/RAG |
|------|-----------|-----------|-------------|----------------|----------------|
| **搜索质量** | ★★★★★ 理论最优 | ★★★★ 近似最优 | ★★★ 较好但波动大 | ★★★★ 依赖模型精度 | ★★★ 由知识库质量决定 |
| **搜索速度** | ★★★ O(10³)s | ★★★ O(10⁴)s | ★★ O(10⁴~10⁵)s | ★★★★★ O(10⁰~10²)s | ★★★★★ O(10¹)s |
| **可解释性** | ★★★★★ 数学透明 | ★★★ 搜索过程难追踪 | ★★ 黑盒 | ★★★ 成本模型可分析 | ★★★★★ 自然语言+引用 |
| **异构适应** | ★★ ILP建模复杂 | ★★★★★ MCTS天然适合 | ★★★ 需定制算子 | ★★★ 需重新profile | ★★★ 依赖历史知识 |
| **泛化能力** | ★★ 每个模型重新搜索 | ★★★ 有限泛化 | ★★★ 有限泛化 | ★★★★ 可泛化同类模型 | ★★★★★ 可复用经验 |
| **工程成熟度** | ★★★ Alpa/Colossal-AI生产化 | ★★★ FlexFlow成熟 | ★★ 学术为主 | ★★★ TAPAS/ColossalAuto | ★ 新范式，PROMPTS刚发表 |
| **扩展至万亿参数** | ★ 组合爆炸 | ★★★ 可扩展到万卡 | ★★★ 可扩展 | ★★★ 需大量profile | ★★★★ 可增量积累 |
| **社区生态** | ★★★ 多个开源实现 | ★★★ FlexFlow为主 | ★★ 分散 | ★★★ 集成在Colossal-AI中 | ★ 仅PROMPTS |

### 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证（<7B模型，1-4 GPU）** | D - 成本模型+剪枝（Colossal-Auto） | 一行代码启动自动并行，搜索成本极低，无需GPU调优经验 | $0-500（搜索几乎无额外成本） |
| **中型生产训练（7B-70B模型，8-64 GPU）** | A - ILP/DP（Galvatron） | 逐层精确搜索，1.2-1.5×吞吐提升，已被华为/字节验证 | $500-2,000（Profiler+DP搜索时间约30分钟） |
| **大型分布式训练（70B-1T参数，128+ GPU）** | B+C 混合（HetAuto的MCTS + EA） | MCTS处理大规模空间，EA保持多样性，支持跨集群 | $2,000-10,000（搜索时间1-5小时，但吞吐提升明显） |
| **异构集群训练** | B - 启发式搜索（HetRL + HetAuto） | 异构场景下自动搜索优势最明显（最高9.17×加速） | $1,000-5,000（异构适配本身就是省钱） |
| **生产环境频繁调优** | E - 多智能体/RAG（PROMPTS） | 利用历史经验快速生成策略，87.5%匹配专家，无需从头搜索 | $500-1,500（主要是LLM API调用费） |
| **超长上下文训练** | F - 编译器自动并行（AutoSP） | 编译器级别自动应用序列并行，2.7×上下文长度提升 | $1,000-3,000（一次开发长期受益） |
| **资源受限/预算敏感** | G - 仿真驱动（SimuMax + G） | 不占真实GPU资源即可评估，快速迭代策略 | $0-200（仅计算成本） |

---

## 四、精华整合

### 4.1 The One 公式

$$
\text{自动并行策略搜索} = \underbrace{\text{成本模型预测}}_{\text{预知未来——\ 不运行也能估算}} + \underbrace{\text{智能搜索算法}}_{\text{探索策略空间——\ 找到最优解}} - \underbrace{\text{搜索时间开销}}_{\text{时间损耗——\ 抵消调优收益}}
$$

### 4.2 一句话解释

> 大模型训练并行策略自动搜索，就像给一栋大楼自动分配电梯数量和调度策略——只不过这栋大楼有几百个"楼层"（Transformer层），几十部"电梯"（GPU），而"电梯井"之间的连接带宽差异巨大（NVLink vs InfiniBand vs Ethernet），系统需要自动算出怎么分配才能在最短时间内把所有人送上楼。

### 4.3 核心架构图

```
模型+硬件
    │
    ▼
┌──────────┐    ┌───────────┐
│ Profiler  │───▶│ 成本模型   │
│ (测速+测存) │    │ (预估性能) │
└──────────┘    └─────┬─────┘
                      │
                      ▼
              ┌─────────────────┐
              │   搜索引擎        │
              │  DP/ILP/MCTS/EA  │
              │  多智能体/遗传    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  最优并行配置     │
              │ DP+TP+PP+SP+CKPT│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  运行时执行引擎    │
              │  通信+显存+调度   │
              └─────────────────┘
```

### 4.4 STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation**（背景+痛点） | 大模型从百亿增长至万亿参数，训练需要的 GPU 集群从数十卡扩展至万卡。分布式并行策略（DP/TP/PP/SP/ZeRO）的组合空间呈指数级增长。人工调优依赖专家经验，耗时数周到数月，且当模型架构、硬件拓扑或批次大小变化时需重新调整。异构集群（混合不同代际 GPU）和跨集群训练进一步加剧了这一挑战。 |
| **Task**（核心问题） | 关键问题是在满足显存约束的前提下，自动化地为给定模型+硬件组合搜索最优并行策略，使端到端训练吞吐量最大化。这需要同时解决三个子问题：(1) 建立高精度的性能预测模型；(2) 在超大规模离散空间中进行高效搜索；(3) 使搜索结果可适应动态变化的训练环境。 |
| **Action**（主流方案） | 技术演进经历了三个阶段：(1) **手工时代**：工程师凭经验配置3D并行（DP+TP+PP）参数，依赖 Megatron-LM、DeepSpeed 等框架；(2) **搜索时代**：Alpa（2022）首次用 ILP 统一搜索 inter/intra-operator 并行，Colossal-AI 将其产品化为"一行代码自动并行"，Galvatron（2025）实现逐层细粒度搜索；(3) **智能时代**：PROMPTS（2026）引入多智能体+RAG 知识驱动搜索，HetAuto（EuroSys 2026）支持跨736异构设备的 MCTS 搜索。搜索时间从数小时降至数秒级。 |
| **Result**（效果+建议） | 当前 SOTA 自动搜索相比默认策略可实现 1.2-1.5× 吞吐提升（同构）和最高 9.17× 提升（异构），搜索时间通常控制在 30 分钟以内。**实操建议**：7B-70B 级别模型推荐 Galvatron（DP搜索，30分钟搞定）；异构集群推荐 HetAuto/HetRL；已积累大量历史调优经验的团队可尝试 PROMPTS 多智能体方案；资源受限场景可先用 SimuMax 仿真评估。 |

### 4.5 理解确认问题

**问题**：假设你有一个 12 层 Transformer 的模型和 4 块 GPU，NVLink 连接 GPU 0-1 和 GPU 2-3 内部，但 GPU 1 到 GPU 2 之间只有慢速网络。在自动搜索并行策略时，TP 应该放在哪些 GPU 上？PP 应该跨越多少个 GPU？为什么？

**参考答案**：最优方案通常是将 TP 放在 NVLink 高速连接的 GPU 对内（如 GPU 0-1 为一组做 TP，GPU 2-3 为另一组做 TP），而 PP 跨越两个 TP 组（即 PP-stage-0 用 {GPU0, GPU1} 做 TP，PP-stage-1 用 {GPU2, GPU3} 做 TP）。这是因为 TP 的通信量远大于 PP（TP 需要在每步前向/反向传播中进行密集的 all-reduce，而 PP 只在阶段间传输激活值的边界数据），因此 TP 必须放在高速链路上。这也是为何自动搜索需要感知硬件拓扑的核心原因——人工很容易忽略此类细节。

---

## 参考来源

- [HetAuto: Cross-Cluster Auto-Parallelism (EuroSys 2026)](https://dl.acm.org/doi/10.1145/3767295.3803590)
- [Koala: Automated Pipeline Schedule Searching (ACM TACO 2025)](https://dl.acm.org/doi/10.1145/3722113)
- [PROMPTS: Multi-Agent Planning for LLM Training/Serving (MLSys 2026)](https://mlsys.org/virtual/2026/poster/3620)
- [Galvatron: Auto Distributed Training for Foundation Models (arXiv 2025)](https://arxiv.org/abs/2504.21411)
- [Galvatron: Automatic Distributed Training for Large Transformer Models (arXiv 2025)](https://arxiv.org/abs/2504.03662)
- [HetRL: Multi-Level Search for RL Training (arXiv 2026)](https://arxiv.org/html/2512.12476v2)
- [Astra: Money-saving Auto Parallel Search (arXiv 2025)](https://www.arxiv.org/abs/2502.13480)
- [UniEP: Unified Expert-Parallel MoE MegaKernel (arXiv 2026)](https://browse-export.arxiv.org/abs/2604.19241)
- [AutoSP: Compiler-Based Auto Sequence Parallelism (arXiv 2026)](https://browse-export.arxiv.org/abs/2604.27089)
- [Maestro: Accelerating Compound LLM Training (arXiv 2026)](https://export.arxiv.org/abs/2605.10501)
- [HAPT: Heterogeneity-Aware Parallel Training (arXiv 2025)](https://www.emergentmind.com/papers/2509.24859)
- [TAPAS: Fast Tensor Parallel Strategy Derivation (arXiv 2023-2025)](https://arxiv.org/html/2302.00247v2)
- [Mist: Memory-Parallelism Co-Optimization (EuroSys 2025)](https://ar5iv.labs.arxiv.org/html/2503.19050v1)
- [Alpa: Automating Inter- and Intra-Operator Parallelism (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/zheng-lianmin)
- [Colossal-AI GitHub](https://github.com/hpcaitech/ColossalAI)
- [Hetu-Galvatron GitHub](https://github.com/PKU-DAIR/Hetu-Galvatron)
- [FlexFlow GitHub](https://github.com/flexflow/flexflow-train)
- [SimuMax v1.1 发布](http://m.chinaaet.com/article/3000175674)
- [Fixstars AIBooster 自动调优](https://www.fixstars.com/en/news/892)
- [SkyPilot Agent GPU Cluster AutoSearch](https://blog.skypilot.co/scaling-autoresearch/)
- [Distributed Training of LLMs: A Survey (ScienceDirect 2025)](https://www.sciencedirect.com/science/article/pii/S2949719125000500)
- [Survey on Auto-Parallelism (OpenReview)](https://openreview.net/forum?id=NsCNCvWudp)
- [Auto-Parallelism with Minimal Memory Redundancy (FITEE 2025)](https://www.fitee.zjujournals.com/en/article/doi/10.1631/FITEE.2300684/)
- [Galvatron on EmergentMind](https://www.emergentmind.com/topics/galvatron-efficient-transformer-training)
