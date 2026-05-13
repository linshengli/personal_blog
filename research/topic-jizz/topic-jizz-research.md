# 大模型多阶段渐进式训练策略优化

> 调研日期：2026-05-13 | 调研范围：2019-2026 核心工作 + 2025-2026 实时数据

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

大模型多阶段渐进式训练策略是指：在大型语言模型的训练过程中，**不是一次性使用全部数据和完整模型架构进行训练**，而是将训练过程划分为多个阶段，在每个阶段中逐步增加模型容量（层数/宽度）、数据难度或训练目标复杂度，从而使模型以"由易到难、由浅入深"的方式渐进地习得语言能力。其核心理念源于人类学习的"最近发展区"理论和课程学习范式。

### 常见误解

1. **误解：渐进式训练=课程学习**。事实：课程学习仅是多阶段渐进式训练的一个子维度。完整的渐进式训练还包含模型架构的增长（如渐进式堆叠层）、训练目标的演进（如从语言建模到指令跟随）、以及训练数据的动态调整。课程学习主要关注数据顺序，而渐进式训练是规模、数据和目标的**三维联动**。

2. **误解：渐进式训练=继续预训练**。事实：继续预训练（Continual Pretraining）是在固定架构上添加新数据训练，而渐进式训练通常伴随着模型容量的动态变化。两者解决的问题不同——继续预训练解决领域迁移，渐进式训练解决训练效率。

3. **误解：小模型训练后直接扩展=重新开始训练**。事实：通过 Function-Preserving Initialization（如 bert2BERT 的 Net2Net 适配），小模型的权重可以无损地初始化为大模型的起始点，使大模型"继承"小模型已学知识，这比重头训练节省约 25%-45% 的计算量。

### 边界辨析

| 易混淆概念 | 核心区别 |
|-----------|---------|
| **课程学习 (Curriculum Learning)** | 仅关注数据样本的呈现顺序，不改变模型架构 |
| **多任务学习 (Multi-Task Learning)** | 同时学习多个任务，没有时间维度上的渐进性 |
| **迁移学习 (Transfer Learning)** | 从一个领域/任务迁移到另一个，通常只有一次迁移 |
| **知识蒸馏 (Knowledge Distillation)** | 大模型指导小模型，不涉及模型生长的逆向过程 |

## 1.2 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│              大模型多阶段渐进式训练系统架构                        │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────┐   ┌──────────────────────┐   ┌───────────────┐     │
│  │ 数据  │ → │  多阶段训练调度器      │ → │  模型评估与    │     │
│  │ 流水线 │   │  (Stage Scheduler)   │   │  监控系统      │     │
│  └──────┘   └──────────┬───────────┘   └───────┬───────┘     │
│                         │                        │           │
│                         ▼                        ▼           │
│              ┌─────────────────────┐   ┌───────────────┐     │
│              │  阶段控制器           │   │  难度评估器    │     │
│              │  Stage 1 → Stage N  │   │  Difficulty   │     │
│              │  (渐进切换逻辑)       │   │  Estimator    │     │
│              └──────────┬──────────┘   └──────┬────────┘     │
│                         │                     │              │
│  ┌──────────────────────┴─────────────────────┴──────────┐  │
│  │                    每个训练阶段                         │  │
│  │  ┌──────────────┐  ┌──────────┐  ┌────────────────┐  │  │
│  │  │ 模型容量管理器  │  │ 数据采样器 │  │ 训练目标控制器  │  │  │
│  │  │ (添加/扩展层)  │  │ (难度采样) │  │ (损失函数切换)  │  │  │
│  │  └──────────────┘  └──────────┘  └────────────────┘  │  │
│  └─────────────────────────────────────────────────────┘  │
│                         │                                  │
│                         ▼                                  │
│              ┌─────────────────────┐                       │
│              │  基础训练引擎         │                       │
│              │  (Megatron/DeepSpeed)│                       │
│              └─────────────────────┘                       │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**核心组件说明：**

| 组件 | 职责 |
|------|------|
| **多阶段训练调度器** | 定义阶段切换时机（基于 token 数、loss 收敛或验证指标），管理阶段间过渡 |
| **模型容量管理器** | 控制模型生长：添加 Transformer 层（深度生长）或扩展隐藏维度（宽度生长），执行 Function-Preserving Initialization |
| **数据采样器** | 按当前阶段配置的难度分布对训练数据进行采样，实现由易到难的课程学习 |
| **训练目标控制器** | 在不同阶段切换损失函数：从基础语言建模到指令跟随、偏好对齐等 |
| **难度评估器** | 使用压缩率、词汇多样性、可读性等指标评估样本难度，支持自适应课程调度 |
| **基础训练引擎** | 实际的分布式训练执行层，负责前向/反向传播、梯度同步和参数更新 |

## 1.3 数学形式化

### 1.3.1 阶段定义

设训练过程被划分为 $K$ 个阶段，第 $k$ 阶段的状态由三元组定义：

$$
\mathcal{S}_k = (\mathcal{D}_k, \mathcal{M}_k, \mathcal{L}_k), \quad k \in \{1, 2, \ldots, K\}
$$

其中 $\mathcal{D}_k$ 为第 $k$ 阶段的数据分布，$\mathcal{M}_k$ 为模型架构参数，$\mathcal{L}_k$ 为损失函数。渐进式训练的约束条件为：$\mathcal{D}_k$ 的难度单调递增，$\mathcal{M}_k$ 的容量单调递增。

### 1.3.2 课程学习加速比

课程学习相对于随机采样的收敛加速比可以量化为：

$$
\eta_{\text{curriculum}} = \frac{T_{\text{random}}}{T_{\text{curriculum}}} \in [1.18, 1.45]
$$

实证研究表明（EACL 2026），在大模型预训练中课程学习可减少 **18%-45%** 的训练步数。当 $\eta > 1$ 时，课程学习有效，且压缩率和词汇多样性（MTLD）作为难度信号时 $\eta$ 最大。

### 1.3.3 模型扩展的 Function-Preserving 条件

将小模型 $\theta_{\text{small}} \in \mathbb{R}^{d_s}$ 扩展为大模型 $\theta_{\text{large}} \in \mathbb{R}^{d_l}$（$d_l > d_s$），Function-Preserving 初始化要求：

$$
f(x; \theta_{\text{large}}^{(0)}) = f(x; \theta_{\text{small}}), \quad \forall x \in \mathcal{X}
$$

对于 Transformer 的深度扩展（从 $L$ 层到 $L+1$ 层），在第 $m$ 层后插入新层时的权重初始化为：

$$
\mathbf{W}_{\text{new}}^{(0)} = \mathbf{W}_{\text{old}}^{(m)}, \quad \mathbf{b}_{\text{new}}^{(0)} = \mathbf{0}
$$

这确保了扩展后的模型在初始化时与扩展前行为一致，使模型可以"无缝继续训练"。

### 1.3.4 计算成本收益模型

渐进式训练相对于独立训练的计算成本节省率：

$$
C_{\text{save}} = 1 - \frac{\sum_{k=1}^{K} T_k \cdot \text{FLOPs}(\mathcal{M}_k)}{\sum_{k=1}^{K} T_k \cdot \text{FLOPs}(\mathcal{M}_K)}
$$

其中 $T_k$ 是第 $k$ 阶段的训练步数，$\text{FLOPs}(\mathcal{M})$ 是模型 $\mathcal{M}$ 的单步计算量。COLM 2025 的实证表明，当采用从小到大的渐进式训练时，$C_{\text{save}} \approx 25\%$。

### 1.3.5 预算衰减调度

在渐进式 RL 训练中（如 GRPO 的推理长度控制），token 预算由以下调度函数管理：

$$
B(t) = \max(B_{\min}, B_0 \cdot \gamma^{\lfloor t / T \rfloor})
$$

其中 $B_0$ 为初始预算，$\gamma \in (0, 1)$ 为衰减因子，$T$ 为衰减间隔，$B_{\min}$ 为最小预算。该调度使模型先探索长推理链，再逐步压缩为简洁推理。

## 1.4 实现逻辑（Python 伪代码）

```python
class ProgressiveTrainer:
    """多阶段渐进式训练管理器"""

    def __init__(self, config: TrainingConfig):
        self.stages = config.stages                # 阶段定义列表
        self.current_stage = 0
        self.model = self._init_base_model(config.base_model)
        self.data_pipeline = DataPipeline(config.data_sources)
        self.difficulty_estimator = DifficultyEstimator(config.difficulty_metrics)

    def train(self):
        """执行完整的多阶段渐进式训练"""
        for stage_idx, stage_config in enumerate(self.stages):
            self.current_stage = stage_idx
            print(f"进入阶段 {stage_idx + 1}/{len(self.stages)}: {stage_config.name}")

            # 阶段前准备：模型扩展
            if stage_config.model_growth:
                self.model = self._grow_model(stage_config.model_growth)

            # 配置当前阶段的数据采样器和训练目标
            sampler = self._build_sampler(stage_config.difficulty_range)
            loss_fn = self._build_loss_fn(stage_config.training_objective)

            # 执行当前阶段训练
            for batch in self.data_pipeline.iterate(sampler, stage_config.num_steps):
                loss = self._training_step(batch, loss_fn)

                if self._should_switch_stage(stage_config.switch_criteria):
                    break

            # 阶段后处理：评估和模型保存
            self._evaluate_and_save(stage_config)

    def _grow_model(self, growth_config: GrowthConfig):
        """模型扩展（深度/宽度生长）"""
        if growth_config.mode == "depth":
            # 在中间层插入新的 Transformer 层（MIDAS 方法）
            insert_position = len(self.model.layers) // 2 if growth_config.middle_insert else -1
            new_layer = copy.deepcopy(self.model.layers[insert_position])
            # Function-Preserving 初始化：新层权重 copy 旧层
            self.model.layers.insert(insert_position + 1, new_layer)
            self.model.reset_biases_at(insert_position + 1)  # bias 置零
        elif growth_config.mode == "width":
            # 扩展 FFN 和 Attention 的隐藏维度
            self.model.expand_hidden_dim(growth_config.expand_ratio)
        return self.model

    def _should_switch_stage(self, criteria: SwitchCriteria) -> bool:
        """判断是否应切换到下一阶段"""
        if criteria.type == "steps":
            return self.step_count >= criteria.threshold
        elif criteria.type == "loss_plateau":
            return self._loss_plateau_detected(criteria.patience)
        elif criteria.type == "accuracy":
            return self.eval_accuracy >= criteria.threshold
```

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 训练加速比 | 1.2x - 1.45x | 同等困惑度下的训练步数对比 | 课程学习相对随机采样的步数减少比 |
| 计算节省率 | 20% - 45% | FLOPs 总量对比 | 渐进式训练相对独立训练的计算节省 |
| 基准平均增益 | +2% - +4% | 各下游基准测试的平均提升 | CGLS 在 1.2B 规模达 +3.9% |
| 吞吐提升 | 1.3x - 8x | 端到端每秒 token 数 | 取决于模型生长策略和分布式系统 |
| 样本效率 | 2x - 100x | 达到同等性能所需样本数 | SPaCe 自步学习最高达 100x |
| 模型参数利用率 | >90% | 各层梯度范数的均匀性 | MIDAS 方法可避免"深度诅咒" |
| 后训练参数重塑率 | >90% | PRISM 方法测量权重变化 | 中训练重塑 90%+ 参数，RL 仅 ~5% |

## 1.6 扩展性与安全性

### 水平扩展

多阶段渐进式训练天然适合水平扩展：
- **并行的模型家族构建**：不同规模的模型可从小模型渐进扩展而来，而非独立训练。COLM 2025 方法用 1B→2B→4B→8B 的渐进式扩展构建模型家族，总计算量降低 ~25%。
- **多领域专家并行训练**：美团 LongCat-Flash-Thinking 方案对不同领域（STEM、Code、Agentic）分别训练专家模型，然后进行参数级融合，实现训练并行化。
- **异步 Rollout 扩展**：DORA 系统的多版本流式训练实现异步 rollout，端到端吞吐提升最高 2.12 倍。

### 垂直扩展

- **单节点优化**：渐进式训练早期使用小模型时，可利用更大的 batch size 和更高的学习率加速训练。
- **混合精度调度**：不同阶段可动态切换精度（FP16/INT8 混合），文心 5.1 的异构计算调度器即采用此策略。
- **层冻结策略**：Growing Transformers 方法将已训练层冻结，只训练新增层，大幅降低活跃参数比例。

### 安全考量

1. **阶段间的知识遗忘**：在模型扩展或数据切换时可能发生灾难性遗忘。CLewR 方法通过"课程重启"（Curriculum Restarts）缓解此问题。
2. **课程偏差**：如果课程数据过于简单，模型可能无法泛化到复杂真实场景。需使用模型自适应的难度定义（如 CCL 方法）。
3. **扩展后的性能退化**：深度生长后的模型可能在初始阶段出现性能下降。Function-Preserving 初始化 + 渐进式学习率 warmup 可缓解。
4. **RL 阶段的奖励黑客**：在渐进式 RL 阶段，模型可能利用奖励函数的漏洞（reward hacking），需配合持续的红队测试。

---

# 第二部分：行业情报

> 数据采集时间：2026-05-13 | 数据来源：GitHub、arXiv、会议论文、技术博客

## 2.1 GitHub 热门项目

### 多阶段训练框架

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **OpenRLHF** | 10,000+ | 生产级 RLHF 管线：SFT → RM → PPO/REINFORCE++/GRPO | Ray + vLLM + DeepSpeed | 2026/02 | [GitHub](https://github.com/OpenRLHF/OpenRLHF) |
| **Alignment Handbook** | 5,300+ | HuggingFace 官方对齐管线：继续预训练 → SFT → DPO | DeepSpeed + HF Transformers | 2025/07 | [GitHub](https://github.com/huggingface/alignment-handbook) |
| **MS-SWIFT** | 3,500+ | 全生命周期：预训练 → SFT → RLHF → Reward Modeling，支持 600+ 模型 | vLLM + SGLang + DeepSpeed | 活跃更新中 | [GitHub](https://github.com/modelscope/ms-swift) |
| **Slime** | 2,000+ | SGLang 原生后训练框架：SFT → RL(PPO/GRPO) | SGLang + Megatron + Ray | 2025/07 | [GitHub](https://github.com/THUDM/slime) |

### 课程学习与渐进式训练

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **Curriculum-RLAIF** | 21 | 数据驱动课程学习用于 RLAIF 奖励模型训练：易→中→难三阶段 | PyTorch + AlpacaFarm | 2026/04 | [GitHub](https://github.com/ljy2222/Curriculum-RLAIF) |
| **E2H-Reasoning** | 22 | Easy-to-Hard 课程 RL 训练：4 种调度策略（Classic/Balanced/Cosine/Gaussian） | PyTorch + TRL + vLLM + DeepSpeed Z3 | 2025/10 | [GitHub](https://github.com/divelab/E2H-Reasoning) |
| **LAIMARK** | 4 | 模型自生成课程 + GRPO 自我改进，HumanEval 76.8% pass@1 | Qwen3-8B + LoRA + Ollama | 2026/04 | [GitHub](https://github.com/seetrex-ai/laimark) |
| **CLewR** | 2 | 课程学习 + 重启机制，缓解 MT 偏好学习中的灾难性遗忘 | TRL + LoRA | 2026/01 | [GitHub](https://github.com/alexandra-dragomir/CLewR) |
| **DUMP** | 50+ | 分布级课程学习：UCB 动态调整采样概率，GRPO 实例化 | PyTorch + GRPO | 2025/04 | [GitHub](https://github.com/ZhentingWang/DUMP) |
| **Curriculum GRPO** | 100+ | "Train Long, Think Short"：渐进式 token 预算衰减调度 GRPO | PyTorch + GRPO | 2025/08 | [GitHub](https://github.com/hammoudhasan/curriculum_grpo) |
| **EvolvedGRPO** | 30+ | 渐进式指令进化 + RL 用于多模态推理 | GRPO + LVLM | 2026 | [GitHub](https://github.com/SHENZHEBEI/EvolvedGRPO) |

### 模型生长与扩展

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **CompoundGrow (Google)** | 50+ | 联合深度/宽度/序列长度渐进增长，BERT-base 加速 73.6% | TensorFlow | 2021（经典） | [GitHub](https://github.com/google-research/google-research/tree/master/grow_bert) |
| **Sparse Growing Transformer** | 7 | 训练时通过渐进式注意力循环实现稀疏深度分配 | PyTorch | 2026 | [GitHub](https://github.com/YaoChen0203/Sparse-Growing-Transformer) |
| **Growing Transformers** | 30+ | 冻结嵌入层上逐层添加 Transformer，LoRA 周期性微调 | PyTorch | 2025/11 | [GitHub](https://github.com/AVBochkov/Embeddings) |

## 2.2 关键论文

### 经典/高影响力（奠基性工作）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 链接 |
|------|----------|------|----------|---------|------|
| bert2BERT: Towards Reusable Pretrained Language Models | Chen et al. / 清华+华为 | 2022 | ACL 2022 | 提出 Function-Preserving Initialization，将小模型知识迁移到大模型，节省 ~45% 计算量 | [ACL](https://aclanthology.org/2022.acl-long.151/) |
| CompoundGrow: Progressive Transformer Growth | Gu et al. / Google | 2021 | NAACL 2021 | 首次联合深度、宽度和序列长度的渐进增长，BERT-base 加速 73.6% | [Google](https://github.com/google-research/google-research/tree/master/grow_bert) |
| Efficient Stagewise Pretraining via Progressive Subnetworks | Panigrahi et al. | 2023 | NeurIPS Workshop | 渐进式子网训练（RaPTr），逐步增大训练路径长度，节省 20-33% FLOPs | [NeurIPS](https://nips.cc/virtual/2023/81177) |
| Progressive Layer Dropping | Zhang & He / Microsoft | 2020 | NeurIPS 2020 | 训练中渐进式丢弃层以加速 Transformer 训练 | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/...) |
| On the Transformer Growth for Progressive BERT Training | 腾讯 | 2021 | ACL 2021 | 系统研究 Transformer 生长策略，分析层初始化和生长时机 | [arXiv](https://arxiv.org/abs/2010.12562) |

### 最新 SOTA（前沿进展）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|--------|------|
| Efficient Construction of Model Family through Progressive Training | Yano et al. / Preferred Networks | 2026 | COLM 2025/2026 | 1B→2B→4B→8B 渐进扩展构建模型家族，计算节省 25%，性能反超独立训练 | 新方法 | [arXiv](https://arxiv.org/html/2504.00623v2) |
| Curriculum-Guided Layer Scaling (CGLS) | Singh, Band & Adeli | 2025 | arXiv | 同步数据难度递增 + 模型深度渐进增长，1.2B 规模 +3.9% 基准增益 | 开创性联动 | [arXiv](https://arxiv.org/html/2506.11389v3) |
| Beyond Random Sampling: Efficient LLM Pretraining via Curriculum Learning | Zhang et al. / Ecole Polytechnique | 2026 | EACL 2026 | 200+ 模型系统研究，课程学习减少 18-45% 训练步数，最佳难度信号：压缩率/MTLD/可读性 | 系统性实证 | [ACL](https://aclanthology.org/2026.eacl-long.271/) |
| What do Language Models Learn and When? The Implicit Curriculum Hypothesis | 2026 | arXiv 2604.08510 | 提出隐式课程假设，4 模型家族发现技能涌现顺序高度一致（ρ=.81） | 理论突破 | [arXiv](https://browse-export.arxiv.org/abs/2604.08510) |
| PRISM: Demystifying Retention and Interaction in Mid-Training | 2026 | arXiv 2603.17074 | 中训练重塑 90%+ 模型权重而 RL 只改 5%，中训练使数学 +15~+40 分 | 中训练系统研究 | [arXiv](https://browse-export.arxiv.org/abs/2603.17074) |
| EvoLM: In Search of Lost Language Model Training Dynamics | 2025 | NeurIPS 2025 Oral | 100+ 模型跨 4 阶段训练动态分析：预训练→继续预训练→SFT→RL 收益递减规律 | 系统性分析 | [NeurIPS](https://nips.cc/virtual/2025/loc/san-diego/oral/119409) |
| Do Depth-Grown Models Overcome the Curse of Depth? | Kapl et al. | 2025 | arXiv | MIDAS 中间层插入策略，训练加速 ~29%，克服深度诅咒 | 架构突破 | [arXiv](https://arxiv.org/html/2512.08819v1) |
| Growing Transformers: Modular Composition on Frozen Substrate | Bochkov | 2025 | arXiv 2507.07129 | 冻结嵌入层上逐步添加 Transformer 层，复杂推理仅在足够深度涌现 | 模块化生长 | [arXiv](https://arxiv.org/html/2507.07129v2) |
| SPaCe: Self-Pace Curriculum Learning for LLM | 2025 | arXiv | 自步学习 + 多臂老虎机样本选择，100 倍样本效率提升 | 样本高效 | [arXiv](https://lrc.perdanauniversity.edu.my/sdi/space-unlocking-sample-efficient-large-language-models-training-with-self-pace-curriculum-learning/) |
| ACER: Automated Curriculum-Enhanced Regimen | Neema et al. / Cerebras | 2025 | arXiv 2510.26336 | Bloom 分类法的合成教科书式课程，MMLU 宏平均 +3% | 领域专业化 | [arXiv](https://arxiv.org/html/2510.26336v1) |
| Post-Training: Unified View of Off-Policy and On-Policy Learning | 南开/华为 | 2026 | arXiv | 后训练统一视角：离策学习（外部轨迹）+ 在策学习（自身采样）的三角色框架 | 系统性综述 | [BAAI](https://hub.baai.ac.cn/view/53945) |
| DORA: Scalable Asynchronous RL System | 2026 | arXiv 2604.26256 | 多版本流式训练 + 动态编排，端到端吞吐提升 2.12 倍 | 系统优化 | [arXiv](https://arxiv.org/html/2604.26256v1) |

## 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| Train Long, Think Short: Curriculum Learning Makes LLMs Think Smarter | Cognaptus | EN | 深度教程 | GRPO 渐进式 token 预算衰减：指数/线性衰减 + 三角奖励函数效果最佳 | 2025/08 | [Cognaptus](https://cognaptus.com/blog/2025-08-13-train-long-think-short-how-curriculum-learning-makes-llms-think-smarter-not-longer/) |
| Scaling Pedagogical Pre-training: From Optimal Mixing to 10B Tokens | HuggingFace Blog | EN | 深度教程 | 知识图谱引导的教学式课程学习，减少 18-45% 训练步数 | 2026/03 | [HF Blog](https://huggingface.co/blog/codelion/scaling-pedagogical-pretraining-10-billion-tokens) |
| From Simple GPT to Production-Style LLM Stack | Sebastian Raschka | EN | 学习路径 | 从 tokenization → attention → GPT → SFT → RLHF 的四阶渐进学习路径 | 2025 | [Raschka FAQ](https://sebastianraschka.com/faq/docs/simple-gpt-to-production-stack.html) |
| 文心 5.1：多维弹性预训练的突破性实践 | 百度开发者中心 | CN | 技术实践 | 多维弹性预训练：动态参数分配器 + 异构计算调度 + 渐进式知识融合 + 三阶段 RL | 2026/05 | [百度](https://developer.baidu.com/article/detail.html?id=6980009) |
| WOWService：四阶段训练打造高质量对话模型 | 美团技术博客 | CN | 架构解析 | 持续预训练 → SFT → 偏好学习 → 推理增强，构建可维护对话系统 | 2025/11 | [CSDN](https://blog.csdn.net/baidu_25854831/article/details/155439227) |
| LongCat-Flash-Thinking：训练三个模型比一个更强 | 知乎/小宇宙 | CN | 深度解析 | 域并行 RL + 模型融合：分别训练 STEM/Code/Agentic 专家再进行参数融合 | 2025/10 | [知乎](https://www.zhihu.com/question/1953499475269101203) |
| Metis-RISE：先 RL 后 SFT 的非传统训练顺序 | QbitAI | CN | 技术分析 | 突破传统 SFT→RL 顺序，先 RL 探索激发潜力再 SFT 针对性补齐 | 2025/07 | [BAAI](https://hub.baai.ac.cn/view/47512) |
| 大模型训练新突破：不对称训练让 AI 学会自我反思 | 开源中国 | CN | 技术分析 | 字节 PCL 方法：训练时加入评估阶段使模型自我反思，推理零额外开销 | 2025/08 | [OpenAtom](http://openatom.openatom.tech/explore/journalism/detail/478455959749005312) |
| 美团 MTGR：生成式推荐 Scaling Law 落地实践 | 美团技术博客 | CN | 工程实践 | HSTU 架构 + 渐进式训练，FLOPs 提升 65 倍，首页订单量 +1.22% | 2025/05 | [美团](https://tech.meituan.com/2025/05/19/meituan-generative-recommendation.html) |
| LLM Post-Training: Off-Policy and On-Policy Unified View | BAAI | CN | 综述 | 后训练统一框架：有效支撑集扩张、策略重塑、行为巩固三角色 | 2026/04 | [BAAI](https://hub.baai.ac.cn/view/53945) |
| nanochat: 4 小时/100 美元训练 ChatGPT 管线 | Andrej Karpathy | EN | 教程 | 预训练→中训练→SFT→可选 RL 的完整渐进管线，560M 参数 | 2025/10 | [MarkTechPost](https://www.marktechpost.com/2025/10/14/andrej-karpathy-releases-nanochat-a-minimal-end-to-end-chatgpt-style-pipeline-you-can-train-in-4-hours-for-100/) |

## 2.4 技术演进时间线

```
2020 ─┬─ Progressive Layer Dropping (Microsoft) → 训练中逐步减少层数加速训练
      │
2021 ─┬─ CompoundGrow (Google) → 首次联合深度/宽度/序列长度渐进增长, BERT-base 加速 73.6%
      ├─ StackBERT (ICML) → 渐进式堆叠层, 验证深度渐进生长可行性
      │
2022 ─┬─ bert2BERT (清华+华为, ACL) → Function-Preserving Initialization 形式化定义
      ├─ ChatGPT 发布 → 行业认识到 RLHF 后训练的重要性, 推动多阶段训练管线
      │
2023 ─┬─ RaPTr (NeurIPS Workshop) → 渐进式子网络训练, 从部分网络到全网络的平滑过渡
      ├─ LLaMA 系列发布 → 开源社区推动预训练+SFT+RL 的标准化三阶段管线
      ├─ OpenRLHF 开源 → 生产级多阶段 RLHF 框架
      │
2024 ─┬─ DeepSeek-V2 → MoE 架构 + 多阶段训练策略验证
      ├─ GRPO 提出 → 无 critic 模型的 RL 算法, 极大简化后训练管线
      ├─ 中训练 (Mid-Training) 概念兴起 → 预训练与后训练之间的关键桥梁阶段
      │
2025 ─┬─ EACL 2026 (提前发布) → 200+ 模型系统验证课程学习在预训练中的 18-45% 加速
      ├─ CGLS → 首次同步数据难度 + 模型深度增长
      ├─ MIDAS/LIDAS → 中间层插入胜于尾部插入
      ├─ EvoLM (NeurIPS Oral) → 跨 4 阶段的训练动态系统性分析
      ├─ "Train Long, Think Short" → 渐进式 token 预算衰减范式
      ├─ Growing Transformers → 冻结基座上逐层生长, 复杂推理在深度足够时涌现
      ├─ Meta ExIt → 自举式迭代自我改进, 单步训练实现多步改进
      ├─ 美团 Metis-RISE → 先 RL 后 SFT 的倒序训练范式
      │
2026 ─┬─ Efficient Model Family Construction (COLM) → 1B→2B→4B→8B 渐进式家族构建, 计算节省 25%
      ├─ Implicit Curriculum Hypothesis → 技能涌现顺序跨模型高度一致的发现
      ├─ PRISM → 中训练重塑 90%+ 参数而 RL 只改 5%
      ├─ DORA → 异步多版本流式训练, 2.12x 吞吐提升
      ├─ 文心 5.1 → 多维弹性预训练 + 三阶段 RL 管线, 成本降至行业 6%
      ├─ MegaScale-Omni (字节) → 多模态多阶段训练系统, 1.27x-7.57x 吞吐提升
      │
      └─ 当前状态：多阶段渐进式训练已从"单一维度的课程学习"演进为"模型容量、数据难度、
         训练目标的**三维联动渐进框架**"，正朝着自动化调度和端到端联合优化的方向发展。
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
2020 ─┬─ Progressive Layer Dropping → 首次将"渐进"概念引入 Transformer 训练
2021 ─┬─ CompoundGrow → 维度扩展：从单维度(深度)到多维度(深度+宽度+序列长度)
2022 ─┬─ bert2BERT → 理论规范：Function-Preserving 初始化成为模型扩展的标准方法
2023 ─┬─ RaPTr + OpenRLHF → 两条路径分化：模型生长 vs 多阶段后训练管线
2024 ─┬─ GRPO + Mid-Training → 简化 RL + 填补"预训练-后训练"鸿沟
2025 ─┬─ 百花齐放：CGLS(联动)、MIDAS(中插)、EvoLM(系统分析)、DORA(异步)
2026 ─┴─ 当前状态：三维联动成熟化，自动化调度成新焦点
```

## 3.2 六种方案横向对比

### 方案概览

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **A. 课程学习 (Curriculum Learning)** | 按难度由易到难组织训练数据 | ① 实现简单，仅需修改数据采样器；② 加速 18-45% 训练步数（EACL 2026 实证）；③ 可作为 warmup 带来 +3.5% 持续提升；④ 不改变模型架构 | ① 独立效果有限，需配合其他策略；② 难度度量标准不统一；③ 课程设计依赖人工先验；④ 对高资源语言数据效果递减 | 中小规模预训练（≤10B tokens） | 低（仅数据排序开销） |
| **B. 渐进式深度生长 (Progressive Depth Growth)** | 训练中逐步增加 Transformer 层数 | ① 计算节省 20-33%（RaPTr）；② 克服"深度诅咒"，各层利用率更均匀；③ 支持模型家族构建（1B→8B）；④ 与课程学习可联合使用 | ① 实现复杂，需 Function-Preserving 初始化；② 层插入位置选择敏感（中间 vs 尾部）；③ 生长后的训练稳定性需额外维护；④ 深度扩展不如宽度扩展效果稳定 | 大规模预训练、模型家族构建 | 中高（需修改训练框架） |
| **C. 多阶段后训练管线 (Multi-Stage Post-Training)** | SFT → RLHF/DPO → 蒸馏等多阶段顺序或交替组合 | ① 标准化程度高，生态工具成熟；② 各阶段目标明确可独立优化；③ 中训练（Mid-Training）可大幅提升数学/代码能力（+15~+40 分）；④ PRISM 发现中训练重塑 90%+ 参数 | ① 阶段间过渡易丢失前期知识（灾难性遗忘）；② 部分阶段（如 SFT）存在"对齐税"；③ EvoLM 揭示过度后训练收益递减；④ 各阶段的最佳数据配比不明确 | 指令跟随、对齐、推理增强 | 中（复用成熟框架如 OpenRLHF） |
| **D. 渐进式 RL 训练 (Progressive RL Training)** | 在 RL 阶段逐步收紧约束或增加任务难度 | ① "Train Long, Think Short" 范式使模型学会高效推理；② E2H 提供理论收敛保证；③ 模型自适应课程（CCL）去除人工设计；④ 自生成课程（LAIMARK）实现自我改进 | ① 课程调度策略（指数/线性/余弦）选择影响大；② 奖励函数设计复杂；③ 对小型模型（<8B）适用性更好；④ RL 训练本身不稳定，课程增加复杂度 | 推理优化（数学/代码）、对齐训练 | 中（基于 GRPO 较简单） |
| **E. 模型扩展训练 (Model Expansion Training)** | 训练小模型后扩展为大模型继续训练 | ① 计算节省 ~25%（COLM 2025）；② 扩展后模型性能可反超独立训练；③ 天然支持模型家族（多个规模版本）；④ 可复用现有预训练 checkpoint | ① Function-Preserving 有精度损失风险；② 扩展后的训练需要仔细调整学习率；③ 宽度扩展不如深度扩展成熟；④ 扩展时机（训练到什么程度再扩展）难以确定 | 模型家族构建、多规模发布 | 低中（复用已训练权重） |
| **F. 同步联动渐进 (Synchronized Progressive Growth)** | 数据难度 + 模型容量 + 训练目标三维同步演进 | ① CGLS 验证联动优于独立维度（+3.9%）；② 文心 5.1 多维弹性框架验证可行性；③ 最接近人类学习方式；④ 综合收益最高 | ① 实现极度复杂，需同时管理多个维度；② 三维同步的最优配比无理论指导；③ 调试困难，问题定位复杂；④ 目前仅大厂有完整实践 | 超大规模预训练（≥100B tokens） | 高（需完整训练框架改造） |

## 3.3 技术细节对比

| 维度 | A. 课程学习 | B. 渐进式深度生长 | C. 多阶段后训练 | D. 渐进式 RL | E. 模型扩展 | F. 同步联动 |
|------|-----------|----------------|--------------|------------|-----------|-----------|
| **性能加速** | 18-45% 步数减少 | 20-33% FLOPs 节省 | N/A（提升质量） | 推理长度可控 | ~25% 计算节省 | 综合最优 |
| **下游增益** | +2~+3.5% | +3.9%（联动时可） | 数学 +15~+40 | +22% (MLE-bench) | 3B→1B 级提升 | 综合最优 |
| **实现易用性** | ★★★★★ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ |
| **生态成熟度** | ★★★☆☆ | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★☆☆☆☆ | ★☆☆☆☆ |
| **社区活跃度** | ★★★★☆ | ★★☆☆☆ | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★☆☆☆☆ |
| **学习曲线** | 低 | 高 | 中 | 中高 | 高 | 极高 |
| **理论完备性** | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| **可扩展性** | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ |

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **学术实验/原型验证** | A. 课程学习 | 无需修改模型架构，仅调整数据采样器即可验证概念。EACL 2026 提供了可直接使用的难度指标和调度策略。 | 2K-10K USD（GPU 租赁） |
| **中小型生产模型（<10B）** | B/C 组合：课程学习 + 后训练管线 | 先用课程学习加速预训练收敛，然后复用 OpenRLHF 或 Alignment Handbook 做标准后训练。工具链成熟，社区支持好。 | 10K-50K USD |
| **大型生产模型（10B-100B）** | C/D 组合：后训练 + 渐进式 RL + Mid-Training | PRISM 已验证中训练的效果（数学 +15~+40 分）。"Train Long, Think Short" 的渐进式 token 预算衰减有效降低推理成本。使用 GRPO 减少 RL 实现复杂度。 | 50K-300K USD |
| **模型家族构建（多规模发布）** | E. 模型扩展训练 | COLM 2025 方案：1B→2B→4B→8B 渐进扩展，比独立训练节省 25% 计算量。扩展后微调学习率是关键技巧。 | 100K-500K USD（总节省 25%） |
| **超大规模/旗舰模型（>100B）** | F. 同步联动渐进 | 参考文心 5.1 和 CGLS 的思路，在数据、模型、目标三个维度上同步渐进。需要定制训练框架，成本最高但收益也最大。 | 1M-10M+ USD |
| **推理效率优化部署** | D. 渐进式 RL 训练 | "Train Long, Think Short" 范式中，GRPO 渐进式压缩 token 预算。百度文心 5.1 成本降至行业 6% 证明了极端优化的可行性。 | 推理成本降低 40-70% |
| **领域专业化/垂直模型** | A+D 组合：课程学习 + 领域渐进式 RL | ACER 方法使用 Bloom 分类法学课的合成数据做课程学习，再配合领域特定 RL。美团 LongCat-Flash 域并行 RL 也是成功案例。 | 20K-100K USD |

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{多阶段渐进式训练} = \underbrace{\text{课程学习}}_{\text{数据维度：由易到难}} + \underbrace{\text{模型生长}}_{\text{容量维度：由小到大}} - \underbrace{\text{知识遗忘}}_{\text{核心损耗：阶段间的灾难性遗忘}}
$$

这个公式揭示了一个悖论：渐进式训练的本质是在"逐步增加难度"和"逐步扩大容量"之间寻找最优节奏，同时必须不断对抗阶段切换带来的知识遗忘——就像攀岩时既要点到更高的位置，又要确保没有滑落。

## 4.2 一句话解释

大模型渐进式训练就像教一个学生——先学加减法再用计算器、先写短句再写长文、先模仿后创作，而不是第一天就让 TA 写一篇博士论文。

## 4.3 核心架构图

```
训练开始 → [阶段1: 简单数据 + 小模型]
               ↓ (数据难度递增 + 模型层数增加)
          [阶段2: 中等数据 + 中层数模型]
               ↓ (加入指令数据 + 切换到 SFT 目标)
          [阶段3: 复杂数据 + 全模型 + 指令学习]
               ↓ (加入奖励信号 + 切换到 RL 目标)
          [阶段4: 全难度数据 + 全模型 + 偏好对齐]
               ↓
训练完成 → [可部署模型]
```

## 4.4 STAR 总结

### Situation（背景与痛点）

当前大模型训练面临的核心矛盾是"规模与效率"的冲突。以 GPT-4 为代表的千亿参数模型训练成本高达数千万美元，且存在严重的资源浪费——训练中后期约 40% 的 token 对模型学习贡献甚微（Tirumala et al., 2023）。同时，传统"从零训练"模式要求每次从头起步，无法复用已有的小模型训练成果。行业急需在保持最终性能的前提下大幅降低训练计算成本的方法。

### Task（核心问题）

如何在多阶段训练过程中，通过动态调整模型架构、数据分布和训练目标，实现计算效率的最大化？关键约束包括：(1) 保证最终模型性能不低于甚至超越独立训练的基线；(2) 避免阶段切换带来的知识遗忘；(3) 自动化调度各阶段的转换时机和参数配比，减少人工调优成本。

### Action（主流方案）

2025-2026 年，该领域经历了从单维度到多维度联动的重要演进。早期工作（CompoundGrow 2021、bert2BERT 2022）专注于模型架构的渐进生长。2023-2024 年随着 RLHF 的成熟，多阶段后训练管线（SFT→RL→蒸馏）成为行业标准。2025 年 CGLS 首次实现了数据难度与模型深度的同步联动，MIDAS 发现了"中间层插入"优于"尾部插入"的重要规律。2026 年的关键突破包括：隐式课程假设的提出（技能涌现具跨模型一致性）、PRISM 对中训练的系统性解析（重塑 90%+ 参数）、以及 Efficient Model Family Construction（从 1B 渐进扩展到 8B，计算节省 25%）。

### Result（效果与建议）

当前最佳实践表明：(1) 中等规模预训练中，课程学习可减少 18-45% 训练步数；(2) 对于 10B+ 模型的家族构建，渐进式模型扩展可节省约 25% 的计算量；(3) 中训练（Mid-Training）是性价比最高的质量提升手段（+15-40 分）。(4) 三维联动渐进（数据+模型+目标同步演进）代表未来方向，但实现复杂度极高，目前仅大厂具备完整实践能力。**实操建议**：对中小团队，优先采用"课程学习 + OpenRLHF 后训练管线"的组合；对大型团队，尝试将中训练纳入标准训练流程并关注 CGLS 类联动策略。

## 4.5 理解确认问题

**Q**: 如果一家 AI 公司想在 8 个 GPU 上以最低成本训练一个 7B 模型，它应该如何设计训练策略？请说明你的方案选择的依据。

**参考答案要点**:
1. **数据层面**：使用难度分层（按压缩率/MTLD 排序），前 30% 训练步数仅使用简单数据（warmup 式课程学习），后续恢复随机采样。根据 EACL 2026 经验，这样可减少约 20% 训练步数。
2. **模型层面**：不从零训练 7B 模型，而是先训练一个 1B 模型收敛后，通过 Function-Preserving 初始化扩展为 7B 模型继续训练（借鉴 COLM 2025 方法）。这样前期 1B 训练速度快 7 倍，总计算量节省约 25%。
3. **训练目标层面**：预训练完成后，加入中训练阶段（对数学和代码数据继续预训练），然后使用 OpenRLHF 进行 SFT + GRPO 对齐。PRISM 表明中训练可加+15~+40 分。
4. **总成本估算**：相比独立训练 7B，该方案可节省约 30-40% 总计算量，约等效于 5-6 个 GPU 跑完全程。

---

# 附录

## A. 数据来源汇总

| 类别 | 数量 | 数据日期范围 |
|------|------|------------|
| GitHub 项目 | 16 个 | 2021-2026 |
| 学术论文 | 18+ 篇 | 2020-2026 |
| 技术博客 | 11 篇 | 2025-2026 |
| 行业实践 | 6 项 | 2025-2026 |

## B. 缩略语对照

| 缩写 | 全称 |
|------|------|
| CGLS | Curriculum-Guided Layer Scaling |
| GRPO | Group Relative Policy Optimization |
| RLHF | Reinforcement Learning from Human Feedback |
| RLAIF | Reinforcement Learning from AI Feedback |
| SFT | Supervised Fine-Tuning |
| DPO | Direct Preference Optimization |
| MIDAS | Middle Insertion Depth-wise Architecture Scaling |
| POCL | Progressive Overload Curriculum Learning |
| CCL | Customized Curriculum Learning |
| FPI | Function-Preserving Initialization |

---

*本报告基于截至 2026-05-13 的公开信息撰写，数据来源包括 GitHub、arXiv、Google Scholar、各大学术会议论文集及技术博客。文中所有影响力和指标数据均为调研时的快照值，可能随时间变化。*
