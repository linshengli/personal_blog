# 大模型稀疏训练：计算效率与精度平衡

> 调研日期：2026-05-22 | 所属领域：大模型训练

---

## 第一部分：概念剖析

### 1.1 定义澄清

**通行定义**：大模型稀疏训练是指在训练或推理过程中，通过选择性激活模型的部分参数（而非全部参数），在保持模型表达能力的同时显著降低计算和存储开销的技术体系。其核心思想是利用神经网络中天然存在的冗余——大量权重接近零、大量神经元输出为零或接近零——通过结构性手段"跳过"这些冗余计算。

**常见误解**：

1. **"稀疏 = 剪枝"**：剪枝（Pruning）是稀疏化的一种手段，但并非全部。稀疏训练还包括稀疏激活（如 ReLU 天然稀疏）、Mixture-of-Experts（MoE）路由稀疏、稀疏注意力等多种形式。剪枝是"参数级的稀疏"，MoE 是"架构级的稀疏"。

2. **"稀疏一定降低精度"**：最新研究表明，在适当的稀疏度下（如 50% 结构化稀疏），经精心设计的稀疏模型可以达到甚至超越稠密模型的精度。SparseForge 在 2:4 稀疏度下以 5B tokens 重训练即超越稠密基线（57.27% vs 56.43% 零样本准确率）。

3. **"稀疏度越高越好"**：稀疏度与精度之间存在非线性的"悬崖效应"——在某些临界点（如 70% 以上），精度可能急剧下降。不同层对稀疏的容忍度差异极大，需要非均匀分配才能达到最优平衡。

4. **"稀疏训练的加速立即可见"**：非结构化稀疏在通用 GPU 上几乎没有实际加速效果（需要特殊硬件或内核支持）。只有结构化稀疏（2:4、N:M）或 MoE 的路由稀疏才能在现有硬件上获得真实加速。

**边界辨析**：

| 邻近概念 | 与稀疏训练的核心区别 |
|---------|------------------|
| **量化（Quantization）** | 量化降低每个参数的"位宽"（如 FP16→INT4），而稀疏减少"活跃参数的数量"。两者正交可叠加（如 OBR 实现 W4A4KV4 + 50% 稀疏）。 |
| **知识蒸馏（Distillation）** | 蒸馏通过训练小模型模仿大模型输出，本质是模型压缩而非计算稀疏化。稀疏训练保持模型架构不变，仅选择性激活参数。 |
| **低秩分解（Low-Rank）** | 低秩分解将权重矩阵分解为两个小矩阵的乘积，减少参数量。稀疏训练不改变矩阵维度，仅使矩阵变"空"（含大量零元素）。 |

---

### 1.2 核心架构

```
┌────────────────────────────────────────────────────────────┐
│              大模型稀疏训练系统架构                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  训练阶段:                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────────┐ │
│  │ 前向传播  │ → │ 稀疏掩码  │ → │ 反向传播  │ → │ 参数更新 │ │
│  │ (仅激活   │   │ 动态调整  │   │ (梯度通   │   │ (剪枝+  │ │
│  │ 子集参数) │   │ 或固定)  │   │ 过掩码)  │   │ 生长)   │ │
│  └──────────┘   └──────────┘   └──────────┘   └─────────┘ │
│       │              │              │              │       │
│       ▼              ▼              ▼              ▼       │
│  ┌────────────────────────────────────────────────────┐    │
│  │              稀疏模式控制器                          │    │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐          │    │
│  │  │Weight│  │Activ.│  │Attn. │  │MoE   │          │    │
│  │  │Prune │  │Prune │  │Mask  │  │Route │          │    │
│  │  └──────┘  └──────┘  └──────┘  └──────┘          │    │
│  └────────────────────────────────────────────────────┘    │
│                                                            │
│  推理阶段:                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ 输入编码  │ → │ 稀疏计算  │ → │ 输出解码  │               │
│  │          │   │ (跳过零)  │   │          │               │
│  └──────────┘   └──────────┘   └──────────┘               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

**组件说明**：

| 组件 | 职责 |
|------|------|
| **稀疏掩码** | 定义哪些参数/神经元/注意力头是活跃的，是稀疏化的"地图" |
| **稀疏模式控制器** | 管理多种稀疏模式（权重剪枝、激活稀疏、注意力稀疏、MoE 路由），决定何时采用何种策略 |
| **动态调整** | 在训练过程中根据损失变化、梯度信息动态调整稀疏模式（如 DST 的动态稀疏训练） |
| **稀疏计算内核** | 在硬件层面跳过零元素计算，只有结构化稀疏（2:4、N:M）才能在此获得真实加速 |

---

### 1.3 数学形式化

#### 公式 1：权重稀疏的掩码操作

$$
\mathbf{W}_{\text{sparse}} = \mathbf{W} \odot \mathbf{M}, \quad M_{ij} \in \{0, 1\}, \quad s = \frac{\sum_{i,j} \mathbb{1}[M_{ij}=0]}{d_{\text{out}} \cdot d_{\text{in}}}
$$

*解释：对权重矩阵 $\mathbf{W}$ 应用二进制掩码 $\mathbf{M}$，$s$ 为稀疏度（零元素占比）。这个简单的点乘操作是所有稀疏化方法的基础。*

#### 公式 2：SparseGPT 的逐层误差补偿

$$
\min_{\delta\mathbf{W}}\|\mathbf{W}\mathbf{X} - (\mathbf{W}+\delta\mathbf{W})_{\text{sparse}}\mathbf{X}\|_F^2 \approx \frac{(\mathbf{W}_{q,:} - \mathbf{W}_{p,:})^2}{[\mathbf{H}^{-1}]_{qq}}
$$

*解释：SparseGPT 的核心——在剪枝第 $q$ 个权重时，用最优脑外科医生（OBS）框架估计其对输出的影响，其中 $\mathbf{H} = 2\mathbf{X}\mathbf{X}^{\top}$ 是 Hessian 矩阵近似。分母 $[\mathbf{H}^{-1}]_{qq}$ 刻画了该权重的"可补偿性"。*

#### 公式 3：MoE 的稀疏路由策略

$$
y = \sum_{i=1}^{E} G(x)_i \cdot E_i(x), \quad G(x) = \text{TopK}(\text{Softmax}(\mathbf{W}_g x + \epsilon\cdot\text{Softplus}(\mathbf{W}_n x)), k)
$$

*解释：MoE 通过一个轻量级门控网络 $G$ 将输入 $x$ 路由到 $E$ 个专家中得分最高的 $k$ 个（通常 $k \ll E$）。$\epsilon\cdot\text{Softplus}(\mathbf{W}_n x)$ 是用于负载均衡的噪声项。活跃参数比例为 $k/E$。*

#### 公式 4：精度-效率的帕累托前沿

$$
\mathcal{L}_{\text{sparse}}(s) = \mathcal{L}_{\text{dense}} + \alpha s^\beta, \quad \text{Cost}(s) = (1-s) \cdot \text{Cost}_{\text{dense}}
$$

*解释：稀疏训练的核心权衡——精度损失 $\mathcal{L}_{\text{sparse}}(s)$ 随稀疏度 $s$ 超线性增长（$\beta > 1$），而计算成本 $\text{Cost}(s)$ 线性下降。最优操作点是在 $s$ 空间中寻找 $\mathcal{L}$ 与 $\text{Cost}$ 的平衡点。Google DeepMind 的 Compression Scaling Laws 进一步揭示 $\alpha$ 和 $\beta$ 与模型规模、数据量的幂律关系。*

#### 公式 5：动态稀疏训练（DST）的梯度信号保留

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \bigg|_{\text{sparse}} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}} \odot \mathbf{M} \quad \text{且} \quad \mathbf{M}_{t+1} = f_{\text{evolve}}(\mathbf{M}_t, \nabla_{\mathbf{W}}\mathcal{L})
$$

*解释：DST 的核心——梯度只在稀疏子集上传播（掩码 $\mathbf{M}$），同时掩码本身随训练动态演化：定期剪掉不重要权重，生长新的连接。这使得稀疏结构本身也成为学习的对象。*

---

### 1.4 实现逻辑（Python 伪代码）

```python
import torch
import torch.nn as nn

class SparseLinear(nn.Module):
    """稀疏线性层 - 体现稀疏训练的核心抽象"""
    def __init__(self, in_features: int, out_features: int, sparsity: float = 0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.sparsity = sparsity
        # 可学习/固定的二进制掩码
        self.register_buffer('mask', self._init_mask(sparsity))

    def _init_mask(self, sparsity: float) -> torch.Tensor:
        """初始化稀疏掩码"""
        mask = torch.ones_like(self.weight)
        n_zeros = int(self.weight.numel() * sparsity)
        flat_idx = torch.topk(self.weight.abs().view(-1), n_zeros, largest=False).indices
        mask.view(-1)[flat_idx] = 0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 前向：仅激活掩码对应参数
        return nn.functional.linear(x, self.weight * self.mask)


class DynamicSparseTrainer:
    """动态稀疏训练器 - 体现稀疏训练的关键算法逻辑"""
    def __init__(self, model: nn.Module, sparsity: float = 0.7,
                 grow_fraction: float = 0.3, dense_update_interval: int = 100):
        self.model = model
        self.target_sparsity = sparsity
        self.grow_fraction = grow_fraction
        self.interval = dense_update_interval
        self.step_count = 0

    def step(self, loss: torch.Tensor):
        """执行一次稀疏训练步骤"""
        loss.backward()
        self.step_count += 1

        # 定期更新稀疏结构（剪枝+生长）
        if self.step_count % self.interval == 0:
            for module in self.model.modules():
                if isinstance(module, SparseLinear):
                    self._update_sparsity(module)

    def _update_sparsity(self, layer: SparseLinear):
        """核心算法：基于梯度的稀疏结构演化"""
        with torch.no_grad():
            # 1. 计算重要性分数（权重 × 梯度幅值）
            importance = layer.weight.abs() * layer.grad.abs()

            # 2. 按分数排序，保留最重要参数
            n_keep = int(layer.weight.numel() * (1 - self.target_sparsity))
            n_current_active = layer.mask.sum().item()
            n_prune = int(n_current_active * self.grow_fraction)
            n_grow = n_prune  # 保持总稀疏度不变

            # 3. 剪枝：移除当前活跃中最不重要的 n_prune 个
            active_importance = importance * layer.mask
            prune_idx = torch.topk(active_importance.view(-1), n_prune,
                                   largest=False).indices
            layer.mask.view(-1)[prune_idx] = 0

            # 4. 生长：在当前不活跃中最可能被需要的 n_grow 个
            inactive_importance = importance * (1 - layer.mask)
            grow_idx = torch.topk(inactive_importance.view(-1), n_grow,
                                  largest=True).indices
            layer.mask.view(-1)[grow_idx] = 1


class MoELayer(nn.Module):
    """混合专家层 - 架构级稀疏的体现"""
    def __init__(self, n_experts: int = 8, top_k: int = 2, d_model: int = 768):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        # 轻量级路由网络
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 仅激活 top-k 个专家，实现架构级稀疏
        scores = self.gate(x)                   # [batch, n_experts]
        weights = torch.softmax(scores, dim=-1)
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k,
                                                  dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 仅计算 top-k 个专家
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i).any(dim=-1)
            if mask.any():
                output[mask] += expert(x[mask]) * \
                    top_k_weights[mask][:, (top_k_indices[mask] == i).float()]
        return output
```

---

### 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 稀疏度 (Sparsity) | 50%-95% | 零参数占比 = 零权重数/总权重数 | 基础指标，但单独使用具有误导性 |
| 精度保持率 | > 95% | 下游任务准确率 / 稠密基线准确率 | 衡量稀疏对能力的实际影响 |
| 理论加速比 | (1-s)× | FLOPs 减少比例 | 仅反映理想情况 |
| 实际加速比 | 1.2×-4.7× | 端到端推理或训练时间对比 | 依赖硬件和稀疏模式 |
| 内存压缩比 | 2×-8× | 模型参数内存 / 稀疏模型内存 | OBR 在 50%稀疏+4bit 下达 6.4× |
| 重训练效率 | < 5B tokens | 恢复至稠密精度所需的 token 数 | SparseForge 仅需 5B tokens |
| 困惑度退化 | ΔPPL < 1.0 | 稀疏前后 PPL 差值 | 生成任务的通用衡量 |

---

### 1.6 扩展性与安全性

**水平扩展**：
- **MoE 的专家并行**：不同专家可分配到不同 GPU 上独立计算，路由层协调负载。DeepSpeed-MoE 支持千卡级专家并行训练。
- **数据并行 + 模型并行**：稀疏模型可结合 ZeRO-3 等优化策略，在保持稀疏结构的同时扩大数据并行规模。FSMoE 在 DeepSpeed-MoE 基础上再提升 1.2-3.0×。
- **动态稀疏训练的分布式挑战**：参数剪枝/生长需要在各 GPU 间同步掩码状态，通信开销随卡数增长。

**垂直扩展**：
- **单节点上限**：2:4 半结构化稀疏是 NVIDIA GPU 原生支持的稀疏模式，可提供约 2× 加速，理论 FLOPS 上限受限于 Tensor Core 的稀疏计算能力。
- **精度-稀疏度曲线**：模型越大，对高稀疏度的容忍度越高（Llama-4-Maverick 400B 在 90% 激活稀疏度下仍能保持鲁棒性）。
- **硬件利用瓶颈**：非结构化稀疏在 GPU 上无法利用 Tensor Core，退化为基础 CUDA 内核计算，实际加速比远低于理论值。

**安全考量**：
- **稀疏对抗攻击**：稀疏模型可能更易受到结构化对抗扰动——攻击者可能针对特定稀疏模式设计对抗样本，干扰模型的关键稀疏路径。
- **信息泄露**：稀疏掩码本身可能泄露训练数据信息。可学习掩码（如 MaskLLM）的分布可能被逆向工程。
- **计算侧信道**：不同稀疏模式的计算时间差异可被用作侧信道，推断输入属性（如 SecretGen 攻击）。
- **鲁棒性退化**：高稀疏度下模型的泛化边界会收缩，专家路由模型在 OOD 样本上可能表现不稳定。

---

## 第二部分：行业情报

### 2.1 GitHub 热门项目

| 项目 | Stars（≈） | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-----------|---------|--------|---------|------|
| Intel Neural Compressor | 2,580 | SOTA 低比特量化 + 稀疏性，LLM 全流程压缩 | Python/PyTorch/ONNX | 2026-02 | [github.com/intel/neural-compressor](https://github.com/intel/neural-compressor) |
| Neural Magic SparseML | 2,136 | 稀疏化配方（pruning+quant+distill） | Python/PyTorch/TF | 2025 活跃 | [github.com/neuralmagic/sparseml](https://github.com/neuralmagic/sparseml) |
| PaddleSlim | 1,535 | 深度模型压缩 + 架构搜索 | Python/PaddlePaddle | 2025 活跃 | [github.com/PaddlePaddle/PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) |
| TF Model Optimization | 1,472 | TF/Keras 量化与剪枝工具包 | Python/TensorFlow | 2025 | [github.com/tensorflow/model-optimization](https://github.com/tensorflow/model-optimization) |
| IST-DASLab SparseGPT | 879 | 开创性 one-shot LLM 剪枝（ICML 2023） | Python/PyTorch | 近期维护 | [github.com/ist-daslab/sparsegpt](https://github.com/ist-daslab/sparsegpt) |
| OpenVINO NNCF | 772 | 神经网络压缩框架 | Python/OpenVINO | 2025 活跃 | [github.com/openvinotoolkit/nncf](https://github.com/openvinotoolkit/nncf) |
| NVIDIA TensorRT-Model-Optimizer | 704 | 统一量化+剪枝+蒸馏+稀疏化 | Python/TensorRT | 2025 活跃 | [github.com/NVIDIA/TensorRT-Model-Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) |
| locuslab Wanda | 550+ | 简单高效的 LLM 剪枝（ICLR 2024） | Python/PyTorch | 2024 | [github.com/locuslab/wanda](https://github.com/locuslab/wanda) |
| Princeton LLM-Pruner | 500+ | 结构化剪枝（NeurIPS 2023） | Python/PyTorch | 2024 | [github.com/horseee/LLM-Pruner](https://github.com/horseee/LLM-Pruner) |
| Princeton LLM-Shearing | 640+ | 结构化剪枝加速预训练（ICLR 2024） | Python/PyTorch | 2024 | [github.com/princeton-nlp/LLM-Shearing](https://github.com/princeton-nlp/LLM-Shearing) |
| FMInference H2O | 319 | Heavy-Hitter Oracle 注意力稀疏化 | Python/PyTorch | 2024 | [github.com/FMInference/H2O](https://github.com/FMInference/H2O) |
| NVlabs MaskLLM | 155+ | 可学习半结构化稀疏（NeurIPS 2024 Spotlight） | Python/Megatron | 2025 | [github.com/NVlabs/MaskLLM](https://github.com/NVlabs/MaskLLM) |
| ETH OBR | 近期开源 | W4A4KV4 + 50% 稀疏联合压缩（ICLR 2026） | Python/PyTorch | 2026-01 | [github.com/csguoh/OBR](https://github.com/csguoh/OBR) |
| LLM-Pruning Collection | 新项目 | JAX 统一剪枝框架（Minitron/ShortGPT/Wanda 等） | Python/JAX | 2026-01 | [zlab-princeton/llm-pruning-collection](https://github.com/zlab-princeton/llm-pruning-collection) |
| STOP | 新项目 | 结构化推理剪枝（DeepSeek-R1 蒸馏优化） | Python/PyTorch | 2026 | [github.com/chenjux/ECN-STOP](https://github.com/chenjux/ECN-STOP) |

---

### 2.2 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/来源 | 核心贡献 | 链接 |
|------|----------|------|----------|---------|------|
| **SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot** | Frantar & Alistarh (IST) | 2023 | ICML 2023 | 首次证明百亿参数 LLM 可在单次前向传播中完成剪枝而几乎不损失精度 | [arXiv:2301.00774](https://arxiv.org/abs/2301.00774) |
| **A Simple and Effective Pruning Approach for LLMs (Wanda)** | Sun et al. (CMU, Meta) | 2023 | ICLR 2024 | 仅凭权重幅值 × 激活范数剪枝，方法极简但效果接近 SparseGPT | [arXiv:2306.11695](https://arxiv.org/abs/2306.11695) |
| **Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning** | Xia et al. (Princeton) | 2023 | ICLR 2024 | 结构化剪枝 + 动态数据加载，用 3% 算力训出同等规模 SOTA 模型 | [arXiv:2310.06694](https://arxiv.org/abs/2310.06694) |
| **MaskLLM: Learnable Semi-Structured Sparsity for LLMs** | Fang et al. (NVIDIA) | 2024 | NeurIPS 2024 Spotlight | Gumbel Softmax 可学习 N:M 稀疏掩码，支持跨任务迁移 | [arXiv:2409.17481](https://arxiv.org/abs/2409.17481) |
| **Compression Scaling Laws: Unifying Sparsity and Quantization** | Frantar et al. (Google DeepMind, IST) | 2025 | — | 统一稀疏和量化的缩放定律，提出"有效参数量"乘子 | [arXiv:2502.16440](https://arxiv.org/abs/2502.16440) |
| **The Journey Matters: Average Parameter Count over Pre-training Unifies Sparse and Dense Scaling Laws** | Jin et al. (Google DeepMind) | 2025 | — | 系统性探索 80+ 剪枝调度策略，用平均参数量统一稀疏/稠密缩放定律 | [arXiv:2501.12486](https://arxiv.org/abs/2501.12486) |
| **Optimal Brain Restoration (OBR)** | Guo et al. (ETH Zürich) | 2025 | ICLR 2026 | 首个 W4A4KV4 + 50% 稀疏联合压缩框架，闭式误差补偿 | [arXiv:2509.11177](https://arxiv.org/abs/2509.11177) |
| **The Unseen Frontier: Pushing Limits of LLM Sparsity (Elsa)** | IST | 2025 | ICLR 2026 | ADMM 无代理约束优化实现 90% 极致稀疏，PPL 比已有方法低 7.8× | [arXiv:2510.01650](https://arxiv.org/abs/2510.01650) |
| **SparseForge: Efficient Semi-Structured LLM Sparsification** | — | 2026 | — | Hessian 引导渐进退火，5B tokens 恢复 2:4 稀疏精度甚至超越稠密 | [arXiv:2605.06402](https://arxiv.org/abs/2605.06402) |
| **Uncovering Intra-expert Activation Sparsity for Efficient MoE** | UC Berkeley & AMD | 2026 | — | 发现预训练 MoE 模型内蕴含 90% 激活稀疏性，MoE 层加速 2.5× | [arXiv:2605.08575](https://arxiv.org/abs/2605.08575) |
| **MTraining: Distributed Dynamic Sparse Attention** | — | 2025/2026 | — | 动态稀疏注意力 + 分布式训练，Qwen2.5-3B 从 32K 扩展至 512K 上下文，吞吐提升 6× | [arXiv:2510.18830](https://arxiv.org/abs/2510.18830) |
| **ELAS: Efficient Pre-Training via 2:4 Activation Sparsity** | — | 2026 | — | Squared ReLU + 2:4 激活稀疏用于低秩 LLM 预训练 | [arXiv:2605.03667](https://arxiv.org/abs/2605.03667) |

---

### 2.3 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| **The Journey Matters: Average Parameter Count over Pre-training** | Google DeepMind Blog | EN | 研究论文 | 稀疏/稠密缩放定律统一框架 | 2025-01 | [arXiv](https://arxiv.org/abs/2501.12486) |
| **Compression Scaling Laws: Unifying Sparsity and Quantization** | Google DeepMind Blog | EN | 研究论文 | 统一稀疏与量化的缩放定律 | 2025-02 | [arXiv](https://arxiv.org/abs/2502.16440) |
| **Circuits Updates - January 2025** | Anthropic | EN | 技术报告 | 稀疏自编码器训练改进，JumpReLU + tanh 稀疏惩罚 | 2025-01 | [transformer-circuits.pub](https://transformer-circuits.pub/2025/january-update/) |
| **MoE Fundamentals: Why Sparse Models Are the Future of AI** | Cerebras | EN | 技术指南 | MoE 核心原理、实践指南、与稠密模型的对比 | 2025 | [cerebras.ai](https://www.cerebras.ai/blog/moe-guide-why-moe) |
| **Decoding DeepSpeed-MoE Distributed Gradient Processing** | CSDN 深度解析 | ZH | 工程实践 | DeepSpeed-MoE 梯度分区、ZeRO 集成、通信优化 | 2025-06 | [CSDN](https://haihaihai.blog.csdn.net/article/details/148453770) |
| **From Dense to Mixture of Experts: The New Economics of AI Inference** | Signal65 | EN | 行业分析 | MoE 推理经济性分析，B200 vs MI355X 对比 | 2025-2026 | [signal65.com](https://signal65.com/research/ai/from-dense-to-mixture-of-experts-the-new-economics-of-ai-inference/) |
| **DeepSpeed System Optimization & Compression for LLMs** | 百度开发者 | ZH | 工程实践 | DeepSpeed 量化、稀疏化、ZeRO-3 全栈优化 | 2025-09 | [百度开发者社区](https://developer.baidu.com/article/detail.html?id=3728734) |
| **千亿级稀疏大模型新突破：基于 MoE 架构的实践解析** | 百度开发者 | ZH | 行业实践 | 146B 参数 MoE（15% 激活），92% 精度保持 + 6× 成本降低 | 2025 | [百度开发者社区](https://developer.baidu.com/article/detail.html?id=6821896) |
| **Robust Training of Neural Networks at Arbitrary Precision and Sparsity** | Google DeepMind | EN | 研究论文 | 任意精度+任意稀疏度的统一训练框架，稳定 A1W1 训练 | 2026-03 | [arXiv:2409.09245](https://arxiv.org/abs/2409.09245) |
| **ZeRO Optimization Strategies for Large-Scale Model Training** | HuggingFace 社区 | EN | 工程实践 | ZeRO 各阶段在 H100 上的性能基准测试与选型推荐 | 2025-09 | [HuggingFace Blog](https://huggingface.co/blog/josh-a/zero-optimization-strategies) |

---

### 2.4 技术演进时间线

```
2016 ── Deep Compression (Han et al.)：剪枝+量化+哈夫曼编码的三阶段压缩框架 → 奠定现代模型压缩方法论基础
2017 ──  Lottery Ticket Hypothesis (Frankle & Carbin)：发现"中奖彩票"子网络可以独立训练至全模型精度 → 开启稀疏训练理论新方向
2019 ──  Switch Transformer (Google)：首个大规模 MoE 语言模型（1.6T 参数），引入 Top-1 路由 → 架构级稀疏进入主流视野
2020 ──  GSP (Rigging the Lottery)：首次将 LTH 引入 Transformer → 稀疏训练开始关注大模型
2021 ──  GLaM (Google)：用 64 专家的 MoE 架构在 1/3 能耗下超越 GPT-3 → 验证了 MoE 规模化的可行性
2022 ──  DST (Dynamic Sparse Training)：参数剪枝与生长动态交替 → 打破静态稀疏，结构本身可学习
2023 ──  SparseGPT (IST)：首次实现百亿参数 LLM 的 one-shot 剪枝（ICML 2023） → 剪枝不再需要重训练
      ──  Mixtral 8x7B (Mistral)：首个成功的开源 MoE 模型 → MoE 普及化
2024 ──  Wanda (ICLR 2024)：极简方法的惊人效果 → 重新审视复杂方法的必要性
      ──  LLM-Shearing / Sheared LLaMA (ICLR 2024)：结构化剪枝 + 动态数据加载 → 剪枝可用于加速预训练
      ──  MaskLLM (NeurIPS 2024 Spotlight)：可学习 N:M 稀疏 → 稀疏掩码本身可跨任务迁移
      ──  DeepSeek-V2：创新 MoE 架构（Multi-Head Latent Attention + DeepSeekMoE） → 国产 MoE 路线崛起
2025 ──  Compression Scaling Laws (Google DeepMind)：稀疏/量化的统一缩放律 → 稀疏训练有了理论基础
      ──  MoE vs Dense 系统化对比研究：公平比较下的条件性结论 → 从业者有了选型依据
      ──  Elsa (ICLR 2026)：ADMM 驱动 90% 极致稀疏 → 突破 50-60% 的稀疏上限
      ──  OBR (ICLR 2026)：W4A4KV4 + 50% 稀疏联合压缩 → 稀疏+量化的最优结合
      ──  Intra-Expert Activation Sparsity：发现 MoE 专家内 90% 天然稀疏 → 开辟新优化维度
      ──  MLPerf Training v6.0 GPT-OSS 20B MoE Benchmark：首个标准化稀疏预训练基准 → 行业标准建立
2026 ──  SparseForge：5B tokens 恢复超越稠密 → 微调成本断崖式下降
      ──  Full Attention Strikes Back：100 训练步内从全注意力到稀疏注意力 → 训练极度高效化
      ──  DECO MoE：20% 激活匹配稠密性能 → 边缘端 MoE 可行
      ──  当前状态：稀疏训练已从"能否做"进入"如何做得更好"阶段，精度-效率平衡的帕累托前沿正在被快速推进
```

---

## 第三部分：方案对比

### 3.1 历史发展时间线

```
2020 前 ── 权重剪枝（Magnitude Pruning / L1 正则化）→ 简单直接但效果有限
2021-2022 ── 结构化剪枝 + LTH（彩票假说）→ 理论突破但实践成本高
2023 ── SparseGPT / Wanda：One-shot 后训练剪枝 → 成本巨降，普及化
2023-2024 ── MoE 大规模应用（Mixtral, DeepSeek, Qwen-MoE）→ 架构级稀疏成为主流
2024-2025 ── N:M 半结构化稀疏 + 可学习掩码 → 硬件友好的稀疏模式
2025-2026 ── 统一框架（联合剪枝+量化+蒸馏）+ 极致稀疏（90%+）→ 精度-效率平衡进入新阶段
2026 ── 当前状态：从"单点技术"走向"系统工程"，多种稀疏技术的组合使用成为标配
```

### 3.2 六种方案横向对比

#### 方案 A：One-Shot 后训练剪枝（SparseGPT / Wanda）

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **One-Shot 剪枝** | 基于 Hessian（SparseGPT）或 权重×激活（Wanda）进行一次性剪枝，无需重训练 | 1. 极高效率（分钟级完成剪枝）<br>2. 无需额外训练数据（少量校准集）<br>3. 在 50% 稀疏度下几乎无损 | 1. 高稀疏度（>70%）精度崩溃<br>2. 非结构化稀疏无硬件加速<br>3. 不支持动态结构调整 | 快速实验、小规模部署、模型压缩基线 | 1 次推理 × 少量校准数据 |

#### 方案 B：结构化剪枝 + 重训练（Sheared LLaMA / LLM-Pruner）

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **结构化剪枝+重训练** | 按结构单元（层/头/神经元）整体剪除，然后重训练恢复 | 1. 真实硬件加速（无需特殊内核）<br>2. 可压缩为任意目标形状<br>3. 预训练加速效果显著（3% 算力达相同性能） | 1. 需要大量重训练数据和计算<br>2. 剪枝不可逆（一旦剪除无法恢复）<br>3. 剪枝策略（剪哪层哪头）需要启发式搜索 | 模型小型化、边缘端部署、预训练加速 | 原始训练的 3%-10% 计算量 + 数十亿 tokens |

#### 方案 C：动态稀疏训练（DST / RigL / SET）

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **动态稀疏训练** | 训练过程中定期剪枝不重要的参数并生长新的连接，稀疏结构动态演化 | 1. 稀疏结构自适应学习<br>2. 同等 FLOPs 下精度优于固定稀疏<br>3. 从稀疏初始化开始训练，降低总计算量 | 1. 工程实现复杂<br>2. 掩码更新的通信/计算开销<br>3. 对超参数敏感（剪枝间隔、剪枝率、生长率） | 大规模分布式训练、定制化稀疏模式 | 等于或略高于同等活跃参数的稠密训练 |

#### 方案 D：混合专家模型（MoE - Switch Transformer / Mixtral / DeepSeekMoE）

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **MoE 架构** | 多个专家网络 + 路由门控，每个 token 仅激活 K 个专家 | 1. 参数量与计算量解耦<br>2. 相同 FLOPs 下精度显著优于同等活跃参数稠密模型<br>3. 可扩展到万亿参数级别 | 1. 总参数量巨大，部署存储成本高<br>2. 负载不均衡导致部分专家"饿死"<br>3. 批处理场景下专家缓存/通信瓶颈<br>4. 小批量推理时内存带宽受限 | 大规模预训练、API 服务、前沿研究 | 训练：同活跃参数量稠密模型；推理：依赖 batch size |

#### 方案 E：半结构化 N:M 稀疏（MaskLLM / 2:4 稀疏）

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **N:M 稀疏** | 每连续 M 个权重中恰好保留 N 个非零值，利用 Tensor Core 原生加速 | 1. 唯一在 GPU 上真实加速的模式（2×）<br>2. N:M 模式可通过稀疏矩阵乘法高效执行<br>3. 与 INT4/FP8 量化正交组合 | 1. 稀疏度受限（2:4 仅 50% 稀疏）<br>2. 需要专用硬件支持（NVIDIA Ampere+）<br>3. 训练时需要额外约束保证 N:M 格式 | GPU 推理加速、与量化联合使用 | 额外训练开销（MaskLLM 需学习 N:M 分布） |

#### 方案 F：联合压缩（OBR / Prune-Quantize-Distill Pipeline）

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **联合压缩** | 剪枝 + 量化 + 蒸馏按序结合，Hessian 引导联合优化 | 1. 极致压缩效果（6.4× 内存减少 + 4.7× 加速）<br>2. 剪枝和量化互为正交增益<br>3. OBR 实现了闭式解，训练-free | 1. 技术组合复杂，调试困难<br>2. 最佳执行顺序需经验（Prune→Quantize→Distill）<br>3. 联合优化的搜索空间组合爆炸 | 极端资源受限部署、移动端、IoT 设备 | 等于各技术独立成本的累加，但总效果 > 独立效果之和 |

---

### 3.3 技术细节对比

| 维度 | One-Shot 剪枝 | 结构化剪枝+重训练 | 动态稀疏训练 | MoE | N:M 稀疏 | 联合压缩 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **最大稀疏度（无损）** | 50% | 50%-70% | 60%-80% | 80%-95%（专家级） | 50%（2:4） | 50%+4bit（当量 87.5%） |
| **训练成本** | 极低 | 中等 | 高 | 高 | 中-高 | 中等 |
| **硬件加速** | ❌（非结构化） | ✅ | ❌（非结构化） | ⚠️（依赖通信） | ✅（Tensor Core） | ✅（结构化+N:M） |
| **部署便利性** | 高 | 中 | 低 | 中 | 高 | 中 |
| **生态成熟度** | 高（SparseGPT/Wanda） | 中（LLM-Shearing） | 低（研究阶段） | 高（DeepSpeed-MoE） | 中（MaskLLM） | 低（新兴） |
| **社区活跃度** | 🌟🌟🌟🌟 | 🌟🌟🌟 | 🌟🌟 | 🌟🌟🌟🌟🌟 | 🌟🌟🌟 | 🌟🌟🌟 |
| **学习曲线** | 低 | 中 | 高 | 高 | 中 | 中-高 |

---

### 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证（<7B 模型）** | One-Shot 剪枝（Wanda）+ INT8 量化 | 极低实施成本，分钟级完成，无需 GPU 集群。先验证稀疏可行性再考虑更复杂的方案 | $200-$500（单 GPU 实例 + 少量校准数据） |
| **中型生产环境（7B-70B 模型，延迟敏感）** | N:M 半结构化稀疏（MaskLLM）+ FP8/INT4 量化 | 在 NVIDIA H100/B200 上获得真实硬件加速（~2×），配合量化可达 4× 以上端到端加速。生态成熟，NVIDIA TensorRT-Model-Optimizer 支持完善 | $3,000-$15,000（多 GPU 推理集群 + 稀疏化微调） |
| **大型分布式系统（>70B 模型，高吞吐）** | MoE 架构（DeepSeekMoE / Mixtral） | 参数量-计算量解耦使万亿参数级训练成为可能；大 batch 下专家并行效率高。DeepSpeed-MoE + ZeRO-3 提供完整系统支持 | $50,000-$500,000（千卡级别训练/推理集群） |
| **极端资源受限部署（移动端/边缘端）** | 联合压缩（OBR / P-Q-D 流水线） | W4A4KV4 + 50% 稀疏实现 6.4× 内存压缩和 4.7× 加速，是唯一能在资源受限设备上运行大模型的技术组合 | $100-$1,000（单边缘设备或小型服务器） |
| **预训练加速（从零训练新模型）** | 结构化剪枝 + 重训练（Sheared LLaMA 范式） | 先在稠密小模型上训练，再通过结构化剪枝扩展到大模型，仅需 3%-10% 的预训练成本即可获得同等性能 | 预训练成本的 3%-10%（如训练 7B 模型需 $50K，本方案仅 $2K-$5K） |
| **前沿研究/探索极限稀疏** | 动态稀疏训练 + ADMM 约束优化（Elsa / DST） | 追求 90%+ 的极致稀疏度时需要动态结构和约束优化的结合。虽然工程复杂，但精度-效率前沿的优势显著 | $10,000-$100,000（大量实验和消融研究） |

---

## 第四部分：精华整合

### 4.1 The One 公式

$$
\text{稀疏训练效率-精度平衡} = \underbrace{\text{结构化稀疏设计}}_{\text{零计算跳过}}
+ \underbrace{\text{重要性引导的误差补偿}}_{\text{最大化每比特/每 FLOP 的信息量}}
- \underbrace{\text{非结构化碎片化}}_{\text{硬件利用率的下降}}
$$

*解释：稀疏训练的核心是在"跳过不需要的计算"和"确保跳过的不重要"之间找到平衡，同时避免非结构化带来的硬件效率损失。最成功的方案（如 OBR、SparseForge）都是在精度保持、结构化设计、低成本实施这三个角力点上同时取得进展。*

### 4.2 一句话解释

> **稀疏训练就像给大模型做"定向手术"——不是去掉整个器官（模型压缩），而是让它在工作时只激活所需的一小部分脑细胞，其余保持休眠，这样既能保持思维能力，又大幅降低了能耗。**

### 4.3 核心架构图

```
                       精度要求
                          ↑
                          │
              ┌───────────┼───────────┐
              │           │           │
        结构化剪枝   半结构化N:M   非结构化剪枝
        (真实加速)   (TensorCore)  (理论加速)
              │           │           │
              └───────────┼───────────┘
                          │
                    联合压缩(剪枝+量化+蒸馏)
                     ─── 最优精度-效率 ───→ 稀疏度增加
```

### 4.4 STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation（背景+痛点）** | 大模型参数量已从数十亿飙升至数万亿，训练和推理成本呈超线性增长。以 GPT-3 175B 为例，单次训练成本超 400 万美元，推理同样昂贵。与此同时，模型内部存在大量冗余——许多权重接近零、大量神经元在特定输入下不激活。如何在不大幅牺牲能力的前提下减少计算和存储开销，成为业界核心挑战。 |
| **Task（核心问题）** | 稀疏训练要解决的关键问题是：**如何在保持模型表达能力（精度）的前提下，通过选择性激活参数子集来最大化计算效率**。核心约束包括：(1) 硬件兼容性——非结构化稀疏在 GPU 上无加速效果；(2) 精度悬崖——超过临界稀疏度后精度急剧下降；(3) 工程代价——部分方案的重训练成本可能抵消稀疏带来的收益。 |
| **Action（主流方案）** | 技术演进经历了四个阶段：(1) **后训练剪枝时代**（SparseGPT/Wanda）——分钟级剪枝，50% 稀疏近乎无损；(2) **结构化革命**（Sheared LLaMA、LLM-Pruner）——按结构单元剪除，获得真实硬件加速；(3) **MoE 架构创新**（Switch Transformer、DeepSeekMoE、Mixtral）——架构级稀疏，将计算量与参数量解耦，实现万亿参数规模；(4) **联合优化新范式**（OBR、SparseForge、Elsa）——剪枝+量化+蒸馏协同，在 90%+ 稀疏度下仍能保持精度。2025-2026 年的核心突破包括：Compression Scaling Laws 为稀疏提供了理论基础、ADMM 约束优化突破 90% 稀疏度门槛、MaskLLM 让稀疏掩码本身可跨任务迁移。 |
| **Result（效果+建议）** | 当前最优成果：OBR 实现 W4A4KV4 + 50% 稀疏联合压缩（6.4× 内存压缩 + 4.7× 加速），SparseForge 仅用 5B tokens 恢复 2:4 稀疏精度超越稠密模型。但存在明显局限：高稀疏度下的推理稳定性、MoE 在大 batch 下的通信瓶颈、以及极端场景下的鲁棒性退化仍是未完全解决的问题。**实操建议**：小型项目首选 Wanda+量化（成本最低）；中型生产优选 N:M 稀疏（真实 GPU 加速）；大规模部署拥抱 MoE 生态（DeepSpeed-MoE + ZeRO-3）；资源极限场景采用 OBR 联合压缩。**避免**：追求理论稀疏度而忽视硬件实际加速效果。 |

---

### 4.5 理解确认问题

**问题**：假设你有一个 7B 参数的 Llama-2 模型部署在 8×A100 服务器上，当前推理延迟过高。你要在以下三种方案中选择：
- (A) 使用 Wanda 剪枝到 70% 非结构化稀疏
- (B) 使用 MaskLLM 训练 2:4 半结构化稀疏
- (C) 使用 SparseGPT 剪枝到 50% 结构化稀疏并按层非均匀分配

**问**：哪种方案能实际降低延迟？为什么？

**参考答案**：**(B) 和 (C)** 都能带来实际加速。原因：
- (A) ❌ 不可选：70% 非结构化稀疏在 A100 上无法利用 Tensor Core，退化为基础 CUDA 内核计算，没有实际速度提升。而且 Wanda 在高稀疏度（70%）下的精度退化远严重于 SparseGPT。
- (B) ✅ 可选：2:4 半结构化稀疏是 NVIDIA Ampere+ 架构 Tensor Core 原生支持的稀疏模式（2× 加速），MaskLLM 的可学习掩码可在目标领域微调中恢复精度。这是"真实加速"方案。
- (C) ✅ 可选：非均匀层间稀疏分配（OWL 方法）+ 结构化模式，可在 50% 整体稀疏度下获得接近 2× 的加速（配合 DeepSparse 等引擎），且由于是逐层自适应，精度保持更优。
- **最优选**：在 B200 或 H100 上，推荐 **B + INT8 量化组合**，可实现约 3-4× 端到端加速，且精度损失小于 1%。

---

> **报告结束** | 调研日期：2026-05-22 | 调研框架版本：v2.0
