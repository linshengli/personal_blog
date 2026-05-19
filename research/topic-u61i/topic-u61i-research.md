# 大模型训练损失函数设计与优化方法 —— 深度调研报告

> **调研日期**：2026-05-19
> **所属领域**：大模型训练
> **报告类型**：技术深度调研（四维度全分析）

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

大模型训练损失函数是衡量模型预测分布与真实分布之间差异的数学函数，其核心作用是提供可微分的优化目标，指导反向传播中的梯度更新。在大语言模型（LLM）的预训练中，最主流的损失函数是**因果语言建模的交叉熵损失（Causal LM Cross-Entropy Loss）**：给定前文 token 序列 $x_{<t}$，模型需要最大化正确下一个 token $x_t$ 的对数概率。

### 常见误解

1. **误解：交叉熵损失是唯一适用于大模型的损失函数**
   事实：虽然交叉熵是预训练阶段的标准选择，但微调和对齐阶段广泛使用 DPO Loss、PPO-Clip Loss、GRPO Loss、Focal Loss 等多种替代方案。2025-2026 年已有大量研究表明 Harmonic Loss、Number Token Loss 等替代方案在特定场景下优于交叉熵。

2. **误解：损失函数值越低，模型性能越好**
   事实：训练损失下降并不总是对应下游任务性能提升。过低的训练损失可能意味着过拟合（记忆而非泛化），且 Neural Collapse 现象表明不同损失函数在充分收敛后可能产生等价的特征表示。

3. **误解：损失函数只参与反向传播，不影响推理阶段**
   事实：损失函数的设计（如是否使用标签平滑、是否对特定 token 加权）会直接影响模型学到的概率分布特性，进而影响推理时的采样质量和多样性。

### 边界辨析

| 相邻概念 | 核心区别 | 示例 |
|---------|---------|------|
| **损失函数 vs. 评估指标** | 损失函数须可微，用于训练优化；评估指标用于衡量最终性能，未必可微 | 交叉熵（可微）vs. BLEU/ROUGE（不可微） |
| **损失函数 vs. 奖励模型** | 损失函数直接作用于模型参数更新；奖励模型是单独训练的打分函数，为 RL 提供奖励信号 | CE Loss vs. Reward Model |
| **损失函数 vs. 正则化项** | 损失函数衡量预测误差；正则化项约束模型复杂度以防范过拟合 | Cross-Entropy + KL 惩罚项中的 KL 部分是正则化 |

## 1.2 核心架构

```
┌─────────────────────────────────────────────────────────────┐
│              大模型训练损失函数系统架构                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  输入序列 → [Token Embedding] → [Transformer Layers]        │
│      ↓                                                      │
│  [LM Head / Unembedding]  →  logits (V维向量)               │
│      ↓                                                      │
│  ┌─────────────────────────────────────────────────┐        │
│  │              损失计算层                            │        │
│  │                                                   │        │
│  │  ┌──────┐    ┌──────────┐    ┌─────────────┐     │        │
│  │  │Softmax│ →  │负对数似然│ →  │Token加权聚合 │     │        │
│  │  └──────┘    └──────────┘    └─────────────┘     │        │
│  │      ↑                              ↑            │        │
│  │  ？替代选择:                    可选组件:         │        │
│  │  Harmonic Loss                   Focal γ权重     │        │
│  │  Number Token Loss               MiLe熵权重      │        │
│  │  f-Divergence Loss              标签平滑         │        │
│  └─────────────────────────────────────────────────┘        │
│      ↓                                                      │
│  [反向传播] → [优化器(AdamW)] → 参数更新                    │
│                                                             │
│  辅助损失（可选）：                                          │
│  ┌──────────────────────────────────────────────────┐       │
│  │ KL散度惩罚(对齐阶段) │ 辅助Z-loss(稳定性) │ 路由损失  │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

**各组件说明：**

| 组件 | 职责 |
|------|------|
| **LM Head** | 将最后一层隐藏状态映射到词表大小的 logits 向量，维度为 `[batch_size, seq_len, vocab_size]` |
| **Softmax** | 将 logits 转换为概率分布 $p(\cdot \mid x_{<t})$，使所有 token 概率之和为 1 |
| **负对数似然** | 计算正确 token 对应概率的负对数 $-\log p(x_t \mid x_{<t})$ |
| **Token加权聚合** | 对所有 token 的损失进行加权平均（可均匀加权，或按难度/熵值加权） |
| **可选替代方案** | 用 Harmonic Loss/Number Token Loss 等替代 Softmax+交叉熵的组合 |
| **辅助损失** | 对齐阶段附加 KL 散度惩罚，或稳定性辅助损失（如 Z-loss） |

## 1.3 数学形式化

### 公式 1：标准交叉熵损失（Causal LM）

$$
\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) = -\frac{1}{T}\sum_{t=1}^{T} \log \frac{\exp(h_t^\top e_{x_t})}{\sum_{v \in \mathcal{V}} \exp(h_t^\top e_v)}
$$

> 对所有位置 $t$ 的正确 token 的对数概率取平均。$h_t$ 是最后一层隐藏状态，$e_v$ 是 token $v$ 的嵌入向量，$\mathcal{V}$ 是词表。这是 LLM 预训练中最核心的优化目标。

### 公式 2：DPO 偏好优化损失

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x,y_w,y_l) \sim D}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]
$$

> 核心思想：在不依赖显式奖励模型的情况下，通过偏好对（chosen/rejected）直接优化策略。$\beta$ 控制对参考策略的偏离程度，$\sigma$ 是 Sigmoid 函数。这是对齐阶段最流行的损失函数之一。

### 公式 3：PPO-Clip 策略梯度损失（带 KL 惩罚）

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) - \beta \cdot \mathbb{KL}[\pi_\theta \parallel \pi_{\text{ref}}]\right]
$$

> 其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ 是重要性采样比率。裁剪机制防止单步更新过大破坏策略稳定性，KL 惩罚项约束策略不偏离参考模型太远。

### 公式 4：GRPO 组相对优势估计

$$
\hat{A}_{i} = \frac{r_i - \mu_G}{\sigma_G}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j,\ \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_G)^2}
$$

> 对同一 prompt 采样 $G$ 个响应，用组内 $G$ 个奖励的均值和标准差对每个响应进行归一化。无需训练值函数网络（critic），大幅降低内存开销。DeepSeek-R1 的核心技术。

### 公式 5：Cut Cross-Entropy 内存优化

$$
\text{CCE Memory} = O(B) \quad \text{vs.} \quad \text{Standard CE Memory} = O(B \times V)
$$

> 标准交叉熵需物化 $B \times V$ 的 logit 矩阵（$B$ 为 token 数，$V$ 为词表大小），CCE 通过自定义 CUDA kernel 在 flash 内存中即时计算 log-sum-exp，将损失计算的内存占用从 $O(B \times V)$ 降至 $O(B)$。对 Gemma 2 2B 而言，从 24GB 降至 1MB。

## 1.4 实现逻辑（Python 伪代码）

```python
class CausalLMLoss:
    """
    因果语言建模损失计算的核心抽象。
    支持标准交叉熵及各种扩展变体。
    """
    def __init__(self, config):
        self.vocab_size = config.vocab_size
        self.label_smoothing = config.get("label_smoothing", 0.0)
        self.loss_type = config.get("loss_type", "cross_entropy")
        # 可选：Focal Loss 参数
        self.focal_gamma = config.get("focal_gamma", 0.0)
        # 可选：MiLe Loss 的熵加权
        self.use_entropy_weight = config.get("use_entropy_weight", False)

    def forward(self, logits, labels):
        """
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]  (pad token 用 -100 掩码)
        """
        # 1. 将 logits 展平为 [B*S, V]，labels 展平为 [B*S]
        flat_logits = logits.view(-1, self.vocab_size)
        flat_labels = labels.view(-1)

        # 2. 计算交叉熵（核心操作）
        if self.loss_type == "cross_entropy":
            loss_per_token = F.cross_entropy(
                flat_logits, flat_labels,
                reduction="none",
                label_smoothing=self.label_smoothing
            )

        elif self.loss_type == "focal":
            # Focal Loss: 降低置信度高的易分样本的权重
            ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
            probs = F.softmax(flat_logits, dim=-1)
            pt = probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1 - pt) ** self.focal_gamma
            loss_per_token = focal_weight * ce_loss

        elif self.loss_type == "mile":
            # MiLe Loss: 基于预测分布的熵动态加权
            ce_loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
            probs = F.softmax(flat_logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            # 高熵 → 难样本 → 更高权重
            weight = 1.0 + entropy / math.log(self.vocab_size)
            loss_per_token = weight * ce_loss

        # 3. 掩码 padding token，只对有效 token 求平均
        mask = (flat_labels != -100).float()
        loss = (loss_per_token * mask).sum() / mask.sum()

        return loss


class DPOLoss:
    """
    Direct Preference Optimization 损失函数。
    在不训练奖励模型的情况下直接优化偏好。
    """
    def __init__(self, beta=0.1):
        self.beta = beta

    def forward(self, policy_chosen_logps, policy_rejected_logps,
                ref_chosen_logps, ref_rejected_logps):
        """
        计算 DPO 损失。
        输入为 chosen/rejected 响应的对数概率（已按序列长度求和）。
        """
        # 对数概率比率差
        log_ratio_chosen = policy_chosen_logps - ref_chosen_logps
        log_ratio_rejected = policy_rejected_logps - ref_rejected_logps

        # DPO 核心公式
        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)
        loss = -F.logsigmoid(logits).mean()

        return loss
```

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **Perplexity** | < 10（7B 模型） | 在验证集上计算 $e^{\mathcal{L}_{\text{CE}}}$ | 衡量模型对序列的预测能力，越低越好 |
| **训练损失收敛值** | 取决于模型规模和数据量 | 训练过程中记录 | 可通过 Scaling Law 预测（Chinchilla 最优计算量） |
| **下游任务准确率** | 因任务而异 | MMLU/GSM8K/HumanEval 等标准基准 | 衡量损失函数设计对实际能力的泛化影响 |
| **损失计算显存** | < 2GB（对 7B 模型） | nvidia-smi 峰值显存监控 | CCE 可将此指标从 24GB+ 降至 ~1MB |
| **训练吞吐量** | > 100K tokens/s/GPU | 每秒处理的 token 数 | CCE 等优化可提升 30-50% |
| **对齐偏好胜率** | > 60%（vs. SFT 基线） | 人工评估或 LLM-as-Judge | DPO/PPO 等对齐损失的最终评估指标 |

## 1.6 扩展性与安全性

### 水平扩展

- **数据并行**：损失计算在各 GPU 上独立进行，只需 All-Reduce 同步梯度，是标准水平扩展方式
- **张量并行/序列并行**：LM Head 和损失计算涉及词表维度的矩阵乘法，在张量并行下需切分词表维度并做 All-Gather
- **流水线并行**：损失计算位于最后一层，仅在最后一个流水线阶段执行，不影响前向效率
- **Ring Attention + DPO**：需注意有效 batch size 的变化——启用 Ring Attention 会改变数据并行度，需相应调整微批大小以保持等效训练动态

### 垂直扩展

- **单节点优化上限**：LM Head 的显存占用随词表大小线性增长（词表扩展如 GPT-4 的 ~100K token），CCE 可缓解但计算量不变
- **Flash Attention + CCE**：可组合使用，将注意力计算和损失计算都留在 flash 内存中，最大化单卡效率
- **梯度检查点**：对损失计算回退前向激活以减少显存，但对 CCE 效果有限（CCE 本身已不物化大矩阵）

### 安全考量

- **Reward Hacking**：对齐阶段（PPO/GRPO）中，策略可能学会利用奖励函数的漏洞而非真正改进。解决方案：合理设置 KL 惩罚系数
- **长度劫持（Length Hacking）**：GRPO 的 token 级平均可能导致模型偏好生成长序列。解决方案：使用 Dr. GRPO 中的固定除数而非序列长度
- **记忆化 vs. 泛化**：过低的交叉熵损失可能意味着模型过度记忆训练数据（潜在隐私风险）。Goldfish Loss 通过随机丢弃 token 来缓解此问题
- **数值稳定性**：Softmax + 大 logits 可能导致数值溢出。常用方案：使用 `log_softmax` + `log-sum-exp` 的数值稳定实现；或添加辅助 Z-loss 防止 logits 无界增长

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

以下是截至 2026 年 5 月，与大模型损失函数设计和优化相关的重要开源项目：

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **LLaMA-Factory** | ~71K | 一站式微调框架，支持 CE/DPO/ORPO/SimPO/KTO/PPO 等 6 种损失函数 | Python, PyTorch, DeepSpeed | 2026-05 | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |
| **Unsloth** | ~59K | 加速微调内核，含自定义 `fast_cross_entropy_loss` kernel 实现 2-4x 加速 | Python, CUDA, Triton | 2026-05 | [GitHub](https://github.com/unslothai/unsloth) |
| **OpenRLHF** | ~15K+ | 高性能 RLHF 框架，支持 PPO/DPO/GRPO/REINFORCE++/KTO | Python, Ray, DeepSpeed | 2026-05 | [GitHub](https://github.com/OpenLLMAI/OpenRLHF) |
| **Axolotl** | ~12K+ | YAML 声明式微调框架，集成 Unsloth 损失加速内核 | Python, YAML, DeepSpeed | 2026-05 | [GitHub](https://github.com/axolotl-ai-cloud/axolotl) |
| **TRL (HuggingFace)** | ~10K+ | Transformer 强化学习库，标准 DPO/PPO 损失实现 | Python, transformers | 2026-05 | [GitHub](https://github.com/huggingface/trl) |
| **Cut Cross-Entropy (Apple)** | ~2K+ | 不物化 logit 矩阵的交叉熵实现，损失计算显存从 24GB 降至 1MB | Python, CUDA, Triton | 2025-07 | [GitHub](https://github.com/apple/ml-cross-entropy) |
| **Liger Kernel** | ~2K+ | 融合训练内核集合，含 Fused Linear Cross Entropy | Python, Triton | 2026-04 | [GitHub](https://github.com/linkedin/Liger-Kernel) |
| **Number Token Loss** | — | 数字 token 的回归式损失（ICML 2025），提升数学推理能力 | Python, PyTorch | 2025 | [GitHub](https://github.com/tum-ai/number-token-loss) |
| **LOST** | — | 低秩稀疏预训练，结合稀疏损失约束 | Python | 2025 | [GitHub](https://github.com/JiaxiLi1/LOST-Low-rank-and-Sparse-Training-for-Large-Language-Models) |
| **Relational Loss** | — | 尺度不变替代损失，声称 4000x 加速收敛 | Python, PyTorch | 2025 | [GitHub](https://github.com/massimilianoconcas0-del/Relational_Loss_ML) |
| **Attractor Models** | — | 隐式微分的固定点损失，770M 超越 1.3B Transformer | Python | 2026-05 | [GitHub](https://github.com/jacobfa/Attractor) |
| **SRaR** | 11 | 逐步评分奖励用于 LLM 推理（RLVR 框架） | Python | 2025 | [GitHub](https://github.com/akarinmoe/SRaR) |
| **Lite-OPD** | 6 | 基于 KL 散度的在线策略蒸馏框架 | Python | 2025 | [GitHub](https://github.com/yedaotian9/Lite-OPD) |

## 2.2 关键论文（12 篇）

### 高影响力经典论文（约 40%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|-------|------|
| **Attention Is All You Need** | Vaswani et al. / Google | 2017 | NeurIPS | 奠定 Transformer 架构，使用交叉熵损失训练 | 100K+ 引用 | [arXiv](https://arxiv.org/abs/1706.03762) |
| **Language Models are Few-Shot Learners (GPT-3)** | Brown et al. / OpenAI | 2020 | NeurIPS | 展示大规模因果 LM + 交叉熵损失的 Scaling Law | 40K+ 引用 | [arXiv](https://arxiv.org/abs/2005.14165) |
| **Training language models to follow instructions (InstructGPT)** | Ouyang et al. / OpenAI | 2022 | NeurIPS | PPO + KL 惩罚的开创性 RLHF 对齐框架 | 12K+ 引用 | [arXiv](https://arxiv.org/abs/2203.02155) |
| **Direct Preference Optimization (DPO)** | Rafailov et al. / Stanford | 2023 | NeurIPS | 无需奖励模型的偏好优化损失函数 | 4K+ 引用 | [arXiv](https://arxiv.org/abs/2305.18290) |
| **Scaling Laws for Neural Language Models** | Kaplan et al. / OpenAI | 2020 | — | 损失值与模型/数据规模的幂律关系 | 8K+ 引用 | [arXiv](https://arxiv.org/abs/2001.08361) |

### 最新 SOTA 论文（约 60%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|-------|------|
| **Cut Your Losses in Large-Vocabulary Language Models** | Wijmans et al. / Apple | 2025 | ICLR Oral | CCE：不物化 logit 矩阵的交叉熵实现，显存从 24GB 降至 1MB | Oral 论文 | [arXiv](https://arxiv.org/abs/2411.09009) |
| **Regress, Don't Guess (Number Token Loss)** | Zausinger et al. / MCML | 2025 | ICML | 数字 token 的回归式损失，提升数学推理 | ICML 录用 | [arXiv](https://arxiv.org/abs/2411.02083) |
| **REINFORCE++: Simple and Efficient LLM Alignment** | Hu / 独立 | 2025 | — | 无 critic 的稳定对齐框架，优于 GRPO 稳定性 | HuggingFace 热点 | [arXiv](https://arxiv.org/abs/2501.03262) |
| **Harmonic Loss Trains Interpretable AI Models** | Baek et al. / MIT | 2025 | NeurIPS | 替代 Softmax 的 HarMax + 距离度量损失 | NeurIPS 2025 | [arXiv](https://arxiv.org/abs/2502.01628) |
| **MiLe Loss: Entropy-Weighted Loss for LLMs** | Su et al. / CAS | 2024 | NAACL Findings | 基于熵的动态 token 加权损失 | NAACL 2024 | [arXiv](https://arxiv.org/abs/2310.19531) |
| **Scaling with Collapse: Predictable LLM Training** | Cerebras | 2026 | — | 归一化损失曲线坍缩 + Celerity 模型族 | 曲线诊断新方法 | [arXiv](https://arxiv.org/abs/2509.25087) |
| **Power-Law Decay Loss for LLM Finetuning** | — | 2025 | — | 基于 token 频率幂律衰减的微调损失 | 聚焦信息稀疏性 | [arXiv](https://arxiv.org/abs/2505.16900) |

## 2.3 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| 为什么大模型损失函数采用交叉熵，而不是 MSE？ | 知乎专栏 | 中文 | 深度分析 | 从梯度角度分析交叉熵在大模型中的不可替代性 | 2025 | [链接](https://zhuanlan.zhihu.com/p/1975608298452572125) |
| Loss 函数综述（回归/分类/度量学习/生成模型） | CSDN 博客 | 中文 | 综述教程 | 涵盖从经典到 LLM 时代的损失函数演变 | 2025 | [链接](https://blog.csdn.net/u013250861/article/details/153472105) |
| 神经网络损失函数全解析：交叉熵/Focal Loss/对比学习 | CSDN 博客 | 中文 | 实战教程 | 公式推导 + PyTorch 代码实战 | 2025 | [链接](https://blog.csdn.net/2501_91798322/article/details/151719673) |
| LLM 强化对齐的四大天王：REINFORCE, PPO, DPO, GRPO | CSDN 博客 | 中文 | 深度对比 | 四种 RL 对齐损失机制的原理与实现 | 2025 | [链接](https://blog.csdn.net/kakaZhui/article/details/148201582) |
| LLM笔记（十三）损失函数及优化器调研报告 | CSDN 博客 | 中文 | 调研报告 | 系统性梳理 LLM 损失函数与优化器选型 | 2025 | [链接](https://blog.csdn.net/weixin_51147313/article/details/148133572) |
| OpenRLHF 项目中 DPO 损失函数实现细节解析 | GitCode 博客 | 中文 | 源码分析 | DPO 损失在分布式训练中的 prompt 掩码实现细节 | 2025 | [链接](https://blog.gitcode.com/eeb5d2961d722739fa40c86dac173fcc.html) |
| Apple's Memory-Saving AI Breakthrough (CCE) | VentureBeat | 英文 | 新闻报道 | Apple CCE 如何在保持性能的同时降低 99.99% 显存 | 2025-02 | [链接](https://venturebeat.com/ai/apples-memory-saving-ai-breakthrough-could-save-enterprises-millions) |
| Focal Loss vs 交叉熵损失：学习率和训练轮数比较 | AwesomeML | 中文 | 实验对比 | 两种损失在不同训练阶段的行为差异及调参建议 | 2025 | [链接](https://awesomeml.com/focal-loss-learning-rate-and-epoch/) |
| A Comprehensive Survey of Loss Functions and Metrics in Deep Learning | Springer AIR | 英文 | 综述论文 | 含 NLP 和 RAG 专用损失函数的全面调查 | 2025 | [链接](https://link.springer.com/article/10.1007/s10462-025-11198-7) |
| Reinforcement Fine-Tuning LLMs With GRPO | DeepLearning.AI | 英文 | 课程笔记 | GRPO 损失计算的逐步推导与实践 | 2025 | [链接](https://learn.deeplearning.ai/courses/reinforcement-fine-tuning-llms-grpo/) |

## 2.4 技术演进时间线

```
2017 ── Transformer (Vaswani et al.)
       └── 确立交叉熵 + Softmax 为 LM 训练的标准范式

2020 ── GPT-3 + Scaling Laws
       └── 揭示 loss 与模型/数据规模的幂律关系，推动大模型竞赛

2022 ── InstructGPT (RLHF)
       └── PPO + KL 惩罚成为对齐阶段主流范式

2023 ── DPO (Direct Preference Optimization)
       └── 颠覆式创新：无需奖励模型，直接优化偏好

2024 ── MiLe Loss · DeepSeek GRPO · Neural Collapse 视角
       ├── 动态熵加权损失引入 token 级自适应
       ├── GRPO 取消 critic 网络，大幅降低对齐训练成本
       └── 理论证明：充分训练后各类损失函数等价

2025 ── CCE · NTL · Harmonic Loss · REINFORCE++ · Liger Kernel
       ├── Apple CCE：将损失显存从 24GB 降至 1MB
       ├── Number Token Loss：数字 token 回归式损失提升数学推理
       ├── Harmonic Loss：替代 Softmax，训练可解释模型
       ├── REINFORCE++：无 critic 稳定对齐，超越 GRPO
       └── Liger Kernel：融合线性交叉熵层加速训练

2026 ── ff-TB Loss Family · G-Loss · Loss Curve Collapse
       ├── f-divergence 损失函数家族统一框架
       ├── G-Loss：图引导的全局语义结构损失
       └── Loss 曲线坍缩作为训练质量诊断工具

当前状态：损失函数设计从"单一交叉熵统治"进入"多元化定制时代"，不同训练阶段
预训练/微调/对齐/数学推理使用专门优化的损失函数。内存效率和计算效率的优化
（CCE/Liger Kernel）正在将损失计算成本推向可忽略的水平。
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
          ┌── 2017-2020: 交叉熵 时代 ──┐
          │  Transformer 确立交叉熵标准   │
          │  Scaling Laws 验证损失幂律    │
          └──────────┬──────────────────┘
                     │
          ┌── 2022-2023: RLHF/对齐 突破 ──┐
          │  PPO-KL (InstructGPT)         │
          │  DPO (Stanford)               │
          └──────────┬──────────────────┘
                     │
    ┌── 2024-2025: 百花齐放 时代 ──────────┐
    │  GRPO (DeepSeek) → 无 critic 对齐    │
    │  MiLe Loss → 动态熵加权 token        │
    │  NTL → 数字 token 回归损失           │
    │  Harmonic Loss → 替代 Softmax       │
    │  CCE → 损失显存革命性降低             │
    │  REINFORCE++ → 无 critic + 更好的 KL │
    └──────────┬──────────────────────────┘
               │
    ┌── 2026+: 统一理论 阶段 ──────────────┐
    │  ff-TB f-divergence 统一框架         │
    │  Loss Curve Collapse 诊断标准        │
    │  AI 自动化损失函数设计                │
    └────────────────────────────────────┘
```

## 3.2 7 种方案横向对比

### 方案一：标准交叉熵（Causal CE）

| 维度 | 说明 |
|------|------|
| **原理** | Softmax + 负对数似然，最大化正确 token 的预测概率 |
| **优点** | (1) 梯度直接高效：$\delta = a - y$ 不存在梯度消失问题；(2) 数学性质好，与信息论中的交叉熵对应；(3) 实现极其成熟，所有框架原生支持；(4) 与 Scaling Law 完美对接 |
| **缺点** | (1) 无法表达 token 间的数值距离（如预测"6"vs"9"损失相同）；(2) 对所有 token 一视同仁，难易样本权重不区分；(3) 需要物化完整的 $B \times V$ logit 矩阵，显存开销大；(4) 在类别不平衡时偏向高频 token |
| **适用场景** | LLM 预训练、通用 SFT 微调（行业黄金标准） |
| **成本量级** | 显存：$O(B \times V)$；计算：$O(B \times V)$ |

### 方案二：Focal Loss

| 维度 | 说明 |
|------|------|
| **原理** | 在 CE 基础上添加 $(1-p_t)^\gamma$ 衰减因子，降低易分样本权重 |
| **优点** | (1) 有效处理长尾/不平衡数据；(2) 在难样本上的持续学习能力强于 CE；(3) 可与标签平滑结合使用 |
| **缺点** | (1) 引入 $\gamma$ 和 $\alpha$ 两个超参数，调优成本高；(2) 预训练阶段未见明显优势（数据本已均衡）；(3) $\gamma$ 过大可能导致精确度提升但召回率下降 |
| **适用场景** | 长尾分布微调（NER、异常检测）、医学 NLP |
| **成本量级** | 显存与 CE 相同；计算略增（多一次幂运算） |

### 方案三：DPO 偏好优化损失

| 维度 | 说明 |
|------|------|
| **原理** | 通过偏好对的隐式奖励建模直接优化策略，使用 Sigmoid 函数对比 chosen/rejected 概率比 |
| **优点** | (1) 无需训练奖励模型，简化为两阶段流程；(2) 训练稳定，超参数少（仅 $\beta$）；(3) 收敛快于 PPO |
| **缺点** | (1) 依赖高质量偏好数据；(2) 静态数据（off-policy），无法利用在线生成的新数据；(3) 对参考模型的选择敏感；(4) 参考模型与策略差距过大时失效 |
| **适用场景** | 偏好对齐、指令跟随优化（LLaMA-3/Claude 等） |
| **成本量级** | 需同时加载策略模型和参考模型（~2x 显存）；计算量与 CE 相当 |

### 方案四：PPO with KL Penalty

| 维度 | 说明 |
|------|------|
| **原理** | 策略梯度 + 裁剪代理目标 + 价值函数估计 + KL 散度惩罚 |
| **优点** | (1) 最成熟的 RLHF 方法，经过大规模验证；(2) 在线学习，可利用实时生成数据；(3) 裁剪机制防止策略崩塌；(4) GAE 优势估计精度高 |
| **缺点** | (1) 需要同时维护 4 个模型（策略/参考/价值/奖励），显存需求高；(2) 超参数多（$\epsilon,\gamma,\lambda,\beta$），调优复杂；(3) 训练速度最慢；(4) 存在 reward hacking 风险 |
| **适用场景** | 大规模对齐训练（InstructGPT/ChatGPT） |
| **成本量级** | 模型量呈 4x 增长；训练时间比 SFT 长 3-10 倍 |

### 方案五：GRPO 组相对策略优化

| 维度 | 说明 |
|------|------|
| **原理** | 对同一 prompt 采样 $G$ 个响应，用组内奖励的 Z-score 作为优势估计，无需 critic 网络 |
| **优点** | (1) 无需价值函数网络，内存减半；(2) 实现简单，训练效率高；(3) DeepSeek-R1/R2 大规模验证有效；(4) 适合数学推理等有明确评分标准的任务 |
| **缺点** | (1) 组内归一化存在统计偏差（$G$ 有限时）；(2) token 级平均引入长度偏差，鼓励冗长回答；(3) 可能过拟合训练集奖励分布；(4) 缺乏价值函数的泛化能力 |
| **适用场景** | 数学推理、竞赛编程（可自动评分任务） |
| **成本量级** | 无 critic（~3x 模型），组大小 $G$ 正比增加采样量 |

### 方案六：Number Token Loss（NTL）

| 维度 | 说明 |
|------|------|
| **原理** | 在标准 CE 基础上，对数字 token 施加 Lp 范数或 Wasserstein-1 距离的回归损失 |
| **优点** | (1) 让模型感知数字间的数值距离（"5" vs "6" 比 "5" vs "9" 好）；(2) 无推理时开销（仅训练时计算）；(3) 可无缝添加到任何 LLM；(4) PyPI 包即装即用 |
| **缺点** | (1) 仅在包含数字 token 的位置生效，收益集中在数学任务；(2) 在 3B 以上规模的验证还不够充分；(3) 需要识别数字 token 的预处理步骤 |
| **适用场景** | 数学推理（GSM8K/ MATH）、科学计算、金融 NLP |
| **成本量级** | 计算量增约 1%（可忽略）；显存与 CE 持平 |

### 方案七：Harmonic Loss（谐波损失）

| 维度 | 说明 |
|------|------|
| **原理** | 用 HarMax 替代 Softmax，用欧氏距离（或余弦距离）替代点积 logits |
| **优点** | (1) 尺度不变性，防止 logits 无界增长；(2) 有限收敛点（CE 有无限个），训练更稳定；(3) 模型权重对应类中心，可解释性强；(4) 数据效率更高，泛化更快 |
| **缺点** | (1) 生态尚不成熟，主流框架原生不支持；(2) 在超大模型（>7B）上的验证有限；(3) 梯度计算比 CE 更复杂；(4) 与现有 CUDA kernel 优化不兼容 |
| **适用场景** | 小到中等规模模型训练、可解释性优先任务 |
| **成本量级** | 计算量略高于 CE；显存相当；无成熟工程优化 |

## 3.3 技术细节对比

| 维度 | 标准 CE | Focal Loss | DPO Loss | PPO+KL | GRPO | NTL | Harmonic |
|------|---------|-----------|---------|--------|------|-----|----------|
| **性能（下游）** | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| **训练效率** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| **显存效率** | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| **易用性** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| **生态成熟度** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★☆☆☆☆ |
| **社区活跃度** | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ | ★★☆☆☆ |
| **学习曲线** | ★☆☆☆☆（低） | ★★☆☆☆ | ★★☆☆☆ | ★★★★★（高） | ★★★★☆ | ★★☆☆☆ | ★★★★☆ |
| **超参数数量** | 0-2 | 2-3 | 1-2 | 5-8 | 3-5 | 1-2 | 2-3 |

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本（以 8×A100 为例） |
|------|---------|---------|---------------------------|
| **小型项目/原型验证（<1B 参数）** | 标准 CE + DPO | 生态最成熟，实现成本最低，快速验证 | ~$2,000-5,000（云 GPU） |
| **中型生产环境（7B-13B 微调）** | CCE + Liger Kernel + DPO | CCE 大幅降低显存，可用更少 GPU | ~$8,000-15,000 |
| **大型分布式系统（70B+ 训练）** | CCE + GRPO/REINFORCE++ | CCE 节省显存，GRPO 无 critic 节省模型量 | ~$50,000-200,000+ |
| **数学推理优先场景** | CE + NTL（组合） | NTL 以 1% 额外成本带来显著数学性能提升 | 同等硬件，额外成本可忽略 |
| **偏好对齐（通用对话）** | DPO 或 REINFORCE++ | DPO 简单高效；REINFORCE++ 在需要在线学习时更优 | ~$10,000-30,000 |
| **可解释性需求场景** | Harmonic Loss | 权重对应类中心，天然可解释 | ~$5,000-10,000（限于中小规模） |
| **长尾分布分类微调** | Focal Loss（γ=2.0） | 难样本聚焦避免高频类别支配 | 与 CE 同等硬件 |
| **训练 LLM 的损失函数研究** | ff-TB 损失家族 + Loss Curve Collapse 诊断 | 统一框架探索不同 f-divergence + 曲线坍缩诊断训练质量 | ~$10,000-50,000 |

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{LLM 训练损失函数} = \underbrace{\text{交叉熵}}_{\text{预训练黄金标准}} + \underbrace{\text{偏好优化损失}}_{\text{对齐人类意图}} - \underbrace{\text{不必要的显存物化}}_{\text{CCE 将其降至 1MB}}
$$

> 核心洞察：大模型训练损失函数的核心挑战不再是"如何设计更好的数学公式"，而是"如何在不同训练阶段为不同任务选择最合适的损失函数，并工程化地降低其计算成本"。

## 4.2 一句话解释

大模型的损失函数就像老师批改作业的方式——预训练时它检查每个字是否写对（交叉熵），微调时它关注模型是否学会了区分好坏回答（DPO/PPO），而最新技术让这个批改过程几乎不消耗额外内存（CCE），甚至让模型能感知数字之间的数值关系（NTL）。

## 4.3 核心架构图

```
预训练阶段：
    输入 → [Transformer] → [Softmax + CE] → 梯度更新
                                ↓
                        困惑度（Perplexity）

微调阶段（SFT）：
    输入 → [Transformer] → [CE with Response Masking] → 梯度更新
                                ↓
                        只对回答部分计算损失

对齐阶段（RLHF）：
    输入 → [策略模型] → 生成响应
              ↓
    奖励/偏好信号 → [DPO/PPO/GRPO Loss] → 梯度更新
              ↓
    KL 散度惩罚 ← [参考模型]（防止偏离太远）
```

## 4.4 STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation（背景+痛点）** | 大模型训练中，损失函数的选择直接影响模型性能和训练效率。然而传统交叉熵损失存在三个固有问题：无法感知 token 间的数值距离、对易/难样本无差别对待、需要物化巨大的 logit 矩阵（24GB+）。此外，对齐阶段的损失函数（PPO）需要维护 4 个模型，训练成本极高。 |
| **Task（核心问题）** | 如何设计适用于大模型不同训练阶段（预训练/微调/对齐）的损失函数，使得：(1) 模型能更好地理解 token 间的语义和数值关系；(2) 对齐过程中策略稳定且成本可控；(3) 损失计算本身不成为训练的内存瓶颈。 |
| **Action（主流方案）** | 行业从"交叉熵一统天下"演进到了"分阶段定制化"：预训练坚守 CE + Scaling Laws 成熟框架；微调引入 MiLe Loss（熵加权）和 Focal Loss（难样本聚焦）；对齐阶段经历了 PPO → DPO → GRPO → REINFORCE++ 的迭代，逐渐抛弃了昂贵的 critic 网络；数字推理任务加入 NTL 回归损失；工程上 Apple 的 CCE 和 Liger Kernel 将损失计算显存开销降低了 99.99%。 |
| **Result（效果+建议）** | 当前最佳实践是"混合损失策略"：预训练用 CE + CCE 优化效率，微调用 CE + 标签平滑，对齐用 DPO（静态数据）或 REINFORCE++（在线数据），数学任务补充 NTL。2025-2026 年的趋势是 f-divergence 统一框架（ff-TB）和 AI 自动化损失设计正在兴起，值得持续关注。 |

## 4.5 理解确认问题

**问题**：如果预训练阶段将标准交叉熵替换为 Number Token Loss，会对模型产生什么影响？何时应该这么做？

**参考答案**：
- **影响**：NTL 仅在数字 token 位置额外施加 Lp 范数或 Wasserstein 回归损失。其效果是让模型感知数字的数值距离（如 $6$ 远离 $5$ 比 $9$ 更"接近"正确预测），从而在数学推理任务上提升准确率。
- **何时使用**：对于通用 LLM 预训练，NTL 作为 CE 的轻量补充（约 1% 额外计算量）是无害有益的。但对于纯文本模型（如对话模型），数字 token 占比较低，收益有限。最优策略是：通用预训练用 CE，在数学领域的继续预训练或微调阶段加入 NTL。

---

# 附录：参考文献

1. Vaswani et al., "Attention Is All You Need", NeurIPS 2017
2. Brown et al., "Language Models are Few-Shot Learners", NeurIPS 2020
3. Ouyang et al., "Training language models to follow instructions", NeurIPS 2022
4. Rafailov et al., "Direct Preference Optimization", NeurIPS 2023
5. Wijmans et al., "Cut Your Losses in Large-Vocabulary Language Models", ICLR 2025 Oral
6. Zausinger et al., "Regress, Don't Guess (Number Token Loss)", ICML 2025
7. Su et al., "MiLe Loss: Entropy-Weighted Loss for LLMs", NAACL 2024 Findings
8. Hu, "REINFORCE++: A Simple and Efficient Approach for Aligning LLMs", 2025
9. Baek et al., "Harmonic Loss Trains Interpretable AI Models", NeurIPS 2025
10. ff-Trajectory Balance: A Loss Family for Tuning GFlowNets, Generative Models, and LLMs, arXiv:2605.15417, 2026
11. G-Loss: Graph-Guided Fine-Tuning of Language Models, arXiv:2604.25853, 2026
12. Scaling with Collapse: Efficient and Predictable Training of LLM Families, Cerebras, 2026
13. Terven et al., "A Comprehensive Survey of Loss Functions and Metrics in Deep Learning", AIR 2025
14. LLaMA-Factory: hiyouga/LLaMA-Factory, GitHub, ~71K Stars
15. OpenRLHF: OpenLLMAI/OpenRLHF, GitHub, ~15K+ Stars
16. Cut Cross-Entropy: apple/ml-cross-entropy, GitHub, ~2K+ Stars

---

> **报告声明**：本报告基于 2026 年 5 月公开可查的论文、GitHub 项目和博客信息编写。Star 数量为近似值，实际数值可能随社区动态变化。选型建议中的成本估算基于典型云 GPU 租赁价格，实际成本因具体配置而异。
