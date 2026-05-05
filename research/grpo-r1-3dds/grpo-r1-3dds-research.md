# 大模型强化学习 GRPO 与 R1 训练方法 — 深度调研报告

> **调研领域**：大模型强化学习 / Reasoning RL
> **调研日期**：2026-05-05
> **核心关键词**：GRPO、DeepSeek-R1、RLVR、Group Relative Policy Optimization、Reasoning Model

---

## 目录

1. [概念剖析](#一概念剖析)
2. [行业情报](#二行业情报)
3. [方案对比](#三方案对比)
4. [精华整合](#四精华整合)

---

## 一、概念剖析

### 1.1 定义澄清

**通行定义**：GRPO（Group Relative Policy Optimization，群体相对策略优化）是一种无批评器（Critic-free）的强化学习算法，由 DeepSeek 团队在 DeepSeekMath 论文（Shao et al., 2024）中首次提出。其核心思想是对每个 prompt 采样 G 条输出轨迹，利用组内奖励的均值和标准差归一化来估计优势函数，从而替代 PPO 中需要独立训练的价值网络（Critic）。GRPO 是 DeepSeek-R1 / R1-Zero 训练管线的核心算法，也是当前大模型推理能力强化训练的主流方案。

**常见误解**：

1. **GRPO 是纯粹 On-Policy 算法** —— 实际上，ICLR 2026 的论文《Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm》通过数学推导证明，GRPO 具有天然的 Off-Policy 解释，重要性采样和裁剪扮演了关键角色。
2. **R1 的"顿悟时刻"（Aha Moment）是纯 RL 创造的涌现行为** —— 2025 年 Sea AI Lab 的研究发现，DeepSeek-V3-Base 在 RL 训练前已经存在自我反思关键词（"wait"、"verify"），RL 只是放大了已有模式而非创造了新能力。
3. **GRPO 等价于去掉 Critic 的 PPO** —— 两者虽然共享裁剪目标函数，但 GRPO 用组内统计量替代价值函数，训练动态和收敛特性有本质差异，且 GRPO 更适合规则奖励场景。

**边界辨析**：

| 对比对象 | 与 GRPO 的核心区别 |
|---------|------------------|
| **PPO** | 使用独立的 Critic 网络估计值函数；需要 4 模型（Actor + Critic + Reference + Reward）；显存开销大 |
| **DPO** | 完全无需奖励模型和 RL 循环，直接在偏好对数据上优化；不适用于数学/代码等可验证任务 |
| **RLHF (PPO-based)** | 依赖人类偏好训练的奖励模型；GRPO 可用规则奖励替代，降低训练成本 |

### 1.2 核心架构

```
┌──────────────────────────────────────────────────────────┐
│                GRPO 训练系统架构                           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Prompt q ──→ [策略网络 π_θ] ──→ 采样 G 条响应 {o_i}     │
│                   │                    │                  │
│                   │                    ▼                  │
│                   │           [奖励函数 R(o_i)]            │
│                   │                    │                  │
│                   │                    ▼                  │
│                   │           [组内归一化]                  │
│                   │      Â_i = (r_i - μ)/σ               │
│                   │                    │                  │
│                   ▼                    ▼                  │
│           ┌──────────────────────────────────┐            │
│           │  GRPO 损失函数（最大化）：         │            │
│           │  L = L_clip + β·KL(π_θ∥π_ref)    │            │
│           └──────────────────────────────────┘            │
│                    │                                      │
│                    ▼                                      │
│           [策略梯度更新 π_θ]                               │
│                                                          │
│  辅助组件：                                               │
│  · Reference 模型 π_ref — 冻结的基座，约束 KL 散度         │
│  · vLLM — 高效推理采样引擎                                 │
│  · DeepSpeed — 分布式训练框架                             │
└──────────────────────────────────────────────────────────┘
```

**组件职责**：

| 组件 | 功能说明 |
|------|---------|
| **策略网络 π_θ** | 待训练的主模型（如 DeepSeek-V3-Base），负责生成响应 |
| **奖励函数 R** | 规则奖励（答案匹配/格式检查）或模型奖励，为每条采样打分 |
| **组内归一化** | 以 G 条响应的均值和标准差为基准计算相对优势，替代 Critic |
| **GRPO 损失** | 裁剪代理目标 + KL 散度约束，确保训练稳定 |
| **Reference 模型** | 冻结的参考模型，用于计算 KL 惩罚，防止策略偏移过远 |

### 1.3 数学形式化

#### 公式 1：GRPO 完整目标函数

$$
J_{GRPO}(\theta) = \mathbb{E}_{q \sim \mathcal{D},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(o|q)} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\; \text{clip}(r_{i,t}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,t} \right) - \beta D_{KL}(\pi_{\theta} \| \pi_{\text{ref}}) \right]
$$

其中 $r_{i,t}(\theta) = \frac{\pi_{\theta}(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t}|q, o_{i,<t})}$ 为重要性采样比。裁剪机制防止单步更新过大。

#### 公式 2：组内相对优势函数

$$
\hat{A}_i = \frac{r_i - \mu_G}{\sigma_G}, \quad \mu_G = \frac{1}{G}\sum_{j=1}^{G} r_j, \quad \sigma_G = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_G)^2}
$$

核心创新：用组内规范化替代价值函数。当响应优于组平均水平时 $\hat{A}_i > 0$，策略增加其概率；反之则降低。

#### 公式 3：KL 散度估计（无偏 K3 估计器）

$$
D_{KL}^{(t)} = \frac{\pi_{\text{ref}}(o_{i,t}|\cdot)}{\pi_{\theta}(o_{i,t}|\cdot)} - \log\frac{\pi_{\text{ref}}(o_{i,t}|\cdot)}{\pi_{\theta}(o_{i,t}|\cdot)} - 1
$$

采用无偏非负估计量，避免简单对数比估计可能为负的问题，确保 KL 惩罚始终为正。

#### 公式 4：GRPO 简化版——单步更新梯度

当每批只做一次梯度更新时（$\pi_{\theta} = \pi_{\theta_{\text{old}}}$），目标简化为：

$$
\nabla_{\theta} J_{GRPO} \approx \mathbb{E}\left[ \frac{1}{G}\sum_{i=1}^G \frac{\hat{A}_i}{|o_i|}\sum_{t=1}^{|o_i|} \nabla_{\theta} \log \pi_{\theta}(o_{i,t}|\cdot) \right] - \beta \nabla_{\theta} D_{KL}
$$

这等价于：对正优势响应增大概率、对负优势响应减小概率，同时受 KL 约束。

### 1.4 实现逻辑（Python 伪代码）

```python
class GRPOTrainer:
    """GRPO 训练器核心抽象，体现去 Critic 的组内相对优势训练范式"""

    def __init__(self, actor: nn.Module, ref_model: nn.Module,
                 reward_fn: callable, group_size: int = 8,
                 clip_eps: float = 0.2, kl_beta: float = 0.04):
        self.actor = actor           # 待训练的策略网络 π_θ
        self.ref_model = ref_model   # 冻结的参考模型 π_ref，用于 KL 约束
        self.reward_fn = reward_fn   # 规则奖励函数（如答案匹配、格式校验）
        self.G = group_size          # 每 prompt 采样数，典型值 8~64
        self.eps = clip_eps          # PPO 裁剪阈值
        self.beta = kl_beta          # KL 惩罚系数

    def train_step(self, prompts: List[str]) -> dict:
        """单步 GRPO 训练"""
        # 阶段 1：Rollout —— 采样 G 条响应
        all_responses = []
        for prompt in prompts:
            responses = self.actor.generate(
                prompt, num_return_sequences=self.G)
            all_responses.extend(responses)

        # 阶段 2：评分 —— 计算每条响应的奖励
        rewards = [self.reward_fn(p, r)
                   for p, r in zip(prompts, all_responses)]

        # 阶段 3：计算组内优势
        advantages = []
        for i in range(0, len(rewards), self.G):
            group_rewards = rewards[i:i+self.G]
            mu, sigma = np.mean(group_rewards), np.std(group_rewards)
            group_adv = [(r - mu) / (sigma + 1e-8)
                         for r in group_rewards]
            advantages.extend(group_adv)

        # 阶段 4：计算 GRPO 损失
        policy_loss = self._compute_clipped_loss(
            all_responses, advantages)
        kl_loss = self._compute_kl_penalty(all_responses)
        total_loss = policy_loss + self.beta * kl_loss

        # 阶段 5：反向传播
        total_loss.backward()
        self.optimizer.step()
        return {"loss": total_loss.item(), "avg_reward": np.mean(rewards)}

    def _compute_clipped_loss(self, responses, advantages):
        """裁剪代理目标"""
        log_ratios = self.actor.log_probs(responses) \
                     - self.old_actor.log_probs(responses)
        ratios = torch.exp(log_ratios)
        clipped = torch.clamp(ratios, 1-self.eps, 1+self.eps)
        return -torch.mean(torch.min(ratios * advantages,
                                     clipped * advantages))
```

### 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **推理准确率** | AIME 2024 > 70% pass@1 | 标准评测集 | DeepSeek-R1-Zero 从 15.6% 提升至 71.0% |
| **训练显存** | < PPO 的 60% | 模型加载 + 梯度显存统计 | 去掉 Critic 减少约 40% 显存 |
| **训练步数** | DAPO 比 GRPO 少 50% | 收敛到同等性能所需步数 | DAPO 在 Qwen2.5-32B 上仅用一半步数达同等水平 |
| **熵保持** | 训练中 policy 熵不塌缩 | 每 step 监测 token 级熵值 | ProGRPO 通过重加权优势解决熵塌缩 |
| **推理 Token 效率** | 同等准确率下减少 50%+ Token | 总生成 Token 数 / 正确解答数 | AutoThink 减少 52% token 同时提升 6.4% 准确率 |
| **训练成本** | 增量训练 ~$294K | GPU 小时 × 单价 | DeepSeek R1 增量 RL 训练：512×H800×80h |

### 1.6 扩展性与安全性

**水平扩展**：
- GRPO 训练天然适配数据并行和模型并行策略。每个 prompt 的 G 条采样可独立分配至不同 GPU，通过 all-reduce 同步组内统计量
- vLLM 作为推理引擎支持大规模 rollout 采样，结合 DeepSpeed ZeRO 实现分布式梯度同步
- 典型扩展方案：256~2048 GPU，group size 随 GPU 数量线性扩展

**垂直扩展**：
- 单节点优化上限受限于模型大小和 batch size。去掉 Critic 后，单 A100 80GB 可训练的模型规模约为 PPO 的 1.5~2 倍
- KV cache 优化（PagedAttention、FlashAttention）可提升 rollout 吞吐
- 梯度 checkpoint、混合精度训练（bf16/fp8）进一步压榨单卡性能

**安全考量**：
- **奖励劫持（Reward Hacking）**：模型可能利用规则漏洞获取高奖励（如输出格式正确但答案错），需设计防破解奖励函数
- **语言混杂**：R1-Zero 训练中出现中英文混杂问题，R1 通过语言一致性奖励缓解
- **过度思考（Overthinking）**：模型对简单问题也生成冗长推理链，增加推理成本，需引入长度惩罚
- **有害内容**：纯 RL 训练可能放大模型已有偏见，R1 第四阶段通过 10.6 万条安全数据集做对齐 RL

---

## 二、行业情报

### 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **deepseek-ai/DeepSeek-R1** | 90K+ | 官方 R1 模型权重与推理代码 | PyTorch | 2025-04 | [GitHub](https://github.com/deepseek-ai/DeepSeek-R1) |
| **huggingface/open-r1** | 38K+ | 完整开源复现 R1 训练管线（含 GRPO） | Python, TRL | 2026-01 | [GitHub](https://github.com/huggingface/open-r1) |
| **modelscope/ms-swift** | 13.5K | 600+ 模型训练框架（CPT/SFT/DPO/GRPO） | PyTorch | 2026-04 | [GitHub](https://github.com/modelscope/ms-swift) |
| **Tencent-Hunyuan/MixGRPO** | 1.1K | 流式 GRPO 效率优化（混合 ODE-SDE） | Python | 2026-02 | [GitHub](https://github.com/Tencent-Hunyuan/MixGRPO) |
| **Jiayi-Pan/TinyZero** | 3K+ | 最小化 R1-Zero 纯 RL 复现 | PyTorch | 2025-03 | [GitHub](https://github.com/Jiayi-Pan/TinyZero) |
| **hkust-nlp/simpleRL-reason** | 1.5K+ | 小模型 + 少量数据复现 R1 推理 | PyTorch | 2025-04 | [GitHub](https://github.com/hkust-nlp/simpleRL-reason) |
| **Unakar/Logic-RL** | 2K+ | 基于逻辑谜题的 R1-Zero 复现 | PyTorch, vLLM | 2025-03 | [GitHub](https://github.com/Unakar/Logic-RL) |
| **DolbyUUU/Logic-RL-Lite** | 800+ | 轻量版 Logic-RL，含"无顿悟时刻"发现 | PyTorch | 2025-05 | [GitHub](https://github.com/DolbyUUU/Logic-RL-Lite) |
| **om-ai-lab/VLM-R1** | 1K+ | R1 式强化学习用于视觉语言模型 | PyTorch | 2025-05 | [GitHub](https://github.com/om-ai-lab/VLM-R1) |
| **agentica-org/DeepScaleR** | 2K+ | 1.5B 推理模型 DeepScaleR-Preview | PyTorch | 2025-04 | [GitHub](https://github.com/agentica-org/DeepScaleR) |
| **lzhxmu/CPPO** | — | Completion Pruning GRPO 加速 | PyTorch | 2025-05 | NeurIPS 2025 |
| **shengjun-zhang/VisualGRPO** | — | E-GRPO 用于视觉生成 RL | PyTorch | 2026-01 | [GitHub](https://github.com/shengjun-zhang/VisualGRPO) |
| **billhhh/KRPO_LLMs_RL** | — | 卡尔曼滤波增强 GRPO 优势估计 | PyTorch | 2025-05 | [GitHub](https://github.com/billhhh/KRPO_LLMs_RL) |
| **xiwenc1/DRA-GRPO** | — | 多样性感知 GRPO（子模互信息） | PyTorch | 2025-05 | [GitHub](https://github.com/xiwenc1/DRA-GRPO) |
| **walkinglabs/hands-on-modern-rl** | — | PPO→DPO→GRPO→RLVR→Agentic 教材 | Python | 2026-05 | [GitHub](https://github.com/walkinglabs/hands-on-modern-rl) |

### 2.2 关键论文

#### 经典高影响力论文（奠基性工作，约 40%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 链接 |
|------|----------|------|----------|---------|------|
| DeepSeekMath: Pushing the Limits of Mathematical Reasoning | Zhihong Shao et al. / DeepSeek | 2024 | arXiv / NeurIPS | **提出 GRPO** 算法，去 Critic 的组内相对优势策略优化 | [arXiv:2402.03300](https://arxiv.org/pdf/2402.03300) |
| DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL | DeepSeek AI | 2025 | Nature 封面 (2025.09) | **四阶段训练管线**（冷启动 SFT→推理 RL→拒绝采样→对齐 RL），首次验证纯 RL 可训练推理模型 | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| DAPO: An Open-Source LLM RL System at Scale | 字节跳动 & 清华 AIR | 2025 | arXiv | **非对称 Clip + 动态采样**，减半训练步数 | [arXiv:2503.14476](https://arxiv.org/pdf/2503.14476) |
| GSPO: Group Sequence Policy Optimization | 通义千问 | 2025 | arXiv | **序列级 Importance Ratio**，从根本上解决 MoE 训练不稳定 | [arXiv:2507.18071](https://arxiv.org/pdf/2507.18071) |

#### 最新 SOTA 论文（前沿进展，约 60%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 链接 |
|------|----------|------|----------|---------|------|
| Understanding R1-Zero-Like Training: A Critical Perspective | Sea AI Lab & NUS | 2025 | arXiv | 揭示 GRPO 的长度偏差和难度偏差，提出 Dr. GRPO | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| Reinforcement Learning with Verifiable Rewards: Dynamics & Success Amplification | — | 2025 | arXiv | 证明 GRPO + 可验证奖励保证成功概率放大 | [arXiv:2503.06639](https://arxiv.org/abs/2503.06639) |
| Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm | — | 2026 | **ICLR 2026** | 证明 GRPO 本质是 Off-Policy，统一 Online PPD 和 Asymmetric REINFORCE | [ICLR 2026](https://iclr.cc/virtual/2026/poster/10011320) |
| RE-GRPO: Reflective Enhanced GRPO | — | 2026 | Neurocomputing | 硬案例池 + LLM 引导反射，训练步数减少 12% | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225032151) |
| ProGRPO: Pro-Group Relative Policy Optimization | — | 2026 | arXiv | 用提示困惑度和答案置信度重加权优势，解决熵塌缩 | [HuggingFace](https://huggingface.co/papers/2602.05281) |
| RM-R1: Reward Modeling as Reasoning | — | 2025 | **ICLR 2026** | 将奖励建模转为推理任务，Chain-of-Rubrics 机制 | [arXiv:2505.02387](https://arxiv.org/abs/2505.02387) |
| AutoThink: Shaping Adaptive Reasoning via Multi-Stage RL | — | 2025 | **NeurIPS** | 三阶段 RL 自适应控制思考深度，减少 52% Token | NeurIPS 2025 |
| Spectral Policy Optimization | — | 2025 | **ICML 2025** | AI 反馈注入响应多样性，解决全错组训练失败 | ICML 2025 |
| VeriFree: Reinforcing General Reasoning without Verifiers | — | 2025 | arXiv | 无验证器的通用推理 RL，拓展到非数学领域 | [arXiv:2505.21493](https://arxiv.org/abs/2505.21493) |
| Parallel-R1: Parallel Thinking via RL | — | 2026 | **ICLR 2026** | 首个并行推理 RL 框架，提升 8.4% 准确率 | [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006733) |
| Personalized GRPO (P-GRPO) | Apple ML Research | 2026 | — | 解耦优势估计与批次统计，适配异构用户偏好 | [Apple ML Research](https://machinelearning.apple.com/research/personalized-group) |
| Noise-corrected GRPO | — | 2025 | arXiv | 贝努力噪声建模，无偏梯度估计，数学 +6.7pp | [arXiv:2510.18924](https://arxiv.org/abs/2510.18924) |

### 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| Advanced Understanding of GRPO | Hugging Face LLM Course | 英文 | 深度教程 | GRPO 数学推导 + 代码实现的完整教学 | 2025 | [HF Course](https://huggingface.co/learn/llm-course/en/chapter12/3b) |
| DeepSeek-R1 技术全景解析（附多阶段训练流程图） | 腾讯云开发者 | 中文 | 架构解析 | R1 四阶段训练管线逐层拆解与核心误区澄清 | 2025 | [腾讯云](https://cloud.tencent.cn/developer/article/2500270) |
| 大模型强化学习全解：从 PPO、DPO 到 DeepSeek 的 GRPO | 阿里云开发者 | 中文 | 对比教程 | PPO/DPO/GRPO 三算法系统对比 | 2025 | [阿里云](https://developer.aliyun.com/article/1709956) |
| 强化学习系列（十二）——GRPO, DAPO, DUPO, GSPO | 腾讯云开发者 | 中文 | 系列文章 | GRPO 演进脉络和各变种的核心创新点 | 2025 | [腾讯云](https://cloud.tencent.cn/developer/article/2592688) |
| GRPO vs Other RL Algorithms: A Guide | HPC-AI Tech Blog | 英文 | 深度对比 | GRPO vs PPO vs DPO 完整对比实验数据 | 2025 | [HPC-AI](https://www.hpc-ai.com/blog/GRPO_vs_other_RL_guide) |
| Mini-R1: Reproduce DeepSeek R1 "Aha Moment" RL Tutorial | Hugging Face Blog | 英文 | 实践教程 | 用 countdown 游戏复现 R1 的 RL 训练全流程 | 2025 | [HF Blog](https://huggingface.co/blog/open-r1/mini-r1-contdown-game) |
| 深度解析：DeepSeek R1-Zero 训练范式与 GRPO 极简优化策略 | 百度智能云 | 中文 | 深度分析 | R1-Zero 纯 RL 训练 + Dr. GRPO 改进方案 | 2025 | [百度云](https://cloud.baidu.com/article/3613515) |
| Post-Training in 2026: GRPO, DAPO, RLVR & Beyond | LLM Stats | 英文 | 综述型 | 2026 年后训练算法全景和趋势分析 | 2026 | [LLM Stats](https://llm-stats.com/blog/research/post-training-techniques-2026) |
| 后训练算法的涌现：从 GRPO 到群智涌现 | 内参 AI | 中文 | 深度分析 | GRPO 商业应用和哲学边界的深度探讨 | 2025 | [内参AI](https://www.neican.ai/insights/grpo-20250901131004921-2/) |
| 解读 GRPO：群体相对策略优化的高效路径 | 百度开发者 | 中文 | 概念解读 | GRPO 数学形式化和实现细节的通俗解读 | 2025 | [百度](https://developer.baidu.com/article/detail.html?id=3806546) |

### 2.4 技术演进时间线

```
2024.02 ── DeepSeek 发布 DeepSeekMath 论文，首次提出 GRPO 算法
             │ 影响：开辟无 Critic 的 LLM 强化学习新范式，显存降低 40%+
             │
2025.01 ── DeepSeek-R1 发布（22页预印本），纯 RL 训练达到 o1 水平
             │ 影响："顿悟时刻"引发全球关注，GRPO 成为业界焦点
             │
2025.01 ── Hugging Face 启动 Open-R1 开源复现项目
             │ 影响：推动 GRPO 训练的民主化和可复现性
             │
2025.03 ── 字节跳动发布 DAPO，非对称 Clip + 动态采样
             │ 影响：训练效率翻倍，GRPO 变种开始涌现
             │
2025.03 ── Sea AI Lab 发表《Understanding R1-Zero-Like Training》
             │ 影响：揭示 GRPO 长度/难度偏差，提出 Dr. GRPO 修正方案
             │
2025.07 ── 通义千问发布 GSPO，序列级 Importance Ratio
             │ 影响：从根本上解决 MoE 模型训练不稳定问题
             │
2025.09 ── DeepSeek-R1 论文登上 Nature 封面
             │ 影响：首个通过顶级期刊同行评审的主流大模型
             │
2026.01 ── DeepSeek 更新 R1 论文至 86 页，公开四阶段完整管线
             │ 影响：训练方法完全透明化，附三中间检查点细节
             │
2026.05 ── 当前状态：
            GRPO 成为推理 RL 事实标准，沿 Token 级（DAPO）和序列级（GSPO）
            两条路线演进。2026 年基调：从"能用"到"稳定高效"的工程成熟阶段
```

---

## 三、方案对比

### 3.1 历史发展时间线

```
2017 ── PPO（OpenAI）── 提出裁剪代理目标 + Critic 网络，RLHF 训练基石
2023 ── DPO（Stanford）── 绕过 RL，直接在偏好对上优化，无需奖励模型
2024.02 ── GRPO（DeepSeek）── 去掉 Critic，组内相对优势，显存减半
2025.03 ── DAPO（字节跳动）── 非对称 Clip + 动态采样 + Token 级 Loss + 超长惩罚
2025.07 ── DUPO（通义千问）── 重复采样 + 2~3x 训练加速，优化 Web Agent 场景
2025.07 ── GSPO（通义千问）── 序列级 Importance Ratio，MoE 稳定性突破
2025.08 ── MEML-GRPO（多专家互学习）── 提升知识多样性，跨模型迁移
2025.10 ── Noise-corrected GRPO ── 针对噪声奖励场景的无偏梯度估计
2026.01 ── E-GRPO（清华）── 熵感知 GRPO，扩展至视觉生成 RL
2026.05 ── 当前状态：去 Critic 已成共识，GSPO 被视为"范式级创新"，有望成为新一代标准
```

### 3.2 五种方案横向对比

| 方案 | 原理 | 优点（3+） | 缺点（3+） | 适用场景 | 成本量级 |
|------|------|-----------|-----------|---------|---------|
| **PPO** | Critic 网络估计值函数 + 裁剪代理目标 | ① 理论完善、验证充分 ② 逐 Token 细粒度优势 ③ 通用 RL 任务适配 | ① 需 4 模型，显存是 GRPO 的 2x ② 实现复杂、调参困难 ③ 奖励模型训练数据成本高 | 通用 RLHF、安全对齐 | 高（10 万美元级 / 单次训练） |
| **DPO** | 跳过 RL 循环，直接在偏好对上优化 | ① 实现最简单 ② 训练稳定 ③ 无需奖励模型和 RL 管线 | ① 需高质量偏好对 ② 无法利用可验证信号 ③ 不适合推理类任务 | 偏好对齐、写作风格调整 | 低（千美元级） |
| **GRPO** | 组内均值/标准差归一化取代 Critic | ① 显存比 PPO 低 40%+ ② 适合规则奖励场景 ③ 数学简洁易实现 | ① 训练不稳定（小批量易崩） ② 熵塌缩问题 ③ 长度偏差（长错误响应惩罚不足） | 推理任务（数学/代码）、R1 风格训练 | 中（29 万美元，DeepSeek R1 增量） |
| **DAPO** | GRPO + 非对称 Clip + 动态采样 | ① 训练步数比 GRPO 少 50% ② 解耦积极探索 ③ Token 级 Loss 去偏 ④ 超长惩罚 | ① 实现复杂度增加 ② 动态采样引入额外超参 ③ 采样效率仍可提升 | 长思维链推理、高精度数学推理 | 中（比 GRPO 略低） |
| **GSPO** | 序列级 Importance Ratio 替代 Token 级 | ① MoE 训练稳定（GRPO 重大突破）② 梯度方差最低 ③ 长序列场景优势明显 ④ 序列级 clip 效果更好 | ① 动态采样和超长惩罚未集成 ② 生态仍在发展中 ③ 序列级 ratio 有轻微信息损失 | MoE 大模型、长序列推理、稀疏奖励 | 中（同 GRPO 量级） |

### 3.3 技术细节对比

| 维度 | PPO | DPO | GRPO | DAPO | GSPO |
|------|-----|-----|------|------|------|
| **性能（AIME 2024 pass@1）** | — | — | ~71%（R1） | ~50%（Qwen2.5-32B，训练步数减半） | 待更多评测 |
| **显存需求** | 最高（~9.8 GB / 小模型） | 较低（~6.8 GB） | 低（~6.2 GB） | 同 GRPO | 同 GRPO |
| **训练稳定性** | 中 | **极高** | 较差（小批量） | 较好 | **最好** |
| **实现复杂度** | 高（需 4 模型） | **低**（2 模型） | 中（3 模型） | 中高 | 中 |
| **生态成熟度** | **最成熟** | 成熟 | 成熟（HF TRL 支持） | 发展中 | 早期 |
| **社区活跃度** | 极高 | 极高 | 极高（2025 年焦点） | 高 | 中 |
| **学习曲线** | 陡峭 | 平缓 | 中等 | 中等偏陡 | 中等 |
| **模型需求** | Actor + Critic + Ref + Reward | Actor + Ref | Actor + Ref + Reward | 同 GRPO | 同 GRPO |
| **数据需求** | 偏好对 + 奖励训练数据 | 偏好对 | 规则奖励或奖励模型 | 规则奖励 | 规则奖励 |
| **可验证任务优势** | 不显著 | 不适用 | **最适合** | 最适合 | 最适合 |

### 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本（含算力） |
|------|---------|---------|-------------------|
| **小型项目 / 原型验证**（1-7B 模型） | **DPO**（偏好数据充足）或 **Open-R1 + GRPO**（推理场景） | DPO 实现简单、训练快、成本最低；GRPO 适合需要推理能力的场景 | **$100~$1,000**（单卡 A100/几天训练） |
| **中型生产环境**（7-32B 模型，推理核心场景） | **GRPO + DAPO 风格改进** | GRPO 生态成熟（HF TRL 支持），DAPO 动态采样可加可不加。显存比 PPO 节省 40%，性价比最优 | **$5,000~$50,000**（4~32 卡 H100，数天~数周） |
| **大型分布式系统**（70B+ MoE，高稳定性要求） | **GSPO** | 序列级 Importance Ratio 解决 MoE 训练不稳定问题，是当前从原理层面最先进的方案 | **$50,000~$300,000**（256~2048 卡 H800，1~4 周） |
| **安全对齐 + 推理双目标** | **R1 四阶段方案**（SFT→GRPO RL→Rejection Sampling→对齐 RL） | DeepSeek 已验证的完整管线，兼顾推理性能和安全性 | **$100,000~$500,000**（DeepSeek R1 增量：29.4 万美元） |
| **非可验证任务（写作、客服）** | **DPO** 或 **PPO + 奖励模型** | 没有明确答案可验证时，偏好对或奖励模型更合适 | **$1,000~$10,000** |

---

## 四、精华整合

### 4.1 The One 公式

$$
\text{GRPO} = \underbrace{\text{组内相对优势}}_{\text{替代 Critic，降低显存}} + \underbrace{\text{裁剪代理目标}}_{\text{稳定策略更新}} - \underbrace{\text{长度/难度偏差}}_{\text{需工程修正}}
$$

### 4.2 一句话解释

**GRPO 是一种让大模型通过"内部考试竞争"来提升推理能力的方法：对一个题目生成多份答卷，用相互比较（而不是外部裁判）算出每份答卷的相对优劣，然后让模型多学习优秀答卷的思路、少学差答卷的错误模式。**

### 4.3 核心架构图

```
Prompt q
   │
   ▼
┌──────────────────┐
│  Policy π_θ      │── 采样 G 条响应 {o₁, o₂, ..., o_G}
│  (主模型)         │
└──────────────────┘
   │
   ▼
┌──────────────────┐
│  Reward R(o_i)   │── 规则评分（答案对错/格式）
└──────────────────┘
   │
   ▼
┌───────────────────────────┐
│  组内归一化 Âᵢ = (rᵢ-μ)/σ │── 相对优势（好于平均 → 奖，差于平均 → 罚）
└───────────────────────────┘
   │
   ▼
┌───────────────────────────────────────┐
│  GRPO Loss                            │
│  = clipped_surrogate + β·KL(π∥π_ref)  │── 更新策略
└───────────────────────────────────────┘
   │
   ▼
输出：推理能力增强的策略网络 π_θ_new
```

### 4.4 STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation（背景+痛点）** | 大语言模型虽然在对话和知识问答上表现优异，但在需要多步推理的数学、编程、逻辑任务上能力有限。传统的 PPO-based RLHF 虽然有效，但需要维护 4 个独立模型（Actor、Critic、Reward、Reference），显存开销巨大（~2x 模型参数量），训练成本极高，中小团队难以负担。同时，DPO 等离线方法无法利用可验证推理信号（如代码运行结果、数学答案对错）。 |
| **Task（核心问题）** | 如何设计一种可扩展、低成本的强化学习算法，让大模型通过可验证的奖励信号自主进化推理能力？核心约束：(1) 显存效率——不能依赖额外的 Critic 网络；(2) 训练稳定性——防止策略崩塌；(3) 探索效率——在稀疏奖励下有效探索解空间。 |
| **Action（主流方案）** | DeepSeek 提出 GRPO：对每个 prompt 采样 G 条输出，用组内均值和标准差做归一化来估计优势函数，彻底去除 Critic 网络，显存降低 40%+。DeepSeek-R1 进一步构建了四阶段训练管线（冷启动 SFT → 推理 GRPO-RL → 拒绝采样再微调 → 对齐 RL），在 AIME 2024 上实现 71.0% 准确率。随后，DAPO（字节跳动）引入非对称 Clip + 动态采样将训练步数减半；GSPO（通义千问）提出序列级 Importance Ratio 从根本上解决 MoE 训练不稳定问题。学术研究同时揭示了 GRPO 的长度偏差和难度偏差，并提出了 Dr. GRPO 等修正方案。 |
| **Result（效果+建议）** | GRPO 系列方法已成为大模型推理训练的事实标准。DeepSeek-R1 增量训练仅花费 29.4 万美元，达到与 OpenAI o1 相当的水平。当前建议：(1) 中小团队使用 Open-R1 + 小模型（7B 级）结合 GRPO 即可获得显著推理提升；(2) 生产环境推荐 DAPO 或 GSPO 以获得更好稳定性；(3) 需警惕 GRPO 的长度膨胀问题，建议引入长度奖励或多阶段自适应机制。未来方向：GSPO 的序列级优化范式有望取代 Token 级 GRPO，成为新一代推理 RL 标准。 |

### 4.5 理解确认问题

**Q**：GRPO 去掉 Critic 网络后，用什么机制来估计优势函数？这个机制在什么情况下会失效？

**A**：GRPO 用"组内相对比较"替代 Critic 网络。具体来说：对每个 prompt 采样 G 条响应，计算每条响应的奖励 r_i，然后以组内均值 μ 为基线、组内标准差 σ 为归一化因子，计算相对优势 Â_i = (r_i - μ)/σ。这本质上是在问"这条响应比组内平均水平好多少？"

**失效场景**（三种典型情况）：
1. **全组都错**（all-negative group）：所有 G 条响应都得到最低奖励，组内无区分度，标准差为零或极小，无法提供有效学习信号（LENS / 频谱策略优化等方案尝试解决此问题）。
2. **全组都对**：同样缺乏对比信息，模型无法从"都是正确的答案中"学会区分优劣。
3. **Group size 过小**（如 G=2）：统计量不稳定，优势估计方差大，训练容易崩溃——实践中通常需要 G≥8。

---

> **数据说明**：本报告中的 GitHub Stars 数据、论文信息、性能指标等均来源于 2025-2026 年的公开信息。Stars 数量为截至搜索时刻的近似值，可能随时间变化。建议读者参考原始链接获取最新数据。

**Sources:**
- [DeepSeek-R1 GitHub (90K+ stars)](https://github.com/deepseek-ai/DeepSeek-R1)
- [Open-R1 Hugging Face (38K+ stars)](https://github.com/huggingface/open-r1)
- [DeepSeekMath: GRPO 原始论文](https://arxiv.org/pdf/2402.03300)
- [DeepSeek-R1 论文](https://arxiv.org/abs/2501.12948)
- [DAPO 论文](https://arxiv.org/pdf/2503.14476)
- [GSPO 论文](https://arxiv.org/pdf/2507.18071)
- [Understanding R1-Zero (Dr. GRPO)](https://arxiv.org/abs/2503.20783)
- [GRPO 官方推导 (Hugging Face Course)](https://huggingface.co/learn/llm-course/en/chapter12/3b)
- [GRPO vs PPO vs DPO 实验对比 (HPC-AI)](https://www.hpc-ai.com/blog/GRPO_vs_other_RL_guide)
- [GRPO, DAPO, DUPO, GSPO 对比 (腾讯云)](https://cloud.tencent.cn/developer/article/2592688)
- [DeepSeek R1 四阶段训练 (腾讯云)](https://cloud.tencent.cn/developer/article/2500270)
- [RE-GRPO 论文 (Neurocomputing)](https://www.sciencedirect.com/science/article/abs/pii/S0925231225032151)
- [P-GRPO (Apple ML Research)](https://machinelearning.apple.com/research/personalized-group)
- [ICLR 2026 Off-Policy GRPO](https://iclr.cc/virtual/2026/poster/10011320)
- [Post-Training in 2026](https://llm-stats.com/blog/research/post-training-techniques-2026)
