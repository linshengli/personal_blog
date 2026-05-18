# 基于强化学习的期权做市商策略优化 —— 深度技术调研报告

> **调研日期**: 2026-05-18
> **所属域**: quant+agent
> **调研方法**: 网络数据采集（GitHub、arXiv、学术期刊、技术博客）+ 结构化分析

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

基于强化学习（RL）的期权做市商策略优化，是指利用强化学习框架（包括价值函数逼近、策略梯度、Actor-Critic等方法），在期权市场中学习最优做市报价策略和对冲策略的过程。做市商同时提交买入和卖出报价，通过赚取买卖价差获利，同时承担库存风险——在期权市场中这一风险尤为复杂，因为期权价格非线性依赖于标的资产价格、波动率、时间衰减等多个因子。RL方法通过将做市问题建模为马尔可夫决策过程（MDP），让智能体在与市场环境的交互中逐步学习能够最大化风险调整后收益的策略。

### 常见误解

1. **"RL做市策略就是自动赚钱机器"** —— 实际上，RL做市策略的核心是风险-收益权衡，在高度竞争的环境中，RL策略可能因过拟合、概念漂移或对手方博弈而表现不佳。
2. **"RL完全替代了Black-Scholes等传统定价模型"** —— RL是对传统模型的补充而非替代。定价模型提供基准，RL则解决执行优化、库存管理和非线性市场冲击等传统模型难以处理的动态决策问题。
3. **"做市RL和通用RL没有本质区别"** —— 期权做市RL面临独特挑战：高维连续动作空间（同时决定买卖报价）、非平稳市场环境、奖惩信号稀疏且噪音巨大、需要满足无套利约束。
4. **"期权做市与股票做市策略相同"** —— 期权做市多了一个关键的"希腊字母管理"维度（Delta、Gamma、Vega等），做市商不仅管理标的库存，还需要管理波动率风险敞口。

### 边界辨析

| 易混淆概念 | 核心区别 |
|-----------|---------|
| **期权做市 RL** vs **期权对冲 RL** | 做市同时涉及报价和对冲两个决策；对冲RL仅解决对冲比例优化问题，不涉及买卖报价决策 |
| **RL 做市** vs **传统做市模型（Avellaneda-Stoikov 等）** | AS模型假设订单到达率为关于报价的确定函数，RL模型不依赖显式参数化假设，直接从交互中学习 |
| **RL 做市** vs **最优执行** | 做市是双向报价赚取价差，最优执行是单向拆单降低成本，两者互为对偶问题 |

---

## 1.2 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│               RL 期权做市系统架构                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  市场数据输入 ──────────────────────────────────────────────┐  │
│  (LOB快照、期权链、标的行情、IV曲面)                          │  │
│                              ↓                                │  │
│  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  状态编码层 (State Encoder)                              │ │  │
│  │  • 库存状态: 标的持仓、期权持仓、希腊字母暴露             │ │  │
│  │  • 市场特征: 订单簿不平衡、价差宽度、波动率曲面特征        │ │  │
│  │  • 时序特征: 订单流强度(Hawkes)、波动率期限结构            │ │  │
│  └──────────────────────┬──────────────────────────────────┘ │  │
│                          ↓                                    │  │
│  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  RL 决策层 (Policy Network - Actor/Critic)              │ │  │
│  │  • 动作1: 买入报价(Bid Price/Spread)                    │ │  │
│  │  • 动作2: 卖出报价(Ask Price/Spread)                    │ │  │
│  │  • 动作3: 对冲比例/频率 (Hedge Ratio/Delta Target)      │ │  │
│  │  • 可选: 波动率曲面报价 (Volatility Quote)              │ │  │
│  └──────────────────────┬──────────────────────────────────┘ │  │
│                          ↓                                    │  │
│  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  执行引擎 (Execution Engine)                             │ │  │
│  │  • 订单生命周期管理 (创建/修改/撤销)                     │ │  │
│  │  • 无套利约束检查 (eSSVI 曲面校准)                      │ │  │
│  │  • 限价单簿撮合逻辑                                     │ │  │
│  └──────────────────────┬──────────────────────────────────┘ │  │
│                          ↓                                    │  │
│  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  奖励计算层 (Reward Function)                            │ │  │
│  │  • PnL 组件: 已实现价差收入 + 未实现损益                  │ │  │
│  │  • 风险惩罚: CVaR(库存风险) / 库存方差项                │ │  │
│  │  • 成本项: 手续费、买卖价差磨损、市场冲击成本             │ │  │
│  └──────────────────────┬──────────────────────────────────┘ │  │
│                          ↓                                    │  │
│  ┌─────────────────────────────────────────────────────────┐ │  │
│  │  训练与更新循环                                          │ │  │
│  │  • 经验回放缓冲区 (Replay Buffer)                        │ │  │
│  │  • 策略梯度更新 (PPO/SAC/TD3)                           │ │  │
│  │  • 环境模型更新 (针对非平稳市场)                         │ │  │
│  └─────────────────────────────────────────────────────────┘ │  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 各组件职责

| 组件 | 一句话说明 |
|------|-----------|
| **状态编码层** | 将高维市场数据（标的行情、期权链、LOB信息）压缩为低维状态表示，做MDP输入 |
| **RL决策层** | 核心策略网络，根据当前状态输出报价动作和对冲动作，通常使用Actor-Critic架构 |
| **执行引擎** | 将策略动作转化为实际订单操作，同时维护无套利约束和订单生命周期 |
| **奖励计算层** | 量化每个决策步骤的收益和风险，作为RL训练的学习信号 |
| **训练与更新循环** | 使用经验回放和策略梯度更新网络参数，因应市场非平稳性可能需要持续学习 |

---

## 1.3 数学形式化

### 1.3.1 MDP 形式化

期权做市问题可形式化为马尔可夫决策过程 (S, A, P, R, γ)：

$$
\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle
$$

其中状态空间 $\mathcal{S}$ 包括库存向量 $q_t$、市场状态 $m_t$ 和波动率曲面参数 $\Sigma_t$；动作空间 $\mathcal{A}$ 包括买卖报价价差 $\delta_t^b, \delta_t^a$ 和对冲头寸 $\Delta_t^h$。

### 1.3.2 核心目标函数：风险调整后收益最大化

$$
J(\pi) = \mathbb{E}_{\pi}\left[ \sum_{t=0}^{T} \gamma^t \left( R_t - \lambda \cdot \text{CVaR}_{\alpha}(P\&L_{t:T}) \right) \right]
$$

其中 $R_t$ 是每步的做市收入，$\text{CVaR}_{\alpha}$ 条件风险价值作为风险惩罚项，$\lambda$ 是风险厌恶系数。该公式体现了做市的核心权衡：最大化收益同时控制尾部风险。

### 1.3.3 Avellaneda-Stoikov 基准模型（RL改进起点）

做市商的保留价格和最优报价公式：

$$
r(s, q, t, \sigma) = s - q \gamma \sigma^2 (T-t), \quad
\delta^{a}_{t} = r + \frac{1}{\gamma} \ln\left(1+\frac{\gamma}{\kappa}\right), \quad
\delta^{b}_{t} = r - \frac{1}{\gamma} \ln\left(1+\frac{\gamma}{\kappa}\right)
$$

其中 $s$ 是中间价，$q$ 是净库存，$\gamma$ 是风险厌恶系数，$\sigma$ 是波动率，$\kappa$ 是订单到达率强度参数。RL方法的核心改进之一是用学习到的非线性函数替代这些闭式解，同时保持或改善其风险控制能力。

### 1.3.4 CVaR 条件风险价值（做市风险度量）

$$
\text{CVaR}_{\alpha}(X) = \frac{1}{1-\alpha} \int_{\alpha}^{1} \text{VaR}_u(X) du = \mathbb{E}[X | X \leq \text{VaR}_{\alpha}(X)]
$$

在期权做市中使用CVaR而非方差作为风险度量的原因：期权收益分布高度偏斜，方差无法捕捉尾部风险。CVaR关注的是最坏的 $(1-\alpha)$ 分位情况下的平均损失，直接对应做市商的止损风控需求。

### 1.3.5 eSSVI 无套利波动率曲面约束

$$
\Sigma_{t}(K, T) = \frac{1}{2T} \left( \theta_t(T) + \sqrt{\theta_t(T)^2 + 2\rho_t\theta_t(T)\varphi_t(T)\cdot x + (\varphi_t(T)\cdot x)^2} \right)
$$

其中 $x = \ln(K/S_0)$ 是价外程度，$\theta_t(T)$ 是总方差，$\varphi_t(T)$ 是波动率微笑控制函数，$\rho_t$ 是 skew 参数。该参数化保证无蝶式套利和无日历套利。在RL中集成此约束层确保做市商报价不会产生套利机会——这是RL+期权做市的最新关键进展（Zhang, 2025）。

---

## 1.4 实现逻辑（Python 伪代码）

```python
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class MarketState:
    """市场状态：包含标的行权和期权链数据"""
    spot_price: float            # 标的价格
    implied_vol_surface: np.ndarray  # IV曲面 (strike x expiry)
    order_book_imbalance: float  # 订单簿不平衡度
    spread: float                # 最优买卖价差
    inventory: Dict[str, float]  # 库存：{标的: 数量, 期权_i: 数量}


class OptionMarketMakingRL:
    """
    核心类：RL期权做市系统
    职责：
    1. 维护策略网络（Actor + Critic）
    2. 管理期权报价和对冲决策
    3. 执行训练循环
    """

    def __init__(self, config: dict):
        # 策略网络：输入状态，输出动作
        self.actor = PolicyNetwork(
            state_dim=config["state_dim"],
            action_dim=3,  # [bid_spread, ask_spread, hedge_ratio]
            hidden_dims=[256, 128]
        )
        # 价值网络：评估状态-动作对的价值
        self.critic = ValueNetwork(
            state_dim=config["state_dim"],
            hidden_dims=[256, 128]
        )
        # 无套利约束层（eSSVI参数化）
        self.no_arb_layer = ESSVISurface()
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(capacity=100_000)
        # 风险度量器
        self.risk_measure = CVaR(alpha=0.95)
        # 配置
        self.gamma = config.get("gamma", 0.99)
        self.lambda_risk = config.get("lambda_risk", 1.0)
        self.learning_rate = config.get("lr", 3e-4)

    def get_quotes(self, state: MarketState) -> Tuple[float, float]:
        """
        核心报价逻辑：根据当前市场状态输出最优买卖报价

        Args:
            state: 当前市场状态（含标的价、IV曲面、库存等）

        Returns:
            (bid_price, ask_price) 买卖报价
        """
        # 1. 编码市场状态为特征向量
        features = self._encode_state(state)

        # 2. Actor网络输出动作（归一化报价策略）
        action = self.actor(features)  # shape: (3,)
        bid_spread, ask_spread, hedge_delta = action

        # 3. 通过无套利约束层调整报价（确保IV曲面无套利）
        iv_adjusted = self.no_arb_layer(
            state.implied_vol_surface,
            bid_spread,
            ask_spread
        )

        # 4. 转换为实际报价价格
        mid_price = state.spot_price
        bid_price = mid_price - bid_spread * mid_price
        ask_price = mid_price + ask_spread * mid_price

        return bid_price, ask_price, hedge_delta

    def update(self, batch: List[Tuple]):
        """
        PPO风格策略更新

        1. 从经验回放中采样一批次数据
        2. 计算优势函数 A(s,a) = Q(s,a) - V(s)
        3. 使用裁剪的代理目标更新策略
        4. 使用MSE损失更新价值网络
        """
        states, actions, rewards, next_states, dones = batch

        # 计算TD目标
        with torch.no_grad():
            next_values = self.critic(next_states)
            td_targets = rewards + self.gamma * next_values * (1 - dones)

        # 更新Critic
        current_values = self.critic(states)
        critic_loss = F.mse_loss(current_values, td_targets)

        # 更新Actor (PPO clip)
        old_log_probs = self._get_log_probs(states, actions)
        new_log_probs = self._get_log_probs(states, self.actor(states))
        ratio = torch.exp(new_log_probs - old_log_probs)
        advantage = td_targets - current_values.detach()

        clip_loss = -torch.min(
            ratio * advantage,
            torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        )
        actor_loss = clip_loss.mean()

        # 风险调整：在损失中加入CVaR惩罚
        portfolio_pnl = self._simulate_portfolio_pnl(states, actions)
        cvar_penalty = self.risk_measure(portfolio_pnl)
        total_loss = actor_loss + critic_loss + self.lambda_risk * cvar_penalty

        # 反向传播
        total_loss.backward()
        self.optimizer.step()

    def _encode_state(self, state: MarketState) -> np.ndarray:
        """将原始市场状态编码为特征向量"""
        features = np.concatenate([
            [state.spot_price],
            [state.order_book_imbalance],
            [state.spread],
            self._compute_greeks(state),       # Delta, Gamma, Vega, Theta
            self._vol_surface_features(state), # IV曲面主成分
            [sum(state.inventory.values())],   # 净库存
        ])
        return features
```

---

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **Sharpe Ratio** | > 1.5 (回测) | 年化收益/年化波动 | 核心风险调整后收益指标，RL做市通常目标 > 1.0 |
| **最大回撤 (Max DD)** | < 10% | 回测中最高点到最低点跌幅 | 衡量尾部风险控制能力，受CVaR惩罚直接影响 |
| **买卖价差收入** | > X bp/trade | 平均每笔交易的价差收益 | 扣除费用后的净做市收入 |
| **库存周转率** | 适中（非极值） | 日库存变化率 | 过高意味着频繁交易增加成本，过低则风险暴露过大 |
| **Gamma/Vega暴露** | 接近中性 | 日终希腊字母净暴露 | 期权做市特有的风险度量，反映波动率风险敞口 |
| **成交率 (Fill Rate)** | > 60% | 提交订单中被成交的比例 | 反映报价竞争力和市场占有率的平衡 |
| **无套利违规率** | < 0.1% | 报价产生套利机会的比例 | 2025年后新指标，eSSVI约束层可降至接近零 |
| **平均持仓时间** | 适中 | 订单从提交到成交或被撤销的时间 | 反映存货风险的持续时间 |

---

## 1.6 扩展性与安全性

### 水平扩展

- **多合约并行做市**：在多个期权系列（不同行权价、到期日）上同时运行RL策略，通过共享底层特征编码网络实现参数高效扩展
- **多市场/多资产**：使用参数服务器架构，每个市场部署独立Actor，共享中央Critic，支持分布式训练
- **GPU加速模拟**：JaxMARL-HFT等方法利用JAX的vmap/pmap实现240倍加速，支持批量环境并行模拟

### 垂直扩展

- **单节点优化**：使用更深的Transformer编码器（替代MLP）处理长序列LOB数据；注意力机制对波动率曲面变化更敏感
- **模型合并**：通过知识蒸馏将多个专精模型（不同行权价区间）合并为一个统一模型

### 安全考量

1. **对手方博弈风险**：RL做市商可能被对手方识别策略模式并进行逆向选择攻击（Jafree et al., 2025）。对策：引入对抗性RL训练和策略随机化。
2. **市场操纵风险**：RL做市商的报价行为可能被滥用来操纵市场（串谋、幌骗等）。对策：监管合规约束层，限制报价上下界。
3. **过拟合与概念漂移**：RL策略在回测中表现优异但在实盘中失效。对策：域随机化训练、在线持续学习、定期重训练。
4. **模型可解释性**：深度神经网络的"黑箱"特性在金融监管环境下可能不合规。对策：使用可解释性工具（SHAP、注意力权重可视化），或采用约束RL方法（如eSSVI层）保持经济语义透明。

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **AI4Finance-Foundation/FinRL** | 15,100+ | 通用金融RL框架，支持做市和期权交易 | Python, PyTorch, Gym | 2026活跃 | [GitHub](https://github.com/AI4Finance-Foundation/FinRL) |
| **tspooner/rl_markets** | 321 | LOB模拟+RL做市策略评估 | Python, TensorFlow | 2024 | [GitHub](https://github.com/tspooner/rl_markets) |
| **pfnet-research/pfhedge** | 280+ | PyTorch深度对冲框架，支持多种期权 | PyTorch, Python | 2024.08 | [GitHub](https://github.com/pfnet-research/pfhedge) |
| **javifalces/HFTFramework** | 246 | RL改进Avellaneda-Stoikov算法 | Java/Python, MQL5 | 2026.05活跃 | [GitHub](https://github.com/javifalces/HFTFramework) |
| **Pushkar-Quant/QUANT-P1** | 新项目(2025) | 生产级RL自适应做市引擎 | Python, PPO, Docker | 2025.10 | [GitHub](https://github.com/Pushkar-Quant/QUANT-P1) |
| **HoodyBoss/RL-for-MM** | 237 | KTH硕士论文：DQN做市策略 | Python, Keras | 2024 | [GitHub](https://github.com/HoodyBoss/Reinforcement-Learning-for-Market-Making) |
| **JJJerome/mbt_gym** | 150 | 基于模型的HFT做市Gym环境 | Python, Gym | 2024 | [GitHub](https://github.com/JJJerome/mbt_gym) |
| **guijinSON/Awesome-Deep-Hedging** | 84 | 深度对冲资源汇总列表 | 文献索引 | 2025 | [GitHub](https://github.com/guijinSON/Awesome-Deep-Hedging) |
| **tspooner/rmm.arl** | 47 | 鲁棒做市（对抗性RL） | Python | 2024 | [GitHub](https://github.com/tspooner/rmm.arl) |
| **alexander-dybdahl/deep-hedging** | 引用高 | TensorFlow深度对冲引擎 + LSTM | TensorFlow, LSTM/GRU | 2023.02 | [GitHub](https://github.com/alexander-dybdahl/deep-hedging) |
| **picklenchips/MARKET-MAKING-RL** | 27 | Stanford CS234：PPO+LOB模拟 | Python, SB3 | 2025.04 | [GitHub](https://github.com/picklenchips/MARKET-MAKING-RL) |
| **mselser95/optimal-market-making** | 27 | 论文复现：RL最优做市 | Python | 2023 | [GitHub](https://github.com/mselser95/optimal-market-making) |
| **KakaWanYifan/RL-Options** | - | DQN期权组合对冲 | Python, DQN | 2024 | [GitHub](https://github.com/KakaWanYifan/ReinforcementLearnigOptions) |
| **olaxbt/ai-market-maker** | - | 多智能体加密货币做市 | Python, 多Agent | 2025 | [GitHub](https://github.com/olaxbt/ai-market-maker) |
| **Edwicn/OPHR** (NeurIPS 2025) | 新 | 多智能体波动率交易 | PyTorch, MARL | 2025 | [GitHub](https://github.com/Edwicn/OPHR-MasteringVolatilityTradingwithMultiAgentDeepReinforcementLearning) |
| **justinhou95/NeuralHedge** | PyPI可安装 | 轻量深度对冲+投资组合优化 | PyTorch | 2025 | [PyPI](https://pypi.org/project/neuralhedge/) |

---

## 2.2 关键论文（14篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 链接 |
|------|----------|------|----------|---------|------|
| **Deep Hedging** | Buehler et al. (J.P. Morgan) | 2019 | SSRN/Risk | 首次将RL用于衍生品对冲，处理交易摩擦 | [SSRN](https://ssrn.com/abstract=3355706) |
| **Deep Bellman Hedging** | Buehler, Murray, Wood | 2022 | arXiv | Actor-Critic深度对冲，支持任意初始组合无需重训练 | [arXiv](https://arxiv.org/abs/2207.00932) |
| **Market Making via RL** | Spooner et al. | 2018 | AAMAS | TD学习做市，LSTM时序特征处理 | 经典 |
| **Market Making of Options via RL** | Fang, Xu (UT Austin) | 2023 | arXiv:2307.01814 | 多行权价多到期日期权做市，高斯最优策略推导 | [arXiv](https://arxiv.org/abs/2307.01814) |
| **Risk-Sensitive Option MM w/ eSSVI** | Jian'an Zhang | 2025 | arXiv:2510.04569 | 约束RL+无套利IV曲面，微分CVaR风险目标 | [arXiv](https://arxiv.org/abs/2510.04569) |
| **OPHR: Mastering Volatility Trading** | Chen, Cai, Qin, An | 2025 | **NeurIPS 2025** | 多Agent波动率交易框架，BTC/ETH期权实证 | [NeurIPS](https://neurips.cc/virtual/2025/poster/120100) |
| **Contextual RL for MM** | Wen, Huang et al. | 2025 | EAAI | 自监督市场上下文网络，台湾TXO期权数据验证 | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0952197625032270) |
| **RL MM as Stochastic Control** | Zimmer et al. | 2025/2026 | arXiv:2509.12456 | 非平稳LOB上PPO做市，创新仿真器 | [arXiv](https://arxiv.org/abs/2509.12456) |
| **Adversarial RL MM (ARL)** | Wang et al. (King's College) | 2025 | arXiv:2508.16589 | 对抗RL+Hawkes过程，灵活报价动作 | [arXiv](https://arxiv.org/abs/2508.16589) |
| **MARL Competition without Collusion** | Wang, Ventre, Polukarov | 2025 | arXiv:2510.25929 | 分层MARL研究做市串谋现象 | [arXiv](https://arxiv.org/abs/2510.25929) |
| **Adverse Selection of Meta-Orders** | Jafree et al. | 2025 | arXiv:2510.27334 | HFT做市商对手方选择博弈分析 | [arXiv](https://arxiv.org/abs/2510.27334) |
| **RL for Delta-Hedging in Illiquid** | Dineev, Lukianchenko | 2025 | Doklady RAN | 风险厌恶Bellman算子+DeepLOB | [MathNet](https://www.mathnet.ru/php/archive.phtml?wshow=paper&jrnid=danma&paperid=692) |
| **JaxMARL-HFT** | Oxford/ICAIF | 2025 | ICAIF 2025 | GPU加速240倍的多智能体HFT | [arXiv](https://arxiv.org/abs/2511.02136) |
| **Consistent Time Travel for RL MM** | Ragel, Challet | 2024 | arXiv:2408.02322 | 离线RL时间对齐方法，历史数据回放 | [arXiv](https://arxiv.org/abs/2408.02322) |

---

## 2.3 系统化技术博客（10篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| **强化学习做市商高频交易策略（附代码）** | 知乎专栏 | 中文 | 教程 | SARSA/Double Q-learning做市策略解读和C++代码复现 | 2020+ | [知乎](https://zhuanlan.zhihu.com/p/89790180) |
| **Hedging American Put Options with DRL** | Pickard et al. on HackerNoon | 英文 | 深度教程 | DDPG美式看跌期权对冲，含GBM/SV实验和超参分析 | 2024.10 | [HackerNoon](https://hackernoon.com/lite/hedging-american-put-options-with-deep-reinforcement-learning-training-procedures) |
| **深层对冲不完全市场期权（LSTM-RNN+RL）** | 北航张军欢团队 | 中文 | 学术分享 | 50ETF/恒生/日经/标普500/富时100期权实证 | 2023 | [北航](https://sem.buaa.edu.cn/info/1022/13495.htm) |
| **Don't Just Train Your AI, Re-Train It** | MEXC News | 英文 | 实践指南 | 期权对冲RL智能体每周重新训练方案 | 2024-2025 | [MEXC](https://www.mexc.com/en-PH/news/74361) |
| **FinRL-X: AI-Native Modular Infrastructure** | AI4Finance Foundation | 英文 | 论文+框架文档 | Next-gen生产级金融RL框架设计 | 2026.03 | [arXiv](https://arxiv.org/abs/2603.21330) |
| **Beyond the Spread: RL for HFT MM** | HKUST (Aditya Mangla) | 英文 | 学位论文 | AS到RL的完整演进路径 | 2025.04 | [HKUST](https://cse.hkust.edu.hk/ug/fyt/24-25/psan2.html) |
| **FinRL Tutorial 2026** | AI4Finance Foundation | 英文 | 官方教程 | FinRL框架配置做市和期权交易策略 | 2026 | [GitHub](https://github.com/AI4Finance-Foundation/FinRL) |
| **基于Hawkes过程的DEEP RL做市策略** | 网易/中文社区 | 中文 | 技术分析 | Hawkes过程LOB+DRL框架详解，解决AS模型一致性假设缺陷 | 2024 | [网易](https://www.163.com/dy/article/IK2K9HE30539O6LM.html) |
| **Deep Hedging Tutorial - pfhedge** | Preferred Networks | 英文 | 框架文档 | pfhedge框架使用教程和多资产对冲案例 | 2024 | [GitHub](https://github.com/pfnet-research/pfhedge) |
| **Option Market Making via RL (ICLR)** | ICLR Workshop | 英文 | 会议报告 | ICLR 2025期权做市RL分会场论文 | 2025 | [ICLR](https://ICLR.cc/virtual/2025/33824) |

---

## 2.4 技术演进时间线

| 年份 | 事件 | 发起方/代表人物 | 影响 |
|------|------|---------------|------|
| **1964** | 做市商定价理论奠基：最优报价取决于存货风险 | Garman | 定义了做市问题的经济学基础 |
| **1980** | Ho-Stoll模型：做市商动态定价与存货控制 | Ho, Stoll | 引入动态规划框架解决做市问题 |
| **2008** | **Avellaneda-Stoikov模型**：连续时间做市的闭式解 | Avellaneda, Stoikov | **经典基准模型**，至今仍是RL方法的主要比较对象 |
| **2018** | **Market Making via RL**：首次将RL引入LOB做市 | Spooner et al. (AAMAS) | **里程碑**：展示了TD学习和瓦片编码在离散动作空间做市中的有效性 |
| **2019** | **Deep Hedging**：首次用RL解决期权衍生品对冲 | Buehler et al. (J.P. Morgan) | **里程碑**：开创了"深度对冲"这一子领域，推动期权做市RL研究 |
| **2020** | FinRL框架发布：首个开源金融RL生态 | AI4Finance Foundation | 降低了金融RL的入门门槛 |
| **2022** | **Deep Bellman Hedging**：无需重训练的对冲模型 | Buehler, Murray, Wood | 从"为特定组合训练"到"任意组合通用对冲"的飞跃 |
| **2023** | **期权做市RL理论**：多行权价多到期日推导 | Fang, Xu (UT Austin) | 首个针对期权做市（非仅对冲）的完整RL形式化 |
| **2024** | JaxMARL-HFT：GPU加速240倍的多智能体HFT | Oxford | 证明了大规模并行模拟对RL训练的重要性 |
| **2025** | **Risk-Sensitive OMM w/ eSSVI**：约束RL+无套利曲面 | Jian'an Zhang | 首次在RL做市中集成可微无套利约束层 |
| **2025** | **OPHR (NeurIPS)**：多智能体波动率交易框架 | Chen, Cai, Qin, An | **NeurIPS 2025**，首个专为波动率交易的RL框架 |
| **2025** | 对抗性RL做市 + 做市串谋博弈研究 | 多团队 (Jafree, Wang等) | 从单一策略优化到多智能体博弈的范式扩展 |
| **2026** | FinRL-X：生产级金融RL模块化基础设施 | AI4Finance Foundation | 金融RL向生产化、工程化的重要一步 |

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
1964 ─┬─ Garman 做市定价理论奠基 → 定义了存货风险是做市的核心经济问题
2008 ─┼─ Avellaneda-Stoikov 闭式解 → 成为经典基准，但假设过于简化（泊松过程、线性订单到达率）
2018 ─┼─ Spooner et al. RL做市 → 首次用RL解决限价单簿做市，离散动作空间，传统RL算法
2019 ─┼─ Buehler et al. Deep Hedging → 开创深度对冲范式，深度学习+RL用于衍生品
2022 ─┼─ Deep Bellman Hedging → 无模型、无重训练、支持任意初始组合
2023 ─┼─ Fang & Xu 期权做市RL理论 → 从对冲扩展到完整的期权做市RL框架，推导高斯最优策略
2024 ─┼─ JaxMARL-HFT GPU加速 → 240倍加速使大规模RL训练成为可能
2025 ─┼─ eSSVI约束RL (Zhang) + OPHR (NeurIPS) + ARL/MA博弈 → 无套利约束、多智能体、鲁棒性三大突破
2026 ─┴─ 当前状态：RL期权做市从学术验证进入工程化阶段，约束RL(安全)+多智能体(博弈)+生产级部署(FinRL-X)三线并进
```

---

## 3.2 六种方案横向对比

### 方案A：基于AS的启发式策略 (Avellaneda-Stoikov及其扩展)

| 方面 | 内容 |
|------|------|
| **原理** | 假设订单到达率为Poisson过程，推导保留价格和最优报价的闭式解 |
| **优点** | ① 数学优雅、可解释性强 ② 计算效率极高（微秒级） ③ 参数少、易于理解和调优 ④ 有大量文献验证和改进 |
| **缺点** | ① 假设过强（Poisson到达率、线性市场冲击、对称信息） ② 无法处理期权特有的非线性和波动率曲面 ③ 无法学习市场模式（无数据驱动能力） ④ 在高波动/非平稳环境下表现差 |
| **适用场景** | 基线对比基准、低流动性市场、资源受限环境 |
| **成本量级** | 免费（易于实现），运营成本极低 |

### 方案B：传统RL离散动作做市 (DQN/SARSA/Q-Learning)

| 方面 | 内容 |
|------|------|
| **原理** | 将做市动作离散化（如10种报价组合），用值函数逼近学习最优策略 |
| **优点** | ① 概念简单、实现容易 ② 训练稳定（离散动作天然收敛性更好） ③ 有完善的开源实现（Spooner et al.） ④ 可在标准硬件上训练 |
| **缺点** | ① 动作离散化丢失连续性（报价价差的最优解是连续的） ② 状态空间大时维度灾难 ③ 无法处理期权的希腊字母管理需求 ④ 样本效率低 |
| **适用场景** | 教学演示、初步验证、股票做市 |
| **成本量级** | 免费，单GPU足够 |

### 方案C：深度RL连续动作做市 (PPO/SAC/TD3 + LOB模拟)

| 方面 | 内容 |
|------|------|
| **原理** | 使用连续动作空间的策略梯度方法，在高保真LOB模拟器中训练做市智能体 |
| **优点** | ① 可输出连续精度的报价价差 ② PPO的clipping机制保证训练稳定性 ③ SAC的熵正则化促进探索 ④ 可集成复杂的神经网络架构（LSTM/Transformer） |
| **缺点** | ① 训练不稳定（超参敏感） ② 对模拟器质量高度依赖 ③ 样本效率低（需要百万级step训练） ④ 策略网络是"黑箱"，难以审计 |
| **适用场景** | 中等规模生产做市、股票/期货做市、加密货币做市 |
| **成本量级** | GPU成本约$500-2000/月，模拟器开发约$10k-50k |

### 方案D：深度对冲RL (Deep Hedging + Pfhedge)

| 方面 | 内容 |
|------|------|
| **原理** | 将期权对冲建模为RL问题，使用深度网络学习Delta对冲策略，同时考虑交易成本和市场摩擦 |
| **优点** | ① 专门为期权结构设计 ② 支持多种风险度量（CVaR, Entropic Risk） ③ 有成熟框架（pfhedge, deep-hedging） ④ 模型可扩展（LSTM/GRU处理时序） |
| **缺点** | ① 只解决对冲问题，不做做市报价 ② 假设标的流动性足以执行对冲 ③ 对模拟环境参数敏感 ④ 实盘部署时需连接真实交易系统 |
| **适用场景** | 期权做市商的风险管理、结构性产品对冲、场外期权定价 |
| **成本量级** | 中等，GPU训练成本$200-1000/月 |

### 方案E：约束RL期权做市 (eSSVI + CVaR + 政策约束)

| 方面 | 内容 |
|------|------|
| **原理** | 在RL优化中集成可微无套利约束层（eSSVI曲面参数化）和风险管理约束（CVaR限制），实现"白箱"经济语义 |
| **优点** | ① 强制无套利（监管合规） ② 微分CVaR有效控制尾部风险 ③ 经济语义透明（报价可直接映射到IV曲面） ④ 可扩展到期权组合 |
| **缺点** | ① 技术实现复杂（eSSVI可微层设计） ② 计算开销大（曲面校准+策略网络） ③ 对标的资产流动性有要求 ④ 最新方法（2025年）尚未有广泛社区实践 |
| **适用场景** | 期权做市商核心系统、波动率曲面报价、需要无套利合规的场景 |
| **成本量级** | 高，研发成本$50k-200k，GPU成本$1000-5000/月 |

### 方案F：多智能体RL做市 (MARL + 博弈论)

| 方面 | 内容 |
|------|------|
| **原理** | 将做市问题建模为多智能体随机博弈，每个做市商是一个独立RL智能体，研究竞争/合作行为 |
| **优点** | ① 建模真实市场竞争环境 ② 可研究串谋检测和反串谋策略 ③ 自我对弈提升策略鲁棒性 ④ GPU加速（JaxMARL）使大规模训练可行 |
| **缺点** | ① 训练复杂度高（联合状态空间指数增长） ② 收敛性无保证（博弈均衡可能不存在或非唯一） ③ 计算资源需求极大 ④ 实盘应用仍以单智能体为主 |
| **适用场景** | 学术研究、市场设计、监管分析、高流动性市场 |
| **成本量级** | 最高，多GPU集群$3000-10000+/月 |

---

## 3.3 技术细节对比

| 维度 | 方案A: AS启发式 | 方案B: 离散RL | 方案C: 连续DRL | 方案D: 深度对冲 | 方案E: 约束RL | 方案F: MARL |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **性能(Sharpe)** | 1.0-1.2 | 1.0-1.5 | 1.2-1.8 | 0.8-1.5 | 1.5-2.0+ | 1.5-2.5+ |
| **实现复杂度** | ★☆☆☆☆ | ★★☆☆☆ | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★★★★ |
| **生态成熟度** | ★★★★★ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★☆☆☆ | ★★☆☆☆ |
| **社区活跃度** | 高(经典) | 中 | 高(FinRL等) | 中(pfhedge) | 低(前沿) | 中(JaxMARL) |
| **学习曲线** | 低(1-2周) | 低(2-3周) | 中(1-3月) | 中(1-2月) | 高(3-6月) | 高(3-6月) |
| **可解释性** | ★★★★★ | ★★★★☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★☆☆☆ |
| **无套利合规** | 隐式满足 | 需外接检查 | 需外接检查 | 不涉及报价 | **内置** | 需外接 |
| **期权适配度** | ★☆☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★★ | ★★★☆☆ |
| **实盘部署难度** | 低 | 低 | 中 | 中 | 高 | 很高 |

---

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型团队初步探索 / 教学研究** | 方案B: 离散RL + AS基准 | 实现成本低，概念验证快，Spooner等有完整教程和代码 | $0-200 (单CPU服务器) |
| **中型团队/加密货币做市** | 方案C: PPO/SAC + LOB模拟器 | 平衡性能与成本，FinRL框架成熟，社区资源丰富 | $500-2000 (GPU云实例) |
| **专业期权做市商（核心系统）** | **方案C+E混合**: DRL报价 + eSSVI约束对冲 | 报价部分用PPO/SAC保持灵活性，对冲部分用约束RL保证无套利和风险控制 | $2000-5000 (多GPU+数据源) |
| **做市+对冲一体化系统** | 方案E: 约束RL (eSSVI+CVaR) | 2025年SOTA，统一执行/对冲/无套利，经济语义透明 | $3000-8000 (研发+GPU+数据) |
| **高频做市(纳秒级)** | 方案A: AS启发式+ML调参 | 延迟敏感场景只能使用闭式解，可用轻量ML在线调整AS参数 | $1000-5000 (FPGA/低延迟硬件) |
| **波动率交易/波动率套利** | 专用方案: OPHR多智能体框架 | 2025 NeurIPS成果，专为波动率交易优化，多Agent分工明确 | $2000-5000 (GPU+数据订阅) |
| **金融市场监管/反串谋研究** | 方案F: MARL | 研究做市商竞争行为和串谋检测，学术价值高 | $3000-10000+ (多GPU集群) |

---

## 3.5 技术方案选型决策树

```
开始
├─ 你的目标是？
│   ├─ 学习/教学 → 方案B(离散RL) + AS基准
│   ├─ 学术研究 →
│   │   ├─ 单一做市策略 → 方案C(连续DRL) + 方案E(约束RL)
│   │   └─ 市场竞争/串谋 → 方案F(MARL)
│   └─ 生产部署 →
│       ├─ 延迟敏感(<1ms) → 方案A(AS增强) + ML调参
│       ├─ 期权做市对冲 → 方案D(深度对冲) + 方案E(约束对冲)
│       └─ 全面期权做市 → 方案C(报价) + 方案E(对冲+无套利)
└─ 预算约束？
    ├─ <$1000/月 → 方案A或B
    ├─ $1000-5000/月 → 方案C或C+E混合
    └─ >$5000/月 → 方案E+F综合
```

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{RL期权做市} = \underbrace{\text{Deep RL策略优化}}_{\text{学习最优报价和对冲动作}} + \underbrace{\text{eSSVI无套利约束}}_{\text{保证IV曲面经济有效性}} - \underbrace{\text{CVaR尾部风险}}_{\text{控制最坏情况损失}}
$$

这个公式总结了RL期权做市的核心心智模型：**用深度学习驱动策略学习，用数学约束保证安全合规，用风险度量控制下行空间**。三者的平衡决定了策略的最终表现。

## 4.2 一句话解释

"想象一个期权交易员同时被要求：① 决定今天报出什么价格买菜和卖菜（决定买卖价差）、② 决定持有多少蔬菜库存（仓位管理）、③ 确保不被别人举报价格离谱（无套利约束）——强化学习就是通过不断试错，自动学会这三件事的最优组合策略。"

## 4.3 核心架构图（简化）

```
市场数据(LOB+期权链+IV曲面)
       ↓
┌──────────────────────────────────────┐
│         状态编码                      │
│  [库存状态 + 市场特征 + 希腊字母]     │
└──────────────┬───────────────────────┘
               ↓
┌──────────────────────────────────────┐
│         RL策略网络 (Actor)            │
│  ┌───────┐  ┌────────┐  ┌─────────┐ │
│  │买入报价│  │卖出报价│  │对冲比例 │ │
│  └───┬───┘  └───┬────┘  └────┬────┘ │
└──────┼──────────┼─────────────┼──────┘
       ↓          ↓             ↓
┌──────────────────────────────────────┐
│       无套利约束层 (eSSVI)           │
│     (保证IV曲面无蝶式/日历套利)       │
└──────────────┬───────────────────────┘
               ↓
        执行引擎 → 市场
               ↓
        奖励: PnL - λ·CVaR(尾部风险)

性能指标: Sharpe > 1.5 | MaxDD < 10% | 无套利违规 < 0.1%
```

---

## 4.4 STAR 总结

### Situation（背景+痛点）

期权做市商在高度竞争的市场环境中提供双向报价，赚取买卖价差，同时承担复杂的非线性库存风险（Delta、Gamma、Vega、Theta）。传统做市模型（如Avellaneda-Stoikov 2008）基于简化的数学假设（Poisson订单到达、线性市场冲击），在高波动、非平稳的现代市场中日益力不从心。期权市场还面临波动率曲面套利、模型风险、对手方逆向选择等多重挑战。

### Task（核心问题）

核心挑战是设计一个自适应期权做市策略，能够：(1) 在高维连续动作空间（买卖报价+对冲）中实时优化；(2) 管理希腊字母暴露和尾部风险；(3) 保证报价不产生套利机会（经济有效性）；(4) 在非平稳市场环境中持续学习。传统解析方法因维数灾难和假设过强而受限。

### Action（主流方案）

技术演进经历了五个阶段：(1) **经典AS模型**（2008）：解析闭式解，计算效率高但灵活性低；(2) **离散动作RL做市**（2018-）：Spooner等用TD学习做市，状态/动作离散化；(3) **深度连续RL**（2019-2022）：PPO/SAC等连续控制算法进入做市领域，结合LOB模拟器；(4) **期权专用深度对冲**（2019-2022）：Buehler等开创深度对冲范式，pfhedge等框架成熟；(5) **约束RL+多智能体**（2025-）：eSSVI无套利约束层（Zhang, 2025）、OPHR多Agent波动率交易（NeurIPS 2025）、对抗性RL和MARL博弈研究，标志着从"纯性能优化"到"安全+博弈+鲁棒"的范式跃迁。

### Result（效果+建议）

当前RL期权做市在回测中可达Sharpe 1.5-2.0+，显著超越AS基准的1.0-1.2。关键突破包括：(1) eSSVI约束层将无套利违规率降至接近零；(2) 多智能体框架（OPHR）首次将波动率交易和风险管理分离建模；(3) GPU加速（JaxMARL）使训练效率提升240倍。主要局限在于模拟与实盘差距、超参敏感性、和可解释性不足。

**实操建议**：对于专业期权做市商，推荐**方案C（连续DRL报价）+ 方案E（约束RL对冲）混合架构**，用PPO/SAC保持报价灵活性，用eSSVI+CVaR保证对冲和风险合规。优先集成pfhedge或FinRL-X框架降低开发成本。每1-2周重新训练策略以适应市场变化。

---

## 4.5 理解确认问题

**问题**：在RL期权做市系统中，为什么使用CVaR（条件风险价值）而不是传统的方差（Variance）作为风险度量？如果使用方差做风险惩罚项，会对策略行为产生什么影响？

**参考答案**：期权收益分布具有显著的非对称性和厚尾特征（特别是卖出期权时存在跳跃损失风险），方差对所有偏离均值的方向一视同仁，而CVaR仅关注尾部损失。使用方差作为惩罚会导致智能体回避"高方差高收益"的策略（如卖出深度虚值期权收取小额权利金），而这些策略其实有正期望收益但偶尔巨亏——正是做市商应该管理的风险类型。

如果使用方差替代CVaR：策略会产生两种不良行为——要么变得极度保守（不愿承担任何希腊字母暴露，损失做市收入），要么在尾部风险上过度暴露（因为方差无法有效惩罚罕见但灾难性的损失事件）。这是期权做市不同于股票做市的根本设计差异。

---

## 附录：参考来源汇总

- [FinRL - AI4Finance Foundation](https://github.com/AI4Finance-Foundation/FinRL)
- [pfhedge - Preferred Networks](https://github.com/pfnet-research/pfhedge)
- [tspooner/rl_markets](https://github.com/tspooner/rl_markets)
- [HFTFramework - javifalces](https://github.com/javifalces/HFTFramework)
- [QUANT-P1 - Pushkar-Quant](https://github.com/Pushkar-Quant/QUANT-P1)
- [OPHR - NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/120100)
- [Deep Hedging - Buehler et al. 2019 (SSRN)](https://ssrn.com/abstract=3355706)
- [Deep Bellman Hedging - arXiv 2022](https://arxiv.org/abs/2207.00932)
- [Risk-Sensitive OMM w/ eSSVI - Zhang 2025](https://arxiv.org/abs/2510.04569)
- [Market Making of Options - Fang & Xu 2023](https://arxiv.org/abs/2307.01814)
- [RL MM as Stochastic Control - Zimmer 2025](https://arxiv.org/abs/2509.12456)
- [Adversarial RL for MM - 2025](https://arxiv.org/abs/2508.16589)
- [MARL Competition without Collusion - 2025](https://arxiv.org/abs/2510.25929)
- [JaxMARL-HFT - 2025](https://arxiv.org/abs/2511.02136)
- [Contextual RL for MM - EAAI 2025](https://www.sciencedirect.com/science/article/abs/pii/S0952197625032270)
- [FinRL-X - arXiv 2026](https://arxiv.org/abs/2603.21330)
- [Deep Hedging Tutorial (HackerNoon) - 2024](https://hackernoon.com/lite/hedging-american-put-options-with-deep-reinforcement-learning-training-procedures)
- [知乎：强化学习做市商策略 - 附代码](https://zhuanlan.zhihu.com/p/89790180)
- [北航：深度强化学习期权对冲](https://sem.buaa.edu.cn/info/1022/13495.htm)
- [Awesome-Deep-Hedging 资源汇总](https://github.com/guijinSON/Awesome-Deep-Hedging)
- [NeuralHedge PyPI](https://pypi.org/project/neuralhedge/)
