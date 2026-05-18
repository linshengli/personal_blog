# 智能体层级化规划与子目标分解机制 · 深度调研报告

> **调研日期**：2026-05-18
> **所属领域**：Agent / AI Agent
> **报告版本**：v1.0

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

智能体层级化规划（Hierarchical Planning）与子目标分解（Subgoal Decomposition）是指：将高层级、长时域的复杂目标任务，自上而下逐层拆解为更小、更易执行的子目标或子任务，并在执行过程中进行层级化协调与控制的机制。核心思想源于经典人工智能中的分层规划（Hierarchical Task Network, HTN），在 LLM Agent 时代被赋予了新的内涵——利用大语言模型的语义理解能力进行动态、递归的目标分解与规划。

### 常见误解

1. **误解：层级化规划 = 流水线式任务拆解**
   实质：真正的层级化规划不是简单的"步骤1→步骤2"线性拆分，而是包含递归分解、依赖关系管理、跨层反馈和动态调整的多层次架构。

2. **误解：子目标分解只需要 LLM 就能自动做好**
   实质：LLM 的语义分解能力虽强，但缺乏可执行性验证和错误恢复机制。最优做法是将 LLM 的语义理解与符号规划的可验证性相结合（如 ChatHTN 范式）。

3. **误解：层级越多越精确**
   实质：过度分解会导致规划膨胀、执行延迟增加、错误累积。研究表明，每个节点 2-4 个子目标是较优实践，超出后收益递减。

### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **Plan-and-Execute** | 先规划再执行，规划和执行是前后两个独立阶段；层级化规划则在执行过程中持续动态分解和调整 |
| **ReAct (Reasoning+Acting)** | 单层"思考-行动-观察"循环，无显式层级结构；层级化规划有显式的抽象层级和跨层信息传递 |
| **CoT / ToT 推理** | 主要在推理空间内展开思维过程，不涉及与环境交互的行动规划；层级化规划关心的是可执行的任务结构 |

---

## 1.2 核心架构

### 层级化规划系统通用架构

```
┌──────────────────────────────────────────────────────────────┐
│                  智能体层级化规划系统架构                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ╔═══════════════════════════════════════════════════════╗     │
│  ║              战略规划层 (Strategic Layer)              ║     │
│  ║  接收原始目标 → 全局分解 → 里程碑定义 → 优先级排序      ║     │
│  ╚═════════════════════╦═════════════════════════════════╝     │
│                         │ 递归分解                             │
│  ╔══════════════════════╩══════════════════════════════════╗    │
│  ║              战术规划层 (Tactical Layer)                ║    │
│  ║  子目标依赖图(DAG) → 资源分配 → 执行路径选择 → 重规划     ║    │
│  ╚══════════════════════╦══════════════════════════════════╝    │
│                         │ 任务下发                             │
│  ╔══════════════════════╩══════════════════════════════════╗    │
│  ║              操作执行层 (Operational Layer)             ║    │
│  ║  原子动作/工具调用 → 环境交互 → 反馈采集 → 状态更新       ║    │
│  ╚═══════════════════════════════════════════════════════╝     │
│                                                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐  │
│  │  工作记忆        │  │  技能库/方法库   │  │  监控与异常   │  │
│  │  (Working Mem)   │  │  (Skill/Method) │  │  (Monitor)   │  │
│  └─────────────────┘  └─────────────────┘  └──────────────┘  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

**各层职责说明**：
- **战略规划层**：接收自然语言描述的终极目标，利用 LLM 或符号规划器产出高层子目标序列，定义关键里程碑和优先级
- **战术规划层**：维护子目标的 DAG（有向无环图）依赖关系，调度执行路径，在失败时触发部分重规划
- **操作执行层**：将原子子目标映射为具体工具调用、API 请求或物理动作，执行并返回结果
- **工作记忆**：跨层级共享环境状态、中间结果和上下文信息
- **技能库/方法库**：存储已验证的子目标分解模式（即 HTN 中的"方法"），供重用

---

## 1.3 数学形式化

### 公式 1：层级化规划的基本分解递归

$$G_0 \rightarrow \{g_1, g_2, ..., g_n\} \mid \bigwedge_{i=1}^{n} \text{Satisfiable}(g_i, s) \land \text{Order}(g_1 \prec g_2 \prec ... \prec g_n)$$

一个顶层目标 $G_0$ 被分解为一组有序子目标 $\{g_i\}$，要求每个子目标在当前状态 $s$ 下可满足，且子目标间有偏序约束。

### 公式 2：子目标可行性判定（SSE 方法）

$$\text{Reachable}(g_i \mid s) = \mathbb{1}\left[ \min_{\pi_h} D(s, \pi_h(g_i)) < \epsilon \right]$$

来自 Strict Subgoal Execution (SSE) 框架：子目标 $g_i$ 从当前状态 $s$ 可达，当且仅当存在高层策略 $\pi_h$，其与目标的距离度量的最小值小于阈值 $\epsilon$。

### 公式 3：规划树的搜索复杂度

$$C_{\text{total}} = \sum_{d=0}^{D} b^d \cdot (C_{\text{LLM}} + C_{\text{verify}})$$

其中 $D$ 为树的最大深度，$b$ 为平均分支因子（建议 2-4），$C_{\text{LLM}}$ 为单次 LLM 调用的成本，$C_{\text{verify}}$ 为可行性验证成本。该公式解释了为何层级化规划的计算成本随深度指数增长。

### 公式 4：多分辨率技能的价值函数

$$V^{\text{MRS}}(s) = \max_{k \in \{1,...,K\}} \left[ Q^k_{\text{meta}}(s, k) + \gamma^{\tau_k} \cdot \mathbb{E}_{s' \sim P^k(\cdot|s, \pi_k)}[V^{\text{MRS}}(s')] \right]$$

来自 Multi-Resolution Skills (MRS)：元控制器选择最适合当前状态的技能分辨率 $k$，其中 $\tau_k$ 为时间跨度，时间跨度越长的技能折扣因子越强。

### 公式 5：重规划触发条件

$$\text{Replan} \iff \exists i : \text{Confidence}(g_i \mid s_t) < \theta_{\text{fail}} \lor \Delta_{\text{state}}(s_t, \hat{s}_t) > \delta_{\text{drift}}$$

重规划触发于两种情况：某个子目标在当前状态的可信度低于阈值 $\theta_{\text{fail}}$，或实际状态 $s_t$ 与预期状态 $\hat{s}_t$ 的偏离度超过漂移容忍度 $\delta_{\text{drift}}$。

---

## 1.4 实现逻辑（Python 伪代码）

```python
class HierarchicalPlanner:
    """层级化规划系统的核心抽象"""

    def __init__(self, llm_backend, max_depth=5, max_children=4):
        self.strategic = StrategicPlanner(llm_backend)      # 战略分解
        self.tactical = TacticalPlanner(llm_backend)         # 依赖管理与调度
        self.operational = OperationalExecutor(llm_backend)  # 原子动作执行
        self.memory = WorkingMemory()                        # 跨层工作记忆
        self.max_depth = max_depth
        self.max_children = max_children

    def plan_and_execute(self, goal: str, state: dict) -> Result:
        """主入口：从目标到执行的完整流程"""
        # 1. 战略层：递归分解
        plan_tree = self.strategic.decompose(
            goal=goal,
            state=state,
            depth=0,
            max_depth=self.max_depth,
            max_children=self.max_children
        )
        # plan_tree 是一个 DAG，节点为子目标，边为偏序依赖

        # 2. 战术层：调度执行 + 动态重规划
        execution_order = self.tactical.topological_sort(plan_tree)
        for subgoal in execution_order:
            actual_state = self.memory.get_current_state()

            # 检查是否需要重规划
            if self._should_replan(subgoal, actual_state):
                plan_tree = self._replan(plan_tree, subgoal, actual_state)
                execution_order = self.tactical.topological_sort(plan_tree)
                continue

            # 3. 操作层：执行
            result = self.operational.execute(subgoal, actual_state)
            self.memory.update(subgoal, result)

        return self.memory.get_final_result()

    def _should_replan(self, subgoal, current_state) -> bool:
        confidence = self.tactical.estimate_confidence(subgoal, current_state)
        drift = self.tactical.measure_drift(current_state, subgoal.expected_state)
        return confidence < 0.3 or drift > 0.5

    def _replan(self, tree, failed_node, current_state) -> DAG:
        """局部重规划：仅重分解失败节点及下游"""
        affected = self.tactical.get_downstream_nodes(tree, failed_node)
        tree.remove_subtree(failed_node)
        new_subtree = self.strategic.decompose(
            goal=failed_node.original_goal,
            state=current_state,
            depth=failed_node.depth
        )
        tree.merge(new_subtree)
        return tree


class StrategicPlanner:
    """战略层：递归目标分解"""

    def decompose(self, goal, state, depth, max_depth, max_children=4):
        if depth >= max_depth or self._is_primitive(goal):
            return GoalNode(goal, is_primitive=True)

        subgoals = self.llm.decompose(goal, state, n=max_children)
        # subgoals = [{"description": "...", "dependencies": [...]}, ...]

        children = []
        for sg in subgoals:
            child_tree = self.decompose(
                sg["description"], state, depth + 1, max_depth, max_children
            )
            children.append(child_tree)

        return GoalNode(goal, children=children, dependencies=sg.get("dependencies", []))
```

---

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 任务成功率 | > 60% (WAH-NL) | 端到端基准测试（多轮评估） | 2026年 SOTA 水平（ReAcTree: 61%） |
| 规划延迟 | < 3s / 层 | LLM 调用计时 | 每层分解的响应时间，取决于模型能力 |
| 执行步数压缩比 | > 2x | 子目标数 ÷ 原子动作数 | 相较于扁平规划的步骤压缩效率 |
| 重规划响应时间 | < 1s | 局部重规划触发到新计划输出 | 关键用户体验指标 |
| 规划有效性 | > 85% | 子目标可执行率 | 分解出的子目标中可被执行的比例 |
| 上下文效率 | < 4K tokens / 子目标 | 每子目标的 token 消耗 | 子目标感知上下文压缩后的效率 |
| 鲁棒性 | > 95% | 50轮执行中不崩溃的概率 | 系统在连续执行中的稳定性 |

---

## 1.6 扩展性与安全性

### 水平扩展

- **多 Agent 并行分解**：同一层级的不同子目标可分配给独立 Agent 节点并行执行（如 ReAcTree 的控制流节点支持并行分支）
- **技能库分布式共享**：已验证的分解模式可存入共享技能库，跨会话复用，减少重复 LLM 调用
- **分治调度**：大规模任务可拆分为独立的子问题分发到不同计算节点

### 垂直扩展

- **单节点优化**：通过结构化输出（Pydantic Schema）约束 LLM 输出格式，减少解析错误和重试
- **上下文压缩**：子目标感知的上下文压缩（HiAgent 方案）可将无关上下文压缩掉，降低 token 消耗
- **缓存分解模式**：对常见任务模式进行方法级缓存（如 ChatHTN 中的学习方法）

### 安全考量

1. **目标漂移**：长时间执行中 Agent 可能偏离原始目标。解决方案：引入"目标锚定"机制，每轮检查当前活动与顶层目标的语义对齐度
2. **分解爆炸**：递归分解可能无限膨胀。解决方案：最大深度限制（典型值 5）+ 最大子目标数限制（典型值 4）
3. **幻觉传播**：上层分解错误会逐级放大到下层执行。解决方案：每个子目标执行后增加可行性验证门控，失败时触发局部重规划
4. **权限级联**：高层 Agent 可能赋予低层 Agent 超出其安全边界的权限。解决方案：最小权限原则，每层 Agent 只能调用其被授权范围的工具

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **MineDojo/Voyager** | 6,900 | 开放式具身 Agent，自动课程+技能库+迭代提示，Minecraft 中持续探索学习 | Python/JS | 2023-05 | [GitHub](https://github.com/MineDojo/Voyager) |
| **SkyworkAI/DeepResearchAgent** | 3,400 | 分层多 Agent 系统，顶层规划 Agent 协调多个专用子 Agent，自演进协议 | Python | 2026-02 | [GitHub](https://github.com/SkyworkAI/DeepResearchAgent) |
| **GoalAct (cjj826/GoalAct)** | 新项目 | 持续更新的全局规划机制 + 分层执行策略，LegalAgentBench SOTA | Python | 2025-04 | [GitHub](https://github.com/cjj826/GoalAct) |
| **OpenGVLab/GITM** | 641 | 三层分层架构(分解器→规划器→接口)，Minecraft 100% 科技树 | Python | 2023-05 | [GitHub](https://github.com/OpenGVLab/GITM) |
| **Choi-JaeWoo/ReAcTree** | 新项目 | 动态 LLM Agent 树，控制流节点协调并行/串行/条件执行，AAMAS 2026 | Python | 2026-02 | [GitHub](https://github.com/Choi-JaeWoo/ReAcTree) |
| **NJUNLP/CogGen** | 新项目 | 认知启发的递归多 Agent 框架，宏观+微观双循环架构，ACL 2026 | Python | 2026 | [GitHub](https://github.com/NJUNLP/CogGen) |
| **mirasurf/cogents-core** | 活跃开发 | DAG 目标分解 (Goalith 模块)，LLMDecomposer 结构化分解 | Python | 2026 | [DeepWiki](https://deepwiki.com/mirasurf/cogents-core/6-configuration-and-environment) |
| **axioma-ai-labs/nevron** | 活跃开发 | 三层规划架构(战略→战术→操作)，PlanTree + ReplanningEngine | Python | 2026 | [GitHub](https://github.com/axioma-ai-labs/nevron) |
| **zju-vipa/Odyssey** | 活跃项目 | 40+ 原子技能 + 183 组合技能，微调 LLaMA-3 用于开放世界 | Python | 2024-06 | [GitHub](https://github.com/zju-vipa/Odyssey) |
| **Itakello/Co-voyager** | 活跃项目 | JSON 分层任务分解，依赖追踪，人机协作 | Python | 2024 | [GitHub](https://github.com/Itakello/Co-voyager) |

> **说明**：标注"新项目"的为 2025-2026 年间发布，stars 仍在快速增长中；"活跃开发"的为框架/库级别的项目，stars 数据随版本发布而变动。

---

## 2.2 关键论文（12 篇）

### 经典高影响力论文（奠基性工作，约 40%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响 |
|------|----------|------|----------|---------|------|
| **Voyager: An Open-Ended Embodied Agent with LLMs** | Wang et al. (NVIDIA) | 2023 | arXiv | 首个 LLM 驱动的终身学习 Agent，自动课程+技能库 | 6.9k GitHub Stars，开创性范式 |
| **Ghost in the Minecraft (GITM)** | Zhu et al. (OpenGVLab) | 2023 | arXiv | 三层分层架构(Decomposer→Planner→Interface)，100% 科技树 | 首个显式分层规划的 LLM Agent |
| **ReAct: Synergizing Reasoning and Acting in LLMs** | Yao et al. (Google/Princeton) | 2022 | ICLR 2023 | 推理-行动闭环范式 | 最广泛采用的 Agent 模式 |
| **Tree of Thoughts: Deliberate Problem Solving with LLMs** | Yao et al. (Google/Princeton) | 2023 | NeurIPS 2023 | 树状多分支推理，搜索+评估 | 推理范式的重大突破 |

### 最新 SOTA 论文（前沿进展，约 60%）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 |
|------|----------|------|----------|---------|-----------|
| **ReAcTree: Hierarchical LLM Agent Trees** | Choi et al. (Korea) | 2025 | AAMAS 2026 Full Paper | 动态 Agent 树+控制流节点，WAH-NL 61% vs ReAct 31% | 均分性能翻倍 |
| **STEP Planner** | Zhou et al. (BIT) | 2025 | IROS 2025 | 跨层级子目标树，闭路分解+终止双模型 | 具身规划新 SOTA |
| **Strict Subgoal Execution (SSE)** | - | 2025 | ICLR 2026 | 图结构 HRL，单步子目标可达性约束 | HRL 可靠性理论突破 |
| **Multi-Resolution Skills (MRS)** | - | 2026 | arXiv | 多分辨率目标预测模块，元控制器选择 | 精度-平滑度权衡建模 |
| **GoalAct: Global Planning & Hierarchical Execution** | Chen et al. (Tsinghua) | 2025 | arXiv | 持续更新全局规划+分层执行策略 | LegalAgentBench +12.22% |
| **ChatHTN: Interleaving LLM and Symbolic HTN** | Muñoz-Avila et al. | 2025 | NeuS 2025 | LLM 近似+符号 HTN 可验证规划 | 可证明正确的 LLM 规划 |
| **HTN Planning with LLM-Generated Heuristics** | Meneguzzi et al. | 2026 | 投稿 NeurIPS 2026 | LLM 生成 Python 启发式函数给 Pytrich 规划器 | 83% 问题减少搜索量 |
| **HCL-GP: Hierarchical Generalized Planning** | Sohrabi et al. (IBM) | 2026 | arXiv | 广义规划+HTN 分解，可复用策略组件 | AppWorld 98.2% 准确率 |

---

## 2.3 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|---------|----------|------|------|---------|------|
| **Hierarchical Planning with Goal Decomposition and Replanning** | Nevron (GitHub Issue #163) | EN | 架构设计文档 | 三层规划的战略→战术→操作架构，PlanTree 数据结构，ReplanningEngine 设计 | 2026 |
| **ReAcTree大模型任务规划入门** | AwesomeML | 中文 | 入门教程 | 从零实现层次化任务分解，动态树构建，3 种控制流节点 | 2026 |
| **Deep Agent 任务规划（Planner）是如何实现的** | CSDN/cnblogs | 中文 | 实战教程 | HTN + ToT + Self-Projection 实现 Planner，含 Python 代码 | 2025-2026 |
| **ReAct, Plan-and-Execute, or Reflection? The Three Agent Patterns** | Gabriel Anhaia (Dev.to) | EN | 模式对比 | 三大 Agent 模式的生产选型指南，含延迟/成本/监控指标 | 2026 |
| **面向自主智能体的任务分解与决策优化：动态规划方法** | 华为云 BBS | 中文 | 理论与实践 | DP + LLM 混合分解，TSP 式子目标序列优化，Python 代码 | 2025-10 |
| **Select-then-Solve: Paradigm Routing for LLM Agents** | arXiv 论文解读 | EN | 研究分析 | 18,000 次运行比较 6 种范式，无单一范式占主导 | 2026-04 |
| **The Best AI Agent Frameworks for 2026** | Signadot | EN | 框架调研 | LangChain/CrewAI/AutoGen 等框架的层级化规划能力对比 | 2026 |
| **AI Agent框架设计全解析：从理论到落地的完整指南** | 百度开发者 | 中文 | 综合指南 | Agent 架构设计全景，含层次化规划模块详解 | 2026 |
| **LangChain vs CrewAI vs AutoGen vs Dify: Complete Agent Framework Comparison** | Agdex (Dev.to) | EN | 框架对比 | 2026 年四大框架的规划、执行、编排能力全面比较 | 2026 |
| **为什么93%的 AI Agent 在复杂任务中"想得清却走不远"？** | FuncLens (CSDN) | 中文 | 深度分析 | 规划-执行失配症的原因分析，3 套已验证的 Prompt-Action 协同模板 | 2026 |

---

## 2.4 技术演进时间线

```
2022 ── ReAct 提出（Yao et al.）── 推理-行动闭环成为 Agent 基础范式
2023 ── Tree of Thoughts（树状推理）── 多分支搜索引入 Agent 推理
2023 ── Voyager（开放式具身 Agent）── 自动课程 + 技能库，首次展示 LLM 驱动的终身学习
2023 ── GITM（三层分层架构）── 首次显式使用分层规划(Decomposer→Planner→Interface)
2023 ── HTN 方法开始被 LLM Agent 社区重新发掘
2024 ── Subgoal-based HRL for Multi-Agent Collaboration
2024 ── EPO（环境偏好优化）── 引入分层 LLM Agent 的训练信号从环境反馈中获得
2025 ── ChatHTN（LLM+符号HTN 交错）── 结合 LLM 灵活性与符号规划的可证明正确性
2025 ── ReAcTree（动态 Agent 树）── 控制流节点实现灵活的并行/串行/条件规划
2025 ── GoalAct（全局规划+分层执行）── 持续更新全局规划与分层执行的紧耦合
2025 ── STEP Planner（子目标树具身规划）── 闭路分解+终止判断双模型
2025 ── Strict Subgoal Execution (SSE) ── 单步子目标可达性约束的理论保证
2026 ── Multi-Resolution Skills (MRS) ── 多时间跨度技能 + 元控制器选择
2026 ── HCL-GP（层次化广义规划）── 可复用策略组件，AppWorld 98.2%
2026 ── HTN Planning with LLM Heuristics ── LLM 生成启发式函数引导符号规划器
2026 ── 当前状态：层级化规划从理论走向工程落地，LLM + 符号系统融合成为主流范式
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
2022-2023 ─┬─ ReAct 范式确立，Agent 进入"推理+行动"时代
           └─ HTN 分层思想开始重新受到关注
2023-2024 ─┬─ GITM/Voyager 展示 LLM Agent 做长程任务规划的可行性
           └─ 分层 HRL (HIRO/HRAC) 提出可达性子目标生成理论
2024-2025 ─┬─ LLM + 符号 HTN 融合（ChatHTN）提供可证明正确的规划
           └─ 动态 Agent 树（ReAcTree）实现规划时动态扩展
           └─ 全局规划+分层执行（GoalAct）引入紧耦合设计
2025-2026 ─┬─ 多分辨率技能（MRS）建模精度-平滑度权衡
           └─ 严格子目标执行（SSE）提供理论可达性保证
           └─ LLM 生成启发式引导符号规划器（HTN+Heuristics）
           └─ Agent 框架原生支持层级化规划（LangGraph/CrewAI/AutoGen）
```

## 3.2 5 种核心方案横向对比

### 方案 A：纯 LLM 提示驱动分解（Zero-shot / Few-shot）

| 维度 | 描述 |
|------|------|
| **原理** | 通过精心设计的 Prompt 让 LLM 直接输出子目标列表，无需额外结构约束 |
| **优点** | (1) 实现极简，仅需提示词；(2) 适合快速原型验证；(3) 灵活适应任意任务领域 |
| **缺点** | (1) 输出的可执行性无保证；(2) 无法约束递归深度；(3) 缺乏错误恢复机制；(4) 幻觉传播无法检测 |
| **适用场景** | 概念验证、简单任务、低风险场景 |
| **成本量级** | ~$0.01-0.05/次 |

### 方案 B：结构化分解框架（LLM + 约束 Schema）

| 维度 | 描述 |
|------|------|
| **原理** | LLM 输出通过 Pydantic/JSON Schema 约束为结构化格式（DAG、子目标树），增强可解析性和可执行性 |
| **优点** | (1) 输出可解析保证；(2) 支持依赖关系建模；(3) 多种回退解析策略；(4) 易集成到现有系统 |
| **缺点** | (1) 仍需 LLM 保证语义正确性；(2) 复杂依赖关系可能超出 Schema 表达能力；(3) 分解质量依赖模型能力 |
| **适用场景** | 中小型生产系统，需要可靠解析的任务 |
| **成本量级** | ~$0.05-0.2/次 |

代表项目：Cogents Core (Goalith)、GoalAct

### 方案 C：LLM + 符号规划器混合（LLM-HTNs）

| 维度 | 描述 |
|------|------|
| **原理** | LLM 负责语义理解和分解生成，符号规划器（如 Pytrich、PANDA）负责可执行性验证和搜索，两者交错运行 |
| **优点** | (1) 可证明正确的规划结果（ChatHTN）；(2) LLM 生成的启发式可显著加速符号搜索；(3) 错误可追溯；(4) 支持在线学习方法减少 LLM 调用 |
| **缺点** | (1) 实现复杂度高；(2) 需要领域建模（方法/操作符定义）；(3) 符号规划器对大规模问题有组合爆炸风险 |
| **适用场景** | 对正确性要求高的领域（航天、医疗、法律）；具有明确操作符结构的任务 |
| **成本量级** | ~$0.1-0.5/次（LLM 调用）+ 符号规划计算成本 |

代表项目：ChatHTN、HTN+LLM Heuristics、HCL-GP

### 方案 D：动态 Agent 树（ReAcTree 范式）

| 维度 | 描述 |
|------|------|
| **原理** | 动态构建 Agent 节点树，每个节点处理一个子目标，控制流节点（顺序/并行/条件）协调执行策略，树结构随任务复杂度动态生长 |
| **优点** | (1) 灵活的动态扩展能力；(2) 并行执行提升效率；(3) 局部失败不导致整体重启；(4) 任务粒度自动适配 |
| **缺点** | (1) 需要防止无限树扩展（深度/广度限制）；(2) 控制流管理复杂度高；(3) 跨节点上下文传递挑战性大 |
| **适用场景** | 复杂、不可预测的长时域任务；需要并行处理的任务 |
| **成本量级** | ~$0.2-1.0/次（取决于树大小） |

代表项目：ReAcTree、STEP Planner

### 方案 E：层级化强化学习（HRL + Subgoal）

| 维度 | 描述 |
|------|------|
| **原理** | 高层策略生成子目标（状态空间中的中间状态），低层策略学习如何到达这些子目标，通过两层或多层 RL 联合训练 |
| **优点** | (1) 可在无专家演示的情况下自动发现子目标；(2) 数学理论完备（收敛性、可达性证明）；(3) 适合连续控制和物理交互 |
| **缺点** | (1) 训练样本效率低；(2) 子目标空间设计困难；(3) 随机环境中子目标可达性难保证；(4) 迁移能力有限 |
| **适用场景** | 机器人控制、游戏 AI、连续控制任务 |
| **成本量级** | 训练：$10K-100K+ GPU 小时；推理：~$0.001/次 |

代表项目：SSE、MRS、HalfWeg、DHP

---

## 3.3 技术细节对比

| 维度 | A: 纯提示驱动 | B: 结构化分解 | C: LLM+符号HTN | D: 动态Agent树 | E: 层级化RL |
|------|:---:|:---:|:---:|:---:|:---:|
| **任务成功率** | 低 (20-35%) | 中 (40-55%) | 高 (70-85%) | 高 (55-65%) | 中-高 (因任务而异) |
| **可解释性** | 低 | 中 | 高（可追溯） | 中-高 | 低 |
| **部署难度** | 极低 | 低-中 | 高 | 中-高 | 极高 |
| **可扩展性** | 差 | 中 | 中 | 优 | 良 |
| **错误恢复** | 无 | 手动 | 自动（局部重规划） | 自动（局部回滚） | 通过训练隐式恢复 |
| **领域迁移** | 优 | 良 | 需重新建模 | 良 | 差（需重新训练） |
| **输入Token效率** | 中 | 优（约束输出） | 中 | 中-低（树大时） | 不适用（无LLM） |
| **成熟度** | 生产就绪 | 生产就绪 | 研究阶段 | 研究→早期生产 | 研究成熟 |
| **社区生态** | 广泛（所有框架） | LangChain/LlamaIndex | 学术社区 | 新兴 | RL 社区 |

---

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **小型项目/原型验证** | 方案 A → B | 从纯提示快速验证，需要结构化解析时过渡到 B | $50-200 |
| **中型生产环境（客服/自动化）** | 方案 B + D | 结构化分解保证可靠性，动态树处理复杂多步骤请求 | $500-3,000 |
| **大型分布式系统（企业级）** | 方案 B + C + D 复合 | LLM+HTN 保证核心流程正确性，动态树处理复杂分支，结构化输出兜底 | $3,000-20,000 |
| **机器人/具身AI** | 方案 E + D | HRL 处理连续控制，LLM Agent 树处理高层语义规划 | $10,000-100,000+（含训练） |
| **对正确性有严格要求的场景** | 方案 C | 符号规划器的可证明正确性是唯一选择 | $1,000-5,000 |
| **高吞吐的标准化任务** | 方案 B（缓存+技能库） | 熟化分解模式缓存后，大幅减少 LLM 调用 | $200-1,000 |

### 2026 年推荐实践

对于 **80% 以上的工业场景**，当前最务实的方案组合是：**以结构化分解框架（方案 B）为骨架**，嵌入动态 Agent 树（方案 D）处理长时域复杂任务的关键路径。在需要正确性保证的核心流程中，引入符号规划器进行可行性验证（方案 C 的轻量级子集）。避免纯提示驱动（方案 A）直接上生产，也避免过早引入完整的 HRL（方案 E）。

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{层级化规划} = \underbrace{\text{语义分解}}_{\text{LLM 将目标拆解为可理解的子目标}} + \underbrace{\text{符号验证}}_{\text{规划器检查子目标是可执行的}} - \underbrace{\text{分解-执行失配}}_{\text{规划与执行间的 gap 导致失败}}
$$

## 4.2 一句话解释

> 就像一位项目经理把 "建一栋楼" 这个大目标拆成 "打地基→盖框架→装水电→内部装修" 等一系列可执行的小任务，并在执行过程中根据实际情况调整计划——智能体层级化规划就是用 AI 来做这件事，只不过是在数字世界里。

## 4.3 核心架构图

```
用户目标: "帮我做一份 Q2 市场分析报告"
        ↓
┌──────────────── 战略分解 ────────────────┐
│  [收集数据] → [分析趋势] → [撰写报告] → [审校发布] │
└────────────────┬──┬──┬──────────────────┘
                 │  │  │
        ┌────────┘  │  └────────┐
        ↓           ↓           ↓
   ┌────────┐ ┌──────────┐ ┌────────┐
   │ 爬取数据│ │ 数据分析 │ │ 内容生成│
   │ 清洗处理│ │ 图表制作 │ │ 格式排版│
   └──┬─────┘ └────┬─────┘ └───┬────┘
      ↓            ↓            ↓
┌──────────────────────────────────────┐
│           操作执行层                   │
│  工具调用: API / 代码 / 数据库 / 文件  │
└──────────────────────────────────────┘
        ↓            ↓            ↓
┌──────────────────────────────────────┐
│         监控 & 重规划门控              │
│  执行失败 → 局部回滚 → 重新分解       │
└──────────────────────────────────────┘
```

## 4.4 STAR 总结

| 部分 | 内容 | 字数 |
|------|------|------|
| **Situation（背景+痛点）** | 当前 LLM Agent 在简单任务上表现优异，但在复杂长时域任务（>10 步）中面临严峻挑战：目标漂移（执行几步后遗忘原始目标）、上下文断层（规划信息在长执行序列中丢失）、分解-执行失配（规划出的子目标不可执行）。研究表明约 93% 的 Agent 在复杂任务中面临"想得清却走不远"的问题。 | 130 字 |
| **Task（核心问题）** | 如何让 Agent 将高层复杂目标可靠地分解为可执行子目标序列，在执行过程中保持目标一致性，并在环境反馈与预期不符时自适应地调整规划——同时控制计算成本在可接受范围内？核心约束是：不依赖完整领域模型、支持动态环境变化、保证子目标的可达性。 | 120 字 |
| **Action（主流方案）** | 技术演进经历了三个阶段：(1) 纯 LLM 提示驱动阶段（2022-2023），ReAct/CoT 建立基础但缺乏结构保证；(2) LLM+结构约束阶段（2023-2025），引入 HTN 方法、结构化输出、技能库（GITM、Voyager、GoalAct）；(3) 融合验证阶段（2025-2026），LLM 与符号规划器交错运行（ChatHTN、SSE、MRS），动态 Agent 树（ReAcTree）实现运行时规划调整，子目标可达性理论提供了可行性保障。 | 165 字 |
| **Result（效果+建议）** | 当前 SOTA 方案在 WAH-NL 上达到 61% 成功率（较 ReAct 翻倍），AppWorld 上达 98.2% 准确率。主要局限：计算成本仍较高（ReCAP 约 ReAct 的 3 倍），泛化能力有限。建议：80% 工业场景采用结构化分解+动态 Agent 树组合，核心流程引入符号验证；避免全线采用纯提示驱动，避免过度分解（每节点 2-4 子目标为优）。 | 110 字 |

## 4.5 理解确认问题

**问题**：在层级化规划系统中，当一个子目标在执行时因环境状态变化而变得不可达，应该如何响应？请从"最小影响"原则出发描述处理流程。

**参考答案**：应当**不重启整个规划**，而是执行局部重规划（Local Replanning）：(1) 检测到子目标 g_i 不可达（置信度 < 阈值或状态偏离 > 容忍度）；(2) 确定 g_i 的依赖影响范围——找到所有直接或间接依赖 g_i 输出的下游子目标集合 D(g_i)；(3) 从规划树中移除 g_i 及其下游 D(g_i) 构成的子树；(4) 以 g_i 的原始父目标为输入，在**当前实际状态**下重新调用分解（而非回退到初始状态）；(5) 将新的子树合并回原规划树；(6) 按新的拓扑排序继续执行。这一过程的关键在于"局部"——只影响失败节点及其下游，不波及无关的并行分支。ReAcTree 和 SSE 框架均在不同程度上实现了这一机制。

---

# 参考资料索引

1. ReAcTree - arXiv:2511.02424 (AAMAS 2026) - https://arxiv.org/abs/2511.02424
2. STEP Planner - arXiv:2506.21030 (IROS 2025) - https://arxiv.org/abs/2506.21030
3. Strict Subgoal Execution - arXiv:2506.21039 (ICLR 2026) - https://arxiv.org/abs/2506.21039
4. Multi-Resolution Skills - arXiv:2505.21410 - https://arxiv.org/abs/2505.21410
5. GoalAct - arXiv:2504.16563 - https://arxiv.org/abs/2504.16563
6. ChatHTN - PMLR v288 (NeuS 2025) - https://proceedings.mlr.press/v288/munoz-avila25a.html
7. HTN Planning with LLM Heuristics - arXiv:2605.07707 - https://arxiv.org/abs/2605.07707
8. HCL-GP - arXiv:2605.06957 - https://arxiv.org/abs/2605.06957
9. Voyager - arXiv:2305.16291 - https://arxiv.org/abs/2305.16291
10. GITM - arXiv:2305.17144 - https://arxiv.org/abs/2305.17144
11. SkyworkAI/DeepResearchAgent - https://github.com/SkyworkAI/DeepResearchAgent
12. Cogents Core (Goalith) - https://deepwiki.com/mirasurf/cogents-core/6-configuration-and-environment
13. Nevron Hierarchical Planning - https://github.com/axioma-ai-labs/nevron/issues/163
14. Select-then-Solve - arXiv:2604.06753 - https://arxiv.org/abs/2604.06753
15. Deep Agent Planner 教程 - https://www.cnblogs.com/yangykaifa/p/19560083
16. 华为云 任务分解博客 - https://bbs.huaweicloud.com/blogs/466424
