# Agent Evaluation 概念剖析

> 调研日期：2026-03-01
> 主题：AI 智能体评估的核心原理与架构

---

## 一、定义澄清

### 通行定义

**Agent Evaluation（智能体评估）** 是指对基于大语言模型（LLM）的智能体系统进行系统性测试和度量的过程，旨在量化其在以下维度的表现：

1. **任务完成能力**：能否正确执行用户指定的目标
2. **工具使用能力**：能否有效调用外部 API、数据库、代码等工具
3. **规划与推理能力**：能否进行多步规划、逻辑推理、问题分解
4. **鲁棒性与安全性**：在边界情况和对抗输入下的表现
5. **效率与成本**：完成任务所需的时间、Token 消耗、API 调用次数

### 常见误解

| 误解 | 正确理解 |
|------|---------|
| **误解 1**：Agent 评估 = LLM 评估 | Agent 评估关注**系统行为**，LLM 评估关注**模型能力**。Agent 评估需考虑工具使用、记忆、规划等 LLM 之外的组件 |
| **误解 2**：准确率高 = 好 Agent | 高准确率可能掩盖**效率低下**（过度调用）、**安全隐患**（Prompt 注入漏洞）、**不可解释**（黑盒决策）等问题 |
| **误解 3**：基准测试 = 生产表现 | 基准测试（如 AgentBench）衡量**上限能力**，生产环境需额外评估**延迟、成本、故障恢复**等工程指标 |
| **误解 4**：LLM-as-a-Judge 完全可靠 | LLM 评判存在**位置偏差**、**自相似偏好**、**复杂推理局限**，需与人类评估校准 |
| **误解 5**：评估是一次性的 | Agent 评估是**持续过程**：模型更新、Prompt 迭代、工具变更都需重新评估 |

### 边界辨析

| 相邻概念 | 核心区别 |
|----------|---------|
| **LLM Benchmark** (MMLU, GSM8K) | 评测**纯语言模型**的知识与推理；Agent 评测**系统级行为**（含工具、记忆、规划） |
| **RAG Evaluation** (RAGAS) | 专注**检索增强生成**的检索质量与生成忠实度；Agent 评估范围更广（多步任务、工具链） |
| **Model Monitoring** | 关注**线上指标**（延迟、错误率、漂移）；Agent 评估含**离线能力评测** |
| **Unit Testing** | 验证**代码逻辑正确性**；Agent 评估处理**概率性输出**与**开放域任务** |

---

## 二、核心架构

```
┌────────────────────────────────────────────────────────────┐
│                    Agent Evaluation System                  │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │  Test Suite  │    │  Evaluator   │    │   Reporter   │ │
│  │   Generator  │───▶│    Engine    │───▶│   & Dashboard│ │
│  └──────────────┘    └──────────────┘    └──────────────┘ │
│         │                   │                   │          │
│         ▼                   ▼                   ▼          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Benchmark Environments                  │   │
│  ├────────────┬────────────┬────────────┬─────────────┤   │
│  │  WebArena  │ AgentBench │  SWE-bench │   Custom    │   │
│  │  (Web UI)  │  (Multi)   │  (Coding)  │  Env (API)  │   │
│  └────────────┴────────────┴────────────┴─────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Agent Under Test (AUT)                  │   │
│  ├────────────┬────────────┬────────────┬─────────────┤   │
│  │    LLM     │   Memory   │   Tools    │  Planner    │   │
│  │  (GPT-4/   │  (Vector/  │  (Search/  │  (CoT/ToT/  │   │
│  │  Claude)   │  Graph)    │  Code/DB)  │   ReAct)    │   │
│  └────────────┴────────────┴────────────┴─────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                 Metrics Layer                        │   │
│  ├────────────┬────────────┬────────────┬─────────────┤   │
│  │  Accuracy  │  Efficiency│   Safety   │  Cost       │   │
│  │  (Pass@k)  │  (Steps/   │  (Reward   │  ($/Task)   │   │
│  │            │   Latency) │  Hacking)  │             │   │
│  └────────────┴────────────┴────────────┴─────────────┘   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**组件职责说明：**

| 组件 | 功能 |
|------|------|
| **Test Suite Generator** | 生成或选取测试用例（来自基准或自定义），定义任务、输入、期望输出、评分标准 |
| **Evaluator Engine** | 执行测试：部署 Agent、运行任务、收集轨迹、计算指标 |
| **Reporter & Dashboard** | 可视化评估结果，支持对比分析、趋势追踪、问题诊断 |
| **Benchmark Environments** | 标准化测试环境（WebArena、AgentBench 等），确保评估可复现 |
| **Agent Under Test (AUT)** | 待评估的智能体，包含 LLM、Memory、Tools、Planner 四大核心组件 |
| **Metrics Layer** | 多维度指标体系，覆盖准确性、效率、安全性、成本 |

---

## 三、数学形式化

### 1. 任务完成率（Task Success Rate）

$$
\text{TSR} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left[\text{Verify}(\tau_i, g_i) = 1\right]
$$

- $N$：测试任务总数
- $\tau_i$：Agent 在第 $i$ 个任务上产生的执行轨迹（Trajectory）
- $g_i$：第 $i$ 个任务的目标描述
- $\text{Verify}(\cdot)$：验证函数，判断轨迹是否达成目标
- $\mathbb{I}[\cdot]$：指示函数，条件为真时取 1，否则为 0

**自然语言解释**：任务完成率 = 成功完成的任务数 / 总任务数

---

### 2. 效率分数（Efficiency Score）

$$
\text{ES} = \frac{1}{N} \sum_{i=1}^{N} \left( w_1 \cdot \frac{S_{\text{opt}}}{S_i} + w_2 \cdot \frac{T_{\text{opt}}}{T_i} + w_3 \cdot \frac{C_{\text{opt}}}{C_i} \right) \cdot \mathbb{I}[\text{Success}_i]
$$

- $S_i$：Agent 在第 $i$ 个任务上的实际步数
- $T_i$：实际耗时（秒）
- $C_i$：实际成本（Token 数或美元）
- $S_{\text{opt}}, T_{\text{opt}}, C_{\text{opt}}$：理论最优值（或基准值）
- $w_1, w_2, w_3$：权重系数（$w_1 + w_2 + w_3 = 1$）

**自然语言解释**：效率分数衡量 Agent 在成功完成任务的前提下，相对于最优解的步数、时间、成本的综合表现。

---

### 3. 工具使用准确率（Tool Call Accuracy）

$$
\text{TCA} = \frac{1}{M} \sum_{j=1}^{M} \mathbb{I}\left[\text{Tool}_j^{\text{pred}} = \text{Tool}_j^{\text{gold}} \land \text{Args}_j^{\text{pred}} \approx \text{Args}_j^{\text{gold}}\right]
$$

- $M$：总工具调用次数
- $\text{Tool}_j^{\text{pred}}$：Agent 预测的第 $j$ 次调用的工具名
- $\text{Args}_j^{\text{pred}}$：预测的参数
- $\approx$：参数相似度（可用编辑距离、语义相似度等度量）

**自然语言解释**：工具使用准确率 = 正确工具调用次数 / 总调用次数

---

### 4. 鲁棒性分数（Robustness Score）

$$
\text{RS} = \frac{1}{K} \sum_{k=1}^{K} \left(1 - \frac{|\text{Score}_{\text{clean}} - \text{Score}_{\text{perturbed}, k}|}{\text{Score}_{\text{clean}}}\right)
$$

- $K$：扰动类型数量（同义替换、噪声注入、对抗 Prompt 等）
- $\text{Score}_{\text{clean}}$：原始输入下的性能得分
- $\text{Score}_{\text{perturbed}, k}$：第 $k$ 种扰动下的性能得分

**自然语言解释**：鲁棒性分数衡量 Agent 在面对输入扰动时的性能稳定性，越接近 1 表示越鲁棒。

---

### 5. 综合评估分数（Overall Evaluation Score）

$$
\text{OES} = \alpha \cdot \text{TSR} + \beta \cdot \text{ES} + \gamma \cdot \text{TCA} + \delta \cdot \text{RS} - \epsilon \cdot \text{Risk}
$$

- $\alpha, \beta, \gamma, \delta, \epsilon$：可配置权重，反映不同场景下的优先级
- $\text{Risk}$：安全风险惩罚项（如 Reward Hacking 检测结果）

**自然语言解释**：综合分数 = 加权和（任务完成率 + 效率 + 工具准确率 + 鲁棒性） - 安全风险惩罚

---

## 四、实现逻辑（Python 伪代码）

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from abc import ABC, abstractmethod
import numpy as np

@dataclass
class Task:
    """测试任务定义"""
    id: str
    description: str
    initial_state: Dict[str, Any]
    goal: str
    success_criteria: Callable[[Dict], bool]
    max_steps: int = 50
    timeout_seconds: int = 300

@dataclass
class Trajectory:
    """Agent 执行轨迹"""
    steps: List[Dict[str, Any]]  # 每步：observation, action, tool_call, response
    success: bool
    total_steps: int
    total_time: float
    total_cost: float
    final_state: Dict[str, Any]

class Environment(ABC):
    """测试环境基类"""

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """重置环境，返回初始状态"""
        pass

    @abstractmethod
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """执行一步，返回新状态"""
        pass

    @abstractmethod
    def verify_success(self, state: Dict[str, Any], goal: str) -> bool:
        """验证是否达成目标"""
        pass

class AgentEvaluationSystem:
    """Agent 评估系统核心类"""

    def __init__(self, config: Dict[str, Any]):
        # 核心组件
        self.agent = config["agent"]  # 待评估的 Agent
        self.environments = config["environments"]  # 环境列表
        self.metrics_config = config["metrics"]  # 指标配置

        # 评估器组件
        self.judge_llm = config.get("judge_llm", None)  # LLM-as-a-Judge
        self.human_evaluators = config.get("human_evaluators", [])

        # 追踪器
        self.tracer = config.get("tracer", None)  # 执行轨迹追踪

    def run_evaluation(self, tasks: List[Task]) -> Dict[str, float]:
        """
        执行完整评估流程
        """
        trajectories = []

        for task in tasks:
            # 1. 选择合适的环境
            env = self._select_environment(task)

            # 2. 执行任务，收集轨迹
            trajectory = self._execute_task(task, env)
            trajectories.append(trajectory)

            # 3. 实时记录指标
            self._record_metrics(trajectory)

        # 4. 计算综合指标
        results = self._compute_aggregate_metrics(trajectories)
        return results

    def _execute_task(self, task: Task, env: Environment) -> Trajectory:
        """执行单个任务"""
        env.reset()
        steps = []
        total_cost = 0.0
        start_time = time.time()

        for step_num in range(task.max_steps):
            # Agent 决策
            observation = env.get_state()
            action = self.agent.decide(observation, task.goal)

            # 执行动作
            response = env.step(action)

            # 记录轨迹
            steps.append({
                "step": step_num,
                "observation": observation,
                "action": action,
                "response": response,
                "cost": action.get("cost", 0.0)
            })
            total_cost += action.get("cost", 0.0)

            # 检查终止条件
            if task.success_criteria(env.get_state()):
                return Trajectory(
                    steps=steps,
                    success=True,
                    total_steps=step_num + 1,
                    total_time=time.time() - start_time,
                    total_cost=total_cost,
                    final_state=env.get_state()
                )

        # 超时或达到最大步数
        return Trajectory(
            steps=steps,
            success=False,
            total_steps=task.max_steps,
            total_time=time.time() - start_time,
            total_cost=total_cost,
            final_state=env.get_state()
        )

    def _compute_aggregate_metrics(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """计算综合指标"""
        n = len(trajectories)

        # 任务完成率 (TSR)
        tsr = sum(1 for t in trajectories if t.success) / n

        # 效率分数 (ES)
        efficiency_scores = []
        for t in trajectories:
            if t.success:
                step_efficiency = 1.0 / t.total_steps  # 简化版
                time_efficiency = 1.0 / t.total_time
                cost_efficiency = 1.0 / t.total_cost
                es = (step_efficiency + time_efficiency + cost_efficiency) / 3
                efficiency_scores.append(es)
        es_avg = np.mean(efficiency_scores) if efficiency_scores else 0.0

        # 工具使用准确率 (TCA) - 需要从轨迹中提取工具调用
        tca_scores = []
        for t in trajectories:
            correct_calls = 0
            total_calls = 0
            for step in t.steps:
                if "tool_call" in step["action"]:
                    total_calls += 1
                    if self._verify_tool_call(step["action"], step["response"]):
                        correct_calls += 1
            if total_calls > 0:
                tca_scores.append(correct_calls / total_calls)
        tca_avg = np.mean(tca_scores) if tca_scores else 1.0  # 无工具调用时默认 1.0

        # 综合分数
        overall = (
            self.metrics_config["alpha"] * tsr +
            self.metrics_config["beta"] * es_avg +
            self.metrics_config["gamma"] * tca_avg
        )

        return {
            "task_success_rate": tsr,
            "efficiency_score": es_avg,
            "tool_call_accuracy": tca_avg,
            "overall_score": overall,
            "num_tasks": n,
            "num_successful": sum(1 for t in trajectories if t.success)
        }

    def _verify_tool_call(self, action: Dict, response: Dict) -> bool:
        """验证工具调用是否正确"""
        # 简化实现：检查工具是否返回错误
        return "error" not in response

    def _select_environment(self, task: Task) -> Environment:
        """根据任务类型选择环境"""
        # 简化实现：返回第一个环境
        return self.environments[0]

    def _record_metrics(self, trajectory: Trajectory):
        """记录单次轨迹的指标（用于实时监控）"""
        if self.tracer:
            self.tracer.log(trajectory)


# ==================== 使用示例 ====================

def example_usage():
    # 1. 定义测试任务
    tasks = [
        Task(
            id="web_001",
            description="在电商网站搜索商品并加入购物车",
            initial_state={"url": "https://example.com"},
            goal="找到'无线鼠标'并加入购物车",
            success_criteria=lambda state: "无线鼠标" in state.get("cart", []),
            max_steps=20,
            timeout_seconds=120
        ),
        # ... 更多任务
    ]

    # 2. 配置评估系统
    config = {
        "agent": my_agent,
        "environments": [WebArenaEnv(), AgentBenchEnv()],
        "metrics": {
            "alpha": 0.5,  # TSR 权重
            "beta": 0.3,   # ES 权重
            "gamma": 0.2   # TCA 权重
        },
        "judge_llm": "gpt-4o",
        "tracer": LangSmithTracer()
    }

    # 3. 执行评估
    evaluator = AgentEvaluationSystem(config)
    results = evaluator.run_evaluation(tasks)

    # 4. 输出结果
    print(f"任务完成率：{results['task_success_rate']:.2%}")
    print(f"效率分数：{results['efficiency_score']:.3f}")
    print(f"工具准确率：{results['tool_call_accuracy']:.2%}")
    print(f"综合得分：{results['overall_score']:.3f}")
```

---

## 五、性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **任务完成率 (TSR)** | > 70% (简单任务), > 40% (复杂任务) | 基准测试（AgentBench/WebArena） | 核心能力指标，但需结合其他指标综合评估 |
| **平均步数 (AST)** | < 10 步 (简单), < 30 步 (复杂) | 轨迹分析 | 反映规划效率，步数过多可能表示循环或迷失 |
| **端到端延迟** | < 5s (简单), < 30s (复杂) | 时间戳测量 | 用户体验关键指标，受 LLM 推理 + 工具调用影响 |
| **工具调用准确率** | > 90% | 黄金标注对比 | 反映 Agent 对工具 API 的理解与使用能力 |
| **Token 效率** | < 5000 tokens/任务 | API 用量统计 | 直接影响成本，优化空间：Prompt 压缩、缓存 |
| **成本/任务** | < $0.05 (简单), < $0.50 (复杂) | API 计费数据 | 商业可行性的关键，GPT-4 vs Claude 成本差异大 |
| **鲁棒性分数** | > 0.85 | 扰动测试集 | 衡量对抗输入、边界情况的稳定性 |
| **安全违规率** | < 1% | 安全测试集 | Reward Hacking、Prompt 注入等风险检测 |
| **人类对齐度** | > 4.0/5.0 | 人类评分 | 主观质量评估，补充自动化指标 |

---

## 六、扩展性与安全性

### 扩展性

| 维度 | 策略 | 上限 |
|------|------|------|
| **水平扩展** | 并行执行测试任务（分布式评估集群） | 千任务级并发，受预算和环境资源限制 |
| **垂直扩展** | 优化单任务评估效率（缓存、批量 LLM 调用） | 10 倍加速（相比串行） |
| **评估维度扩展** | 插件化指标定义，支持自定义评估器 | 无上限，取决于领域知识 |
| **环境扩展** | 标准化 Environment 接口，快速接入新基准 | 取决于基准平台开放性 |

### 安全性考量

| 风险类型 | 描述 | 防护措施 |
|----------|------|----------|
| **Reward Hacking** | Agent 利用评估规则漏洞获取高分但未真正完成任务 | 多指标交叉验证、人工抽检、对抗测试 |
| **Prompt 注入** | 测试环境中嵌入恶意指令影响 Agent 行为 | 输入过滤、沙箱环境、输出审计 |
| **数据泄露** | 测试数据污染（Agent 提前见过答案） | 数据隔离、动态生成测试用例 |
| **评估偏差** | LLM-as-a-Judge 的位置偏差、自相似偏好 | 多评判者投票、与人类评估校准 |
| **过度优化** | 针对特定基准过拟合，丧失泛化能力 | 多基准交叉评估、保留人工评估集 |
| **资源滥用** | 评估过程中 Agent 过度调用工具导致成本爆炸 | 预算限制、速率限制、异常检测 |

---

## 七、评估系统设计原则

1. **可复现性**：固定随机种子、版本锁定、环境快照
2. **模块化**：Agent、环境、指标、评判器可独立替换
3. **可扩展**：插件架构支持新基准、新指标、新工具
4. **透明性**：完整轨迹记录、可追溯决策过程
5. **效率优先**：并行执行、结果缓存、增量评估
6. **人机协作**：自动化为主，人工校准为辅

---

*本节完。下一节：行业情报 → 方案对比 → 精华整合*
