# AgentLoop 循环技术 - 概念剖析

> 调研日期：2026-02-28
> 主题：AgentLoop 循环技术的核心原理与架构

---

## 1. 定义澄清

### 通行定义

**AgentLoop**（Agent 循环）是 AI Agent 系统中的核心执行控制机制，它定义了 Agent 如何持续感知环境、做出决策、执行动作并观察结果的迭代过程。AgentLoop 是 Agent"思考"和"行动"的基本节奏，决定了 Agent 的行为模式和响应特性。

AgentLoop 的核心价值在于将静态的 LLM 推理转化为动态的、持续的行为序列，使 Agent 能够与环境交互、从反馈中学习、并在需要时调整策略，最终完成复杂的多步任务。

### 常见误解

| 误解 | 正解 |
|------|------|
| "AgentLoop 就是 while 循环调用 LLM" | AgentLoop 包含状态管理、终止条件、错误处理等复杂机制 |
| "循环次数越多效果越好" | 过多循环增加成本和延迟，需要设置合理上限 |
| "所有 Agent 使用相同循环模式" | 不同场景需要不同的循环策略（同步/异步、串行/并行） |
| "AgentLoop 只控制执行流程" | 还涉及记忆更新、资源管理、安全监控等多重职责 |

### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **AgentLoop vs EventLoop** | EventLoop 处理异步事件；AgentLoop 专注 Agent 决策 - 执行周期 |
| **AgentLoop vs 工作流引擎** | 工作流是预定义流程；AgentLoop 支持动态决策和自适应 |
| **AgentLoop vs 强化学习循环** | RL 循环优化策略参数；AgentLoop 执行推理和工具调用 |

---

## 2. 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                    AgentLoop 系统架构                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│                    ┌─────────────┐                           │
│                    │  触发条件   │                           │
│                    │  (Trigger)  │                           │
│                    └──────┬──────┘                           │
│                           ↓                                  │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  感知状态   │ ──→ │  决策推理   │ ──→ │  执行动作   │    │
│  │  (Observe)  │     │  (Reason)   │     │  (Act)      │    │
│  └─────────────┘     └─────────────┘     └──────┬──────┘    │
│         ↑                                       │          │
│         │              ┌─────────────┐          │          │
│         └──────────────│  观察结果   │←─────────┘          │
│                        │  (Result)   │                     │
│                        └──────┬──────┘                     │
│                               ↓                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐   │
│  │  终止判断   │ ←── │  状态更新   │ ←── │  记忆存储   │   │
│  │  (Done?)    │     │  (Update)   │     │  (Memory)   │   │
│  └──────┬──────┘     └─────────────┘     └─────────────┘   │
│         │                                                   │
│    ┌────┴────┐                                              │
│    ↓         ↓                                              │
│ [继续]    [终止] → 输出结果                                 │
└──────────────────────────────────────────────────────────────┘
```

**组件职责说明：**

| 组件 | 职责 |
|------|------|
| **触发条件** | 检测循环启动条件（用户输入、定时任务、事件触发） |
| **感知状态** | 收集环境信息、读取记忆、解析上下文 |
| **决策推理** | LLM 分析状态、生成计划、选择动作 |
| **执行动作** | 调用工具、执行代码、发送消息 |
| **观察结果** | 捕获执行反馈、错误信息、环境变化 |
| **记忆存储** | 更新对话历史、记录关键信息 |
| **状态更新** | 刷新内部状态、任务进度追踪 |
| **终止判断** | 检查任务完成、超限、用户中断等条件 |

---

## 3. 数学形式化

### 3.1 AgentLoop 状态转移

对于第 $t$ 次循环，状态转移方程：

$$S_{t+1} = f_{\text{loop}}(S_t, \text{LLM}(S_t), \text{Env}(\text{Act}_t))$$

其中：
- $S_t$ 表示第 $t$ 次循环的内部状态
- $\text{LLM}(S_t)$ 表示 LLM 基于当前状态生成的决策
- $\text{Env}(\text{Act}_t)$ 表示执行动作后的环境反馈

**自然语言解释**：下一次状态由当前状态、LLM 决策和环境反馈共同决定。

### 3.2 终止条件判定

循环终止的判定函数：

$$\text{Done}(S_t) = \mathbb{I}[\text{Complete}(S_t) \lor (t \geq T_{\max}) \lor \text{Interrupt}(S_t)]$$

其中 $T_{\max}$ 为最大循环次数限制。

**自然语言解释**：当任务完成、达到最大次数或被用户中断时，循环终止。

### 3.3 累积回报计算

对于 $T$ 次循环的总回报：

$$R = \sum_{t=1}^{T} \gamma^{t-1} \cdot r_t$$

其中 $\gamma$ 为折扣因子，$r_t$ 为第 $t$ 次循环的即时奖励。

**自然语言解释**：总回报是各次循环奖励的折现累积，早期奖励权重更高。

### 3.4 循环效率指标

循环效率定义为：

$$\text{Efficiency} = \frac{\text{TaskQuality}}{\text{LoopCount} \times \text{CostPerLoop}}$$

**自然语言解释**：效率等于任务质量除以总成本（循环次数 × 单次成本）。

---

## 4. 实现逻辑（Python 伪代码）

```python
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import time

class LoopStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"

@dataclass
class LoopState:
    """循环状态容器"""
    iteration: int = 0
    max_iterations: int = 20
    status: LoopStatus = LoopStatus.RUNNING
    history: List[Dict] = None
    context: Dict[str, Any] = None
    result: Any = None
    error: Optional[str] = None

class AgentLoop:
    """AgentLoop 核心类"""

    def __init__(self, llm, tools: List[Callable],
                 max_iterations: int = 20,
                 timeout: int = 300):
        self.llm = llm
        self.tools = {t.__name__: t for t in tools}
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.state = LoopState(max_iterations=max_iterations)

    def run(self, initial_task: str) -> Any:
        """
        运行 AgentLoop
        完整流程：感知 → 决策 → 执行 → 观察 → 更新 → 判断
        """
        start_time = time.time()
        self.state.context = {"task": initial_task, "history": []}

        while self._should_continue():
            # 检查超时
            if time.time() - start_time > self.timeout:
                self.state.status = LoopStatus.TIMEOUT
                self.state.error = "Loop timeout"
                break

            # 阶段 1: 感知状态
            observation = self._observe()

            # 阶段 2: 决策推理
            decision = self._reason(observation)

            # 阶段 3: 执行动作
            action_result = self._act(decision)

            # 阶段 4: 观察结果
            feedback = self._get_feedback(action_result)

            # 阶段 5: 状态更新
            self._update_state(decision, feedback)

            # 阶段 6: 终止判断
            if self._check_termination(decision, feedback):
                self.state.result = feedback
                self.state.status = LoopStatus.COMPLETED
                break

            self.state.iteration += 1

        return self._finalize()

    def _should_continue(self) -> bool:
        """判断是否继续循环"""
        return (
            self.state.status == LoopStatus.RUNNING and
            self.state.iteration < self.state.max_iterations
        )

    def _observe(self) -> Dict:
        """
        感知阶段：收集当前状态信息
        """
        return {
            "task": self.state.context.get("task"),
            "history": self.state.context.get("history", []),
            "current_step": self.state.iteration
        }

    def _reason(self, observation: Dict) -> Dict:
        """
        决策阶段：LLM 分析并生成行动计划
        """
        prompt = self._build_reasoning_prompt(observation)
        response = self.llm.generate(prompt)
        return self._parse_decision(response)

    def _act(self, decision: Dict) -> Any:
        """
        执行阶段：调用工具或执行动作
        """
        action_name = decision.get("action")
        action_args = decision.get("args", {})

        if action_name in self.tools:
            tool = self.tools[action_name]
            return tool(**action_args)
        elif action_name == "final_answer":
            return action_args.get("answer")
        else:
            return {"error": f"Unknown action: {action_name}"}

    def _get_feedback(self, action_result: Any) -> Dict:
        """
        观察阶段：获取执行反馈
        """
        return {
            "result": action_result,
            "success": action_result is not None,
            "error": getattr(action_result, "error", None)
        }

    def _update_state(self, decision: Dict, feedback: Dict):
        """
        更新阶段：记录历史信息
        """
        self.state.context["history"].append({
            "iteration": self.state.iteration,
            "decision": decision,
            "feedback": feedback
        })

    def _check_termination(self, decision: Dict, feedback: Dict) -> bool:
        """
        判断是否应该终止循环
        """
        # 任务已完成
        if decision.get("action") == "final_answer":
            return True
        # 执行出错且无法恢复
        if feedback.get("error") and self._is_fatal_error(feedback):
            return True
        return False

    def _build_reasoning_prompt(self, observation: Dict) -> str:
        """构建推理 Prompt"""
        history_str = "\n".join([
            f"Step {h['iteration']}: {h['decision']}"
            for h in observation["history"][-5:]  # 最近 5 步
        ])
        return f"""当前任务：{observation['task']}
历史步骤:
{history_str}

当前是第 {observation['current_step'] + 1} 步。
请分析当前状态并决定下一步行动。
可用动作：{list(self.tools.keys()) + ['final_answer']}
"""

    def _parse_decision(self, response: str) -> Dict:
        """解析 LLM 返回的决策"""
        # 简化解析逻辑
        return {"action": "final_answer", "args": {"answer": response}}

    def _is_fatal_error(self, feedback: Dict) -> bool:
        """判断是否为致命错误"""
        return False  # 简化处理

    def _finalize(self) -> Any:
        """循环结束，返回最终结果"""
        return self.state.result
```

---

## 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **任务完成率** | > 85% | 标准测试集 | 成功完成的任务比例 |
| **平均循环次数** | 3-8 次/任务 | 执行日志统计 | 完成任务所需的平均循环数 |
| **单次循环延迟** | < 2s | 端到端基准测试 | 单次循环的平均耗时 |
| **超时率** | < 5% | 压力测试 | 触发超时终止的任务比例 |
| **错误恢复率** | > 70% | 错误注入测试 | 从错误中恢复并完成任务的比例 |
| **Token 效率** | > 80% | Token 用量/任务质量 | 有效使用 Token 的比例 |

---

## 6. 扩展性与安全性

### 水平扩展

- **并行循环**：多个独立任务并发执行，共享工具池
- **分布式状态**：使用 Redis 等存储循环状态，支持故障转移
- **负载均衡**：根据任务复杂度动态分配计算资源

### 垂直扩展

- **循环优化**：智能预判减少不必要的循环次数
- **缓存机制**：对重复查询结果进行缓存
- **批量执行**：合并相似动作，减少 LLM 调用

### 安全考量

| 风险 | 防护措施 |
|------|---------|
| **无限循环** | 最大循环次数限制、超时机制 |
| **资源耗尽** | Token 配额管理、速率限制 |
| **危险动作** | 动作白名单、执行前审查 |
| **状态污染** | 循环间状态隔离、定期清理 |
| **敏感泄露** | 输出过滤、记忆脱敏 |
