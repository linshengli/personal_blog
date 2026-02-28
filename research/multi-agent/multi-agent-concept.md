# Multi Agent 技术 - 概念剖析

> 调研日期：2026-02-28
> 主题：Multi Agent 技术的核心原理与架构

---

## 1. 定义澄清

### 通行定义

**Multi Agent 系统**（Multi-Agent System, MAS）是由多个相互独立的智能体（Agent）组成的分布式系统，这些 Agent 通过通信、协作、竞争或协商等方式交互，共同完成单个 Agent 难以解决的复杂任务。

在 LLM 时代，Multi Agent 技术特指基于大语言模型的多 Agent 协作系统，每个 Agent 具备独立的角色、目标和能力，通过自然语言进行交互和协调。

### 常见误解

| 误解 | 正解 |
|------|------|
| "Multi Agent 就是多个 LLM 实例" | Multi Agent 强调 Agent 间的交互协议、协作机制和涌现行为 |
| "Agent 越多效果越好" | Agent 数量增加会带来通信开销和协调困难，存在最优规模 |
| "Multi Agent 只能用于对话" | 可用于软件开发、数据分析、创意创作、决策支持等多种场景 |
| "Multi Agent 系统完全自主" | 大多数系统仍需人工监督和干预，尤其是关键决策 |

### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **Multi Agent vs Single Agent** | Single Agent 独立完成任务；Multi Agent 通过协作涌现更强能力 |
| **Multi Agent vs 分布式系统** | 分布式系统强调计算资源分布；Multi Agent 强调智能体自主性和交互 |
| **Multi Agent vs 群体智能** | 群体智能强调简单个体的涌现行为；Multi Agent 允许复杂 Agent 协作 |

---

## 2. 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                 Multi Agent 系统架构                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  Agent 1    │ ───→│  通信层     │←─── │  Agent 2    │    │
│  │  (角色 A)   │     │  (消息传递)  │     │  (角色 B)   │    │
│  └─────────────┘     └──────┬──────┘     └─────────────┘    │
│         ↑                    ↓                    ↑          │
│         │              ┌─────────────┐            │          │
│         │              │  协调层     │            │          │
│         │              │ (任务分配)  │            │          │
│         │              └──────┬──────┘            │          │
│         │                     ↓                   │          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  Agent 3    │ ───→│  共享状态   │←─── │  Agent N    │    │
│  │  (角色 C)   │     │  (记忆/知识) │     │  (角色 N)   │    │
│  └─────────────┘     └─────────────┘     └─────────────┘    │
│                                                              │
│                    ┌─────────────┐                           │
│                    │  监督层     │                           │
│                    │ (人工/Human) │                          │
│                    └─────────────┘                           │
└──────────────────────────────────────────────────────────────┘
```

**组件职责说明：**

| 组件 | 职责 |
|------|------|
| **Agent 节点** | 独立智能体，具备角色、目标、记忆和工具使用能力 |
| **通信层** | 管理 Agent 间消息传递、格式转换、路由 |
| **协调层** | 任务分配、冲突解决、同步/异步调度 |
| **共享状态** | 公共记忆、知识库、任务进度追踪 |
| **监督层** | 人工监督、关键决策审批、异常干预 |

---

## 3. 数学形式化

### 3.1 多 Agent 协作收益

对于 $n$ 个 Agent 协作完成的任务 $T$，协作收益：

$$\text{Gain}(n) = \frac{\text{Performance}_{\text{multi}}(n) - \text{Performance}_{\text{single}}}{\text{CommunicationCost}(n)}$$

**自然语言解释**：协作收益等于多 Agent 性能提升除以通信开销。

### 3.2 通信复杂度

对于全连接通信和中心化协调两种模式：

- **全连接**：$O(n^2)$ 通信链路
- **中心化**：$O(n)$ 通信链路

**自然语言解释**：中心化架构通信开销随 Agent 数量线性增长，全连接呈平方增长。

### 3.3 任务分配优化

任务 $T$ 分配给 $n$ 个 Agent 的最优解：

$$\min \sum_{i=1}^{n} \text{Cost}(A_i, T_i) \quad \text{s.t.} \quad \bigcup_{i=1}^{n} T_i = T$$

**自然语言解释**：在任务完整覆盖的约束下，最小化总执行成本。

### 3.4 共识达成概率

对于 $n$ 个 Agent 投票达成决策，共识概率：

$$P_{\text{consensus}} = \sum_{k=\lceil n/2 \rceil}^{n} \binom{n}{k} p^k (1-p)^{n-k}$$

其中 $p$ 为单个 Agent 正确决策概率。

**自然语言解释**：多数投票的共识概率遵循二项分布累积。

---

## 4. 实现逻辑（Python 伪代码）

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import asyncio

class MessageType(Enum):
    TASK = "task"
    RESPONSE = "response"
    COORDINATION = "coordination"
    BROADCAST = "broadcast"

@dataclass
class Message:
    """Agent 间通信消息"""
    sender: str
    receiver: str
    msg_type: MessageType
    content: Any
    timestamp: float

class Agent:
    """单个 Agent 类"""

    def __init__(self, name: str, role: str, llm=None):
        self.name = name
        self.role = role
        self.llm = llm
        self.mailbox: List[Message] = []
        self.state: Dict[str, Any] = {}

    async def receive(self, message: Message):
        """接收消息"""
        self.mailbox.append(message)

    async def process(self) -> Optional[Message]:
        """处理消息并生成响应"""
        if not self.mailbox:
            return None
        # 处理逻辑
        pass

class CommunicationLayer:
    """通信层，管理消息传递"""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()

    def register(self, agent: Agent):
        """注册 Agent"""
        self.agents[agent.name] = agent

    async def send(self, message: Message):
        """发送消息"""
        await self.message_queue.put(message)

    async def broadcast(self, sender: str, content: Any):
        """广播消息给所有 Agent"""
        for name, agent in self.agents.items():
            if name != sender:
                msg = Message(sender=sender, receiver=name,
                            msg_type=MessageType.BROADCAST,
                            content=content, timestamp=time.time())
                await self.send(msg)

    async def route(self):
        """消息路由"""
        while True:
            message = await self.message_queue.get()
            target = self.agents.get(message.receiver)
            if target:
                await target.receive(message)

class Coordinator:
    """协调器，管理任务分配和冲突解决"""

    def __init__(self, communication: CommunicationLayer):
        self.comm = communication
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.agent_workload: Dict[str, int] = {}

    async def assign_task(self, task: Any, target_agent: str):
        """分配任务给指定 Agent"""
        message = Message(
            sender="coordinator",
            receiver=target_agent,
            msg_type=MessageType.TASK,
            content=task,
            timestamp=time.time()
        )
        await self.comm.send(message)
        self.agent_workload[target_agent] = \
            self.agent_workload.get(target_agent, 0) + 1

    async def balance_load(self):
        """负载均衡：将任务分配给工作量最少的 Agent"""
        if not self.task_queue.empty():
            task = await self.task_queue.get()
            target = min(self.agent_workload,
                        key=self.agent_workload.get)
            await self.assign_task(task, target)

class MultiAgentSystem:
    """Multi Agent 系统核心类"""

    def __init__(self):
        self.comm_layer = CommunicationLayer()
        self.coordinator = Coordinator(self.comm_layer)
        self.shared_state: Dict[str, Any] = {}

    def add_agent(self, agent: Agent):
        """添加 Agent 到系统"""
        self.comm_layer.register(agent)

    async def run(self, initial_task: str) -> Any:
        """
        运行 Multi Agent 系统
        流程：任务分解 → 分配 → 执行 → 协作 → 汇总
        """
        # 阶段 1: 任务分解
        subtasks = await self._decompose_task(initial_task)

        # 阶段 2: 任务分配
        for i, task in enumerate(subtasks):
            agent_name = self._select_agent(task)
            await self.coordinator.assign_task(task, agent_name)

        # 阶段 3: 并发执行和协作
        results = await self._execute_concurrently()

        # 阶段 4: 结果汇总
        final_result = await self._aggregate_results(results)
        return final_result

    async def _decompose_task(self, task: str) -> List[str]:
        """使用 LLM 分解任务"""
        pass

    def _select_agent(self, task: str) -> str:
        """根据任务类型选择合适的 Agent"""
        pass

    async def _execute_concurrently(self) -> List[Any]:
        """并发执行所有子任务"""
        pass

    async def _aggregate_results(self, results: List[Any]) -> Any:
        """汇总所有结果"""
        pass
```

---

## 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **任务完成率** | > 90% | 标准测试集 | 成功完成的任务比例 |
| **协作效率增益** | > 30% | 对比 Single Agent | 多 Agent 相对单 Agent 的性能提升 |
| **通信开销占比** | < 20% | 总时间/通信时间 | 通信时间占总执行时间的比例 |
| **冲突解决率** | > 95% | 冲突场景测试 | 成功解决的冲突比例 |
| **共识达成时间** | < 5 轮 | 对话轮次统计 | 达成一致所需的平均轮次 |
| **系统可扩展性** | 线性至 50+Agent | 压力测试 | 性能随 Agent 数量的变化 |

---

## 6. 扩展性与安全性

### 水平扩展

- **动态 Agent 注册**：支持运行时添加/移除 Agent
- **分层协调**：多层次的协调器架构，减少单点压力
- **联邦学习**：Agent 间共享知识但不共享私有数据

### 垂直扩展

- **Agent 能力增强**：提升单个 Agent 的 LLM 能力和工具集
- **记忆优化**：向量数据库支持大规模长期记忆
- **并行执行**：异步并发执行多个子任务

### 安全考量

| 风险 | 防护措施 |
|------|---------|
| **恶意 Agent** | 身份验证、行为审计、隔离执行 |
| **信息泄露** | 消息加密、访问控制、最小权限原则 |
| **共识攻击** | 拜占庭容错、多轮验证、人工监督 |
| **资源耗尽** | 配额限制、优先级调度、超时机制 |
| **协调器故障** | 冗余备份、选举机制、故障转移 |
