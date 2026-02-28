# Multi Agent 技术 - 方案对比

> 调研日期：2026-02-28
> 主题：Multi Agent 的主流方案横向评估

---

## 1. 历史发展时间线

```
1980s ─┬─ 经典 MAS 研究 → 分布式 AI 理论基础
2000s ─┼─ MARL 兴起 → 多 Agent 强化学习
2023 ──┼─ AutoGen 发布 → LLM 多 Agent 框架里程碑
2024 ──┼─ MetaGPT/CrewAI → 垂直领域/角色协作
2025 ──┼─ Debate/Society → 辩论/社会模拟新方向
2026 ──┴─ 当前状态：企业级应用加速落地
```

---

## 2. 五种主流方案横向对比

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **AutoGen** | 多 Agent 对话协作，基于消息传递 | - 多 Agent 原生支持<br>- 对话式编程简洁<br>- 微软背书<br>- 灵活可扩展 | - 调试困难<br>- 文档分散<br>- 单 Agent 场景过重 | 多 Agent 协作、研究实验 | 中 |
| **CrewAI** | 基于角色（Role）的任务分配和流程编排 | - API 简洁易用<br>- 角色抽象清晰<br>- 流程可视化<br>- 快速上手 | - 功能相对单一<br>- 扩展性有限<br>- 生态较小 | 中小型多 Agent 项目、快速原型 | 低 - 中 |
| **MetaGPT** | 模拟软件公司流程的多 Agent 框架 | - 领域专用（软件开发）<br>- SOP 标准化流程<br>- 多角色协作 | - 垂直领域局限<br>- 定制成本高<br>- 学习成本 | 自动化软件开发、代码生成 | 中 |
| **LangGraph** | Chain + State Graph 编排，支持循环和多 Agent | - 生态最成熟<br>- 组件丰富<br>- 支持状态持久化<br>- 社区活跃 | - 学习曲线陡峭<br>- 代码冗长<br>- 早期版本抽象泄露 | 复杂 Agent 工作流、企业级应用 | 中 - 高 |
| **AgentScope** | 阿里出品，支持游戏/应用的多 Agent 平台 | - 多模态支持<br>- 游戏/仿真优化<br>- 中文支持好<br>- 阿里背书 | - 国内生态局限<br>- 国际社区小<br>- 文档以中文为主 | 游戏开发、仿真模拟、中文场景 | 中 |

---

## 3. 技术细节对比

| 维度 | AutoGen | CrewAI | MetaGPT | LangGraph | AgentScope |
|------|---------|--------|---------|-----------|------------|
| **性能** | 中等，消息传递有延迟 | 较高，轻量级 | 中等，SOP 流程固定 | 中等，状态图有开销 | 高，C++ 后端优化 |
| **易用性** | 较高，对话式 API | 高，角色抽象直观 | 中等，需理解 SOP | 中等，需理解状态机 | 中等，配置复杂 |
| **生态成熟度** | 高，微软支持 | 中，快速增长 | 中，垂直领域 | 高，LangChain 生态 | 中，阿里生态 |
| **社区活跃度** | 高 | 高 | 中 | 非常高 | 中（国内高） |
| **学习曲线** | 中等 | 平缓 | 中等 | 陡峭 | 中等 |
| **可调试性** | 较低，消息追踪难 | 高，流程清晰 | 中等 | 中等，有可视化 | 高，工具有支持 |
| **多模态支持** | 支持 | 有限 | 有限 | 支持，依赖 LLM | 支持，原生优化 |
| **中文支持** | 一般 | 一般 | 一般 | 一般 | 优秀 |

---

## 4. 各方案核心特性详解

### 4.1 AutoGen

**核心抽象**：对话（Conversation）

```python
from autogen import AssistantAgent, UserProxyAgent

llm_config = {"config_list": [...]}
assistant = AssistantAgent("assistant", llm_config=llm_config)
user_proxy = UserProxyAgent("user_proxy", code_execution_config=True)

# 发起对话
user_proxy.initiate_chat(assistant, message="Build a tic-tac-toe game")
```

**适用场景**：
- 多 Agent 协作研究
- 代码生成和执行
- 需要灵活对话模式的场景

---

### 4.2 CrewAI

**核心抽象**：角色（Role）+ 任务（Task）

```python
from crewai import Agent, Task, Crew

researcher = Agent(role='Researcher', goal='Find answers', backstory=...)
writer = Agent(role='Writer', goal='Write content', backstory=...)

task1 = Task(description='Research topic', agent=researcher)
task2 = Task(description='Write article', agent=writer)

crew = Crew(agents=[researcher, writer], tasks=[task1, task2])
result = crew.kickoff()
```

**适用场景**：
- 快速构建多 Agent 系统
- 角色分工明确的协作任务
- 非技术背景用户

---

### 4.3 MetaGPT

**核心抽象**：SOP（标准作业程序）

```python
from metagpt.roles import ProjectManager, Engineer
from metagpt.team import Team

team = Team()
team.hire([ProjectManager(), Engineer()])
team.run_project("Build a snake game")
```

**适用场景**：
- 自动化软件开发
- 需要标准化流程的场景
- 代码生成项目

---

### 4.4 LangGraph

**核心抽象**：状态图（State Graph）

```python
from langgraph.graph import StateGraph, END

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 构建图
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()
```

**适用场景**：
- 需要循环执行的工作流
- 复杂条件分支的编排
- 需要状态持久化的长周期任务

---

### 4.5 AgentScope

**核心抽象**：演员（Actor）模型

```python
import agentscope
from agentscope.agents import Agent

# 初始化
agentscope.init(model_configs=...)

# 创建 Agent
agent1 = Agent(name="Alice")
agent2 = Agent(name="Bob")

# 消息传递
response = agent1.reply("Hello!")
```

**适用场景**：
- 游戏开发和仿真
- 多模态交互
- 中文场景应用

---

## 5. 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **研究实验** | AutoGen | 灵活性高，多 Agent 对话机制完善 | $200-1000 |
| **快速原型** | CrewAI | API 简洁，快速上手，1 天可出原型 | $50-200 |
| **软件开发自动化** | MetaGPT | 领域专用，SOP 流程成熟 | $500-2000 |
| **企业级工作流** | LangGraph | 生态成熟，可维护性强，支持迭代 | $500-2000 |
| **游戏/仿真** | AgentScope | 多模态优化，中文支持好 | $200-1000 |
| **大规模社会模拟** | AgentScope + 自研 | 支持万级 Agent 仿真 | $2000-10000+ |

---

## 6. 选型决策树

```
                    ┌─────────────────┐
                    │  你的需求是什么？ │
                    └────────┬────────┘
                             │
        ┌────────────┬───────┴───────┬────────────┐
        ↓            ↓               ↓            ↓
   ┌────────┐  ┌────────┐    ┌──────────┐  ┌────────┐
   │ 快速原型│  │软件开发│    │企业级    │  │游戏/   │
   │        │  │        │    │工作流    │  │仿真    │
   └───┬────┘  └───┬────┘    └─────┬────┘  └───┬────┘
       ↓           ↓               ↓           ↓
   ┌───────┐  ┌──────────┐   ┌─────────┐  ┌────────┐
   │CrewAI │  │MetaGPT   │   │LangGraph│  │Agent   │
   │       │  │          │   │         │  │Scope   │
   └───────┘  └──────────┘   └─────────┘  └────────┘
```
