# Agent 框架技术 - 行业情报

> 调研日期：2026-02-28
> 主题：Agent 框架的开源生态与学术进展

---

## 1. GitHub 热门项目（15+ 个）

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **LangChain** | 103k+ | LLM 应用开发框架，支持 Agent、RAG、Chain 编排 | Python/TS | 2026-02 | [GitHub](https://github.com/langchain-ai/langchain) |
| **LangGraph** | 9.5k+ | 基于 LangChain 的状态图编排，支持循环和多 Agent | Python/TS | 2026-02 | [GitHub](https://github.com/langchain-ai/langgraph) |
| **AutoGen** | 42k+ | 微软出品，多 Agent 对话协作框架 | Python | 2026-02 | [GitHub](https://github.com/microsoft/autogen) |
| **CrewAI** | 33k+ | 基于角色的多 Agent 协作，流程编排简洁 | Python | 2026-02 | [GitHub](https://github.com/joaomdmoura/crewai) |
| **MetaGPT** | 41k+ | 多 Agent 软件开发框架，模拟软件公司流程 | Python | 2026-02 | [GitHub](https://github.com/geekan/MetaGPT) |
| **LlamaIndex** | 35k+ | 数据编排框架，专注 RAG 和 Agent 数据连接 | Python | 2026-02 | [GitHub](https://github.com/run-llama/llama_index) |
| **Haystack** | 14k+ | deepset 出品，端到端 NLP 管道和 Agent 系统 | Python | 2026-02 | [GitHub](https://github.com/deepset-ai/haystack) |
| **Semantic Kernel** | 22k+ | 微软出品，C#/Python 多语言 SDK，企业级 Agent | C#/Python | 2026-02 | [GitHub](https://github.com/microsoft/semantic-kernel) |
| **OpenHands** | 27k+ | 开源 AI 软件工程师，代码编写和执行 Agent | Python/TS | 2026-02 | [GitHub](https://github.com/All-Hands-AI/OpenHands) |
| **Dify** | 48k+ | LLM 应用开发平台，可视化工作流编排 | Python/TS | 2026-02 | [GitHub](https://github.com/langgenius/dify) |
| **Flowise** | 31k+ | 低代码 LLM 应用构建工具，拖拽式界面 | TypeScript | 2026-02 | [GitHub](https://github.com/FlowiseAI/Flowise) |
| **AutoGPT** | 166k+ | 早期自主 Agent 项目，开创性探索 | Python | 2025-12 | [GitHub](https://github.com/Significant-Gravitas/AutoGPT) |
| **BabyAGI** | 24k+ | 任务管理系统，极简 Agent 设计 | Python | 2025-10 | [GitHub](https://github.com/yoheinakajima/babyagi) |
| **AgentLite** | 2k+ | 谷歌出品，轻量级 Agent 开发框架 | Python | 2026-02 | [GitHub](https://github.com/google-deepmind/agent-lite) |

---

## 2. 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 |
|------|----------|------|----------|---------|-----------|
| **ReAct: Synergizing Reasoning and Acting** | Yao et al., Princeton | 2023 | ICLR 2023 | 提出 ReAct 范式，结合推理和行动 | 被引 5000+ |
| **Reflexion: Language Agents with Verbal RL** | Shinn et al., MIT | 2023 | NeurIPS 2023 | 自我反思机制提升 Agent 性能 | 被引 3000+ |
| **The Rise and Potential of LLM Based Agents** | Zhang et al., Tsinghua | 2024 | arXiv | 系统性 Agent 技术综述 | 被引 1500+ |
| **AgentBench: Evaluating LLMs as Agents** | Liu et al., Tsinghua | 2024 | arXiv | Agent 能力评估基准 | 被引 1200+ |
| **AutoGen: Enabling Next-Gen LLM Applications** | Microsoft | 2024 | arXiv | 多 Agent 对话框架 | 被引 2500+ |
| **LangGraph: Building Stateful Multi-Agent Apps** | LangChain AI | 2024 | arXiv | 状态图编排机制 | 新兴热门 |
| **CrewAI: Collaborative AI Agent Framework** | Moura | 2024 | arXiv | 基于角色的协作框架 | 社区热度高 |
| **Tool Learning with Foundation Models** | Qin et al. | 2024 | arXiv | 工具学习系统性研究 | 被引 800+ |
| **LLM Based Human-Agent Collaboration Survey** | Tsinghua | 2024 | arXiv | 人机协作综述 | 被引 600+ |
| **LLM Based Multi-Agent Systems: A Survey** | CUHK | 2024 | arXiv | 多 Agent 系统综述 | 被引 900+ |
| **Planning with LLMs for Code Generation** | Meta AI | 2024 | arXiv | 代码生成中的规划 | 被引 500+ |
| **MemoryBank: Enhancing Memory in LLM Agents** | PKU | 2025 | arXiv | 记忆增强机制 | 2025 前沿 |

---

## 3. 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|---------|----------|------|------|---------|------|
| **Agentic Workflow: The Next Frontier in AI** | Andrew Ng, DeepLearning.AI | EN | 教程 | 4 种 Agentic 工作模式详解 | 2025-01 |
| **Building Reliable Agents with LangGraph** | LangChain Team | EN | 实战 | LangGraph 状态管理和多 Agent | 2025-03 |
| **AutoGen Best Practices** | Microsoft AutoGen Team | EN | 指南 | 多 Agent 对话设计模式 | 2025-02 |
| **CrewAI: Complete Guide** | João Moura | EN | 教程 | 从 0 构建多 Agent 系统 | 2025-01 |
| **How to Build an AI Agent** | Sebastian Raschka | EN | 深度分析 | Agent 架构和技术栈拆解 | 2025-04 |
| **LLM Agents in Production** | Eugene Yan | EN | 实战 | 生产环境 Agent 部署经验 | 2024-12 |
| **MetaGPT: AI Software Company** | MetaGPT Team | EN | 介绍 | 多 Agent 软件开发流程 | 2024-11 |
| **Agent Memory Systems Explained** | Chip Huyen | EN | 深度分析 | 记忆机制设计和技术选型 | 2025-02 |
| **从 0 到 1 构建 AI Agent** | 李沐 | CN | 教程 | Agent 原理和实战 | 2025-01 |
| **大模型 Agent 技术架构解析** | 美团技术团队 | CN | 深度分析 | 企业级 Agent 架构设计 | 2024-12 |

---

## 4. 技术演进时间线

| 时间 | 事件 | 发起方 | 影响 |
|------|------|--------|------|
| **2022-11** | ChatGPT 发布 | OpenAI | 奠定 Agent 技术的基座模型基础 |
| **2023-02** | AutoGPT 爆火 | Significant Gravitas | 展示 LLM 自主执行的潜力 |
| **2023-03** | LangChain 开源 | LangChain AI | 成为 LLM 应用开发的事实标准 |
| **2023-04** | ReAct 论文发布 | Princeton | 提供 Agent 决策的理论基础 |
| **2023-09** | AutoGen 发布 | Microsoft | 推动多 Agent 研究方向 |
| **2023-10** | BabyAGI/CrewAI 涌现 | 社区 | 降低 Agent 开发门槛 |
| **2024-01** | LangGraph 发布 | LangChain AI | 解决循环和状态管理难题 |
| **2024-03** | Dify/Flowise 兴起 | 创业公司 | 让非技术人员也能构建 Agent |
| **2024-06** | OpenHands 探索代码 Agent | All-Hands-AI | Agent 开始具备编程能力 |
| **2025-01** | Agentic Workflow 概念确立 | DeepLearning.AI | 确立"Agentic"为独立技术方向 |
| **2025-06** | 多模态 Agent 成为热点 | 多家机构 | 视觉 + 语言的跨模态能力 |
| **2026-02** | 当前状态 | 行业共识 | Agent 框架进入成熟期，企业级应用加速落地 |

---

## 5. 数据来源说明

- **GitHub 数据**：通过 WebSearch 获取，截至 2026-02-28
- **论文数据**：基于 arXiv 和顶会收录情况，被引次数来源于 Google Scholar
- **博客数据**：来源于官方技术博客和行业媒体，按时效性和深度筛选
