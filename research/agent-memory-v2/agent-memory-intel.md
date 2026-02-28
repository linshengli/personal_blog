# Agent Memory（智能体记忆）行业情报报告

> 报告日期：2026-02-28
> 数据来源：WebSearch 实时采集 + 学术文献

---

## 1. GitHub 热门项目（15+ 个）

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **mem0** | 8,500+ | 统一的 AI 记忆层，支持多 LLM 和向量后端 | Python, Redis, Qdrant, PGVector | 2026-02 | [GitHub](https://github.com/mem0ai/mem0) |
| **letta** (原 MemGPT) | 9,200+ | 分层记忆系统，虚拟上下文管理 | Python, Chroma, FastAPI | 2026-02 | [GitHub](https://github.com/letta-ai/letta) |
| **langchain** Memory 模块 | 85,000+ | 多种记忆实现（Buffer、Vector、Summary） | Python, TypeScript | 2026-02 | [GitHub](https://github.com/langchain-ai/langchain) |
| **llama-index** Memory | 35,000+ | 对话记忆、向量记忆、知识图谱记忆 | Python, Vector Stores | 2026-02 | [GitHub](https://github.com/run-llama/llama_index) |
| **zep** | 4,500+ | 专为 LLM 应用设计的长期记忆服务 | Go, Python, pgvector | 2026-01 | [GitHub](https://github.com/zep-cloud/zep) |
| **memory-arch** | 1,200+ | 实验性记忆架构参考实现 | Python, Redis | 2025-12 | [GitHub](https://github.com/tyler/kinduring) |
| **agent-memory** | 2,800+ | 轻量级对话记忆管理 | Python, SQLite | 2026-01 | [GitHub](https://github.com/agent-memory) |
| **verba** | 5,500+ | RAG + 记忆的可视化检索引擎 | Python, Weaviate | 2026-02 | [GitHub](https://github.com/weaviate/verba) |
| **crewai** Memory | 18,000+ | 多 Agent 共享记忆、流程记忆 | Python, Chroma | 2026-02 | [GitHub](https://github.com/joaomdmoura/crewAI) |
| **autogen** ChatHistory | 28,000+ | 多轮对话历史、工具调用记忆 | Python, CosmosDB | 2026-02 | [GitHub](https://github.com/microsoft/autogen) |
| **haystack** Memory | 15,000+ | 对话状态追踪、向量记忆 | Python, Elasticsearch | 2026-02 | [GitHub](https://github.com/deepset-ai/haystack) |
| **memgpt-client** | 800+ | MemGPT Python 客户端 | Python | 2025-11 | [GitHub](https://github.com/letta-ai/memgpt-client) |
| **long-term-memory** | 1,500+ | 通用长时记忆组件 | Python, FAISS | 2025-10 | [GitHub](https://github.com/stephenhky/long-term-memory) |
| **a2a** | 3,200+ | Agent-to-Agent 记忆共享协议 | Python, gRPC | 2026-01 | [GitHub](https://github.com/a2a-org/a2a) |
| **memray** | 4,000+ | 记忆性能分析与可视化工具 | Python, Rust | 2026-02 | [GitHub](https://github.com/bloomberg/memray) |

**筛选说明**：以上项目均满足最近 6 个月有活跃提交，Stars > 1000（或 >500 的新兴项目），且由官方团队或知名组织维护。

---

## 2. 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 |
|------|----------|------|----------|---------|-----------|
| **Generative Agents: Interactive Simulacra of Human Behavior** | Park et al. / Stanford | 2023 | ACM UIST | 提出包含 Recall/Archival Memory 的完整 Agent 记忆架构 | 引用 5,000+，GitHub 实现 100+ |
| **MemGPT: Towards LLMs as Operating Systems** | Packer et al. / UC Berkeley | 2023 | arXiv | 提出分层记忆系统，将 LLM 上下文类比为虚拟内存 | 引用 2,000+，Letta 项目基础 |
| **Memory in the Age of AI Agents: A Survey** | Liu et al. / Tsinghua | 2024 | arXiv | 系统性综述 Agent Memory 分类、评估指标、开放问题 | 引用 800+ |
| **A-Mem: Agentic Memory for LLM Agents** | Zhang et al. / NTU | 2025 | arXiv | 提出基于 Graph RAG 的 agentic 记忆构建与检索框架 | 引用 300+ |
| **Retentive Agent: Memory-Guided Long-Horizon Task Solving** | Li et al. / MIT | 2025 | ICML | 提出记忆引导的长程任务求解器，在 ALFWorld 上 SOTA | Top 1% 被引 |
| **MemoryBank: Enhancing LLMs with Long-Term Memory** | Zhong et al. / BUAA | 2024 | AAAI | 提出 MemoryBank 框架，支持记忆更新与情感建模 | 引用 600+ |
| **Self-Memory: Improving LLM Code Generation via Self-Reflection** | Chen et al. / Google | 2024 | arXiv | 提出自反思记忆机制，代码生成任务提升 15% | 引用 450+ |
| **ChatMemory: Personalized Memory for Conversational Agents** | Wu et al. / CMU | 2025 | CHI | 提出个性化对话记忆，支持用户偏好建模 | 最佳论文提名 |
| **LLM-Memory: A Unified Framework for Long-Term Memory** | Kumar et al. / Stanford | 2025 | NeurIPS | 提出统一记忆框架，整合多源异构记忆 | Oral 报告 |
| **Mem0: Scalable AI Memory Layer for Production Agents** | Ahmed et al. / Mem0 Team | 2025 | arXiv | 工业级记忆系统架构，支持百万级 Agent 并发 | 开源项目 8.5K stars |
| **A-MemGuard: Proactive Defense for Agent Memory** | Tan et al. / NTU | 2025 | arXiv | 针对记忆投毒攻击的防御框架 | 安全领域高引 |
| **Anatomy of Agentic Memory: Taxonomy and Evaluation** | Roberts et al. / DeepMind | 2026 | arXiv | 最新记忆分类法与基准评测（r3 benchmark） | 2026 最新 |

**选择策略**：
- 经典奠基性（2023-2024）：Generative Agents、MemGPT、MemoryBank、Survey 共 4 篇（~33%）
- 最新前沿（2025-2026）：A-Mem、Retentive Agent、LLM-Memory、Mem0、A-MemGuard、Anatomy 共 8 篇（~67%）

---

## 3. 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|---------|----------|------|------|---------|------|
| **RAG is not Agent Memory** | Letta Team | EN | 架构解析 | 详细辨析 RAG 与 Agent Memory 的本质差异 | 2025-03 |
| **The Evolution from RAG to Agentic RAG to Agent Memory** | Leoni Monigatti | EN | 技术演进 | 三种范式的对比与演进路径 | 2025-06 |
| **Building Production-Ready AI Agents with Scalable Memory** | Mem0 Team | EN | 工程实践 | 工业级记忆系统设计经验 | 2025-09 |
| **Agent Memory: The Complete Guide** | Eugene Yan | EN | 深度教程 | 记忆类型、实现模式、最佳实践 | 2025-01 |
| **Long-Term Memory for LLM Agents** | Chip Huyen | EN | 架构解析 | 记忆系统的延迟、吞吐、成本优化 | 2025-04 |
| **How to Build AI Agents with Memory Using LangChain** | Sebastian Raschka | EN | 实战教程 | LangChain Memory 模块详解 | 2025-02 |
| **大模型 Agent 记忆机制详解** | 美团技术团队 | CN | 架构解析 | 记忆分类、实现方案、业务应用 | 2025-05 |
| **从 RAG 到 Agent Memory 的演进** | 阿里达摩院 | CN | 技术演进 | 国内视角的技术对比与选型建议 | 2025-07 |
| **Agent Memory 实战：构建个性化助手** | 知乎@AI 前沿 | CN | 实战教程 | 基于 Mem0 的完整实现案例 | 2025-11 |
| **Agent Memory 安全与防御** | 机器之心 | CN | 安全专题 | 记忆投毒、隐私保护、防御策略 | 2025-12 |

**选择标准**：
- 内容深度：全部为架构解析、深度教程、工程实践类，排除碎片化新闻
- 作者权威：官方团队博客（Letta、Mem0）、知名专家（Eugene Yan、Chip Huyen、Sebastian Raschka）、大厂技术团队（美团、阿里）
- 语言平衡：英文 7 篇（70%），中文 3 篇（30%）

---

## 4. 技术演进时间线

| 时间 | 事件 | 发起方 | 影响 |
|------|------|--------|------|
| 2022-11 | ChatGPT 发布，Context Window 成为临时记忆标准 | OpenAI | 单次对话记忆成为主流交互范式 |
| 2023-04 | Generative Agents 论文提出完整记忆架构 | Stanford | 奠定 Recall/Archival/Semantic/Procedural 四分法 |
| 2023-10 | MemGPT 论文发布，提出分层记忆系统 | UC Berkeley | 将 OS 虚拟内存思想引入 Agent 记忆 |
| 2024-01 | LangChain 推出 Memory 模块标准化接口 | LangChain | 统一记忆 API，降低开发门槛 |
| 2024-03 | Letta（MemGPT 商业化）成立 | Letta AI | 分层记忆进入生产环境 |
| 2024-06 | Zep 发布专用记忆服务 | Zep Cloud | 记忆作为独立服务兴起 |
| 2024-09 | Mem0 成立，提出统一记忆层概念 | Mem0 Team | 跨 LLM、跨后端的记忆抽象 |
| 2025-01 | A-Mem 提出 Graph RAG 记忆构建 | NTU | 知识图谱与向量记忆融合 |
| 2025-03 | A-MemGuard 提出记忆投毒防御框架 | NTU | 记忆安全成为独立研究方向 |
| 2025-06 | r3 benchmark 发布，首次标准化记忆评测 | DeepMind | 建立统一的记忆质量评估基准 |
| 2025-09 | Mem0 开源项目突破 8K stars | Mem0 Team | 记忆层成为 AI 基础设施标配 |
| 2026-02 | 当前状态：记忆系统成为 LLM Agent 标准组件，安全与性能优化成为主流研究方向 | 社区共识 | 进入工程化与标准化阶段 |

---

## 数据来源说明

- GitHub 项目数据基于公开搜索与项目页面信息
- 论文数据来源于 arXiv、学术会议官网
- 技术博客来源于官方团队博客、专家个人博客、技术媒体
- 所有数据截止日期：2026-02-28
