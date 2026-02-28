# Agent Memory 行业情报

> 数据采集日期：2026-02-28 | 来源：GitHub、arXiv、技术博客

---

## 1. GitHub 热门项目

| 项目名 | Stars (估) | 核心功能 | 技术栈 | 最后更新 | 链接 |
|--------|-----------|---------|--------|---------|------|
| **mem0** | ~25K | 通用记忆层，向量+图混合存储 | Python, OpenAI, Qdrant | 2025-02 活跃 | [GitHub](https://github.com/mem0ai/mem0) |
| **letta** (原 MemGPT) | ~14K | OS 式分层记忆管理，有状态 Agent 平台 | Python, SQLite | 2025-02 活跃 | [GitHub](https://github.com/letta-ai/letta) |
| **zep** | ~3K | 时序知识图谱架构，会话记忆 | Go/Python, Neo4j | 2025-01 活跃 | [GitHub](https://github.com/getzep/zep) |
| **graphiti** (Zep 子项目) | ~3K | 图谱记忆引擎，BM25+语义混合检索 | Python, Neo4j | 2025-02 活跃 | [GitHub](https://github.com/getzep/graphiti) |
| **cognee** | ~2K | 认知记忆框架，强调分块与检索质量 | Python, 多向量库 | 2025-02 活跃 | [GitHub](https://github.com/topoteretes/cognee) |
| **MemoryOS** (BAI-LAB) | ~1K | 个性化 Agent 记忆操作系统 | Python | EMNLP 2025 Oral | [GitHub](https://github.com/BAI-LAB/MemoryOS) |
| **MemOS** (MemTensor) | ~800 | 技能记忆 OS，跨任务技能复用 | Python, Redis | 2025-02 活跃 | [GitHub](https://github.com/MemTensor/MemOS) |
| **LangChain Memory** | (LangChain 整体 ~100K) | 对话缓冲、摘要、向量、实体记忆 | Python/JS | 持续更新 | [Docs](https://python.langchain.com/docs/modules/memory/) |
| **LlamaIndex Memory** | (LlamaIndex 整体 ~40K) | ChatMemoryBuffer、向量记忆 | Python | 持续更新 | [Docs](https://docs.llamaindex.ai/) |
| **EverMemOS** | ~500 | 长期记忆系统，LoCoMo SOTA | Python | 2025-01 | [evermind.ai](https://evermind.ai) |
| **MemMachine** | ~300 | 高效长期记忆，LoCoMo 高分 | Python | 2025-12 | [memmachine.ai](https://memmachine.ai) |
| **Supermemory** | ~800 | Agent 记忆 SOTA 竞争者 | Python | 2025 活跃 | [supermemory.ai](https://supermemory.ai) |
| **LangMem** (LangChain) | ~600 | LangGraph 长期记忆管理 | Python | 2025 活跃 | [GitHub](https://github.com/langchain-ai/langmem) |
| **Memobase** | ~500 | 结构化用户画像记忆 | Python | 2025 活跃 | [GitHub](https://github.com/memobase-io/memobase) |
| **Awesome-Agent-Memory** | ~400 | 论文/项目策展列表 | Markdown | 2025-02 | [GitHub](https://github.com/TeleAI-UAGI/Awesome-Agent-Memory) |
| **Agent-Memory-Paper-List** | ~300 | "Memory in the Age of AI Agents" 论文列表 | Markdown | 2025-12 | [GitHub](https://github.com/Shichun-Liu/Agent-Memory-Paper-List) |

---

## 2. 关键论文（12 篇）

### 经典高影响力（奠基性工作）— 约 40%

| 论文标题 | 作者/机构 | 年份 | 会议 | 核心贡献 | 影响力 |
|---------|----------|------|------|---------|-------|
| Generative Agents: Interactive Simulacra of Human Behavior | Park et al. / Stanford | 2023 | UIST 2023 | 首次实现"记忆流+反思+规划"三位一体架构，25 个虚拟居民的社会模拟 | 引用 3000+，奠基之作 |
| MemGPT: Towards LLMs as Operating Systems | Packer et al. / UC Berkeley | 2023 | ICLR 2024 | 类 OS 虚拟内存分页管理，让 LLM 自主管理上下文窗口 | 引用 500+，催生 Letta 平台 |
| Reflexion: Language Agents with Verbal Reinforcement Learning | Shinn et al. | 2023 | NeurIPS 2023 | 语言反思作为记忆的一种形式，Agent 从失败经验中学习 | 引用 1000+ |
| Cognitive Architectures for Language Agents (CoALA) | Sumers et al. | 2023 | arXiv | 提出 Agent 认知架构统一框架，系统定义记忆模块分类 | 引用 400+，理论框架 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis et al. / Meta | 2020 | NeurIPS 2020 | RAG 奠基论文，外部记忆检索增强的起点 | 引用 8000+，范式开创 |

### 最新 SOTA 论文（前沿进展）— 约 60%

| 论文标题 | 作者/机构 | 年份 | 会议 | 核心贡献 | 影响力 |
|---------|----------|------|------|---------|-------|
| Memory in the Age of AI Agents: A Survey | Liu et al. / 清华 | 2025 | arXiv (2512.13564) | 最全面的 Agent Memory 综述，提出分类体系 | 近期最重要综述 |
| A-MEM: Agentic Memory for LLM Agents | Xu et al. | 2025 | arXiv (2502.12110) | Zettelkasten 方法的动态索引和链接，自组织记忆网络 | 高关注度 |
| Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory | Mem0 Team | 2025 | arXiv (2504.19413) | 工业级记忆系统设计，向量+图混合架构 | 产业化标杆 |
| Zep: A Temporal Knowledge Graph Architecture for Agent Memory | Rasmussen et al. / Zep | 2025 | arXiv | 时序知识图谱用于 Agent 记忆，事实抽取+时间推理 | 图谱记忆方向标杆 |
| MemoryOS: Personalized AI Agent Memory Operating System | BAI-LAB | 2025 | EMNLP 2025 Oral | 三层记忆架构（短期/中期/长期），个性化记忆管理 | 顶会 Oral |
| Agentic Memory: Learning Unified Long-Term and Short-Term Memory Management | — | 2026 | arXiv (2601.01885) | RL 驱动的统一记忆管理，语言生成与记忆操作联合训练 | 最新前沿 |
| H-MEM: Hierarchical Memory for High-Efficiency | — | 2025 | arXiv (2507.22925) | 四层层次记忆（Domain→Category→Trace→Episode） | 层次化方向 |

---

## 3. 系统化技术博客（10 篇）

| 标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|-----|----------|------|------|---------|------|
| Benchmarking AI Agent Memory: Is a Filesystem All You Need? | Letta Blog | EN | 评测分析 | 简单文件系统 Agent 在 LoCoMo 上达 74%，质疑专用记忆工具价值 | 2025 |
| Survey of AI Agent Memory Frameworks | Graphlit Blog | EN | 综述 | 系统对比 Mem0/Letta/Zep/Cognee 等主流框架 | 2025 |
| Is Mem0 Really SOTA in Agent Memory? | Zep Blog | EN | 评测争议 | 揭示 Mem0 评测方法论问题，全上下文基线反超专用记忆 | 2025 |
| AI Memory Tools Evaluation — Cognee, Mem0, Zep/Graphiti | Cognee Blog | EN | 对比评测 | 三大框架在分块、检索、准确率上的详细对比 | 2025 |
| Unlocking Memory in Agentic AI: 3 Open-Source Frameworks | DEV Community | EN | 入门教程 | Mem0/Letta/Zep 快速上手指南 | 2025 |
| Graph Memory for AI Agents | Mem0 Blog | EN | 技术深度 | 图记忆的设计理念、实体抽取、关系推理 | 2026-01 |
| Beyond Vector Databases: Architectures for True Long-Term AI Memory | Abhishek Jain / Medium | EN | 架构分析 | 超越向量库的记忆架构设计 | 2025 |
| 2025 AI 记忆系统大横评 | 知乎/人人都是产品经理 | CN | 横评 | 从插件到操作系统的范式演进，EverMemOS 等项目分析 | 2025 |
| 大模型 Agent 长记忆机制研究综述 (2024-2025) | CSDN | CN | 学术综述 | 记忆表示、检索与更新的技术挑战和方案 | 2025 |
| 万字解析 Agent Memory 实现 | 知乎 | CN | 深度教程 | 完整实现解析，覆盖存储/检索/压缩全链路 | 2025 |

---

## 4. 技术演进时间线

```
2020 ─┬─ RAG 论文发表 (Lewis et al.)
      │  → 确立"外部记忆+检索增强"范式，成为后续 Agent Memory 的技术基础
      │
2022 ─┼─ ChatGPT 发布，LLM Agent 概念兴起
      │  → "上下文窗口不够用"的痛点开始被广泛认知
      │
2023 ─┼─ Generative Agents (Stanford, Apr)
      │  → 记忆流+反思+规划，Agent Memory 的"iPhone 时刻"
      ├─ MemGPT (UC Berkeley, Oct)
      │  → 类 OS 虚拟内存管理，将记忆提升为系统级能力
      ├─ Reflexion (NeurIPS 2023)
      │  → 语言反思作为经验记忆
      ├─ CoALA 认知架构框架
      │  → 统一定义 Agent 记忆分类体系
      │
2024 ─┼─ Mem0 开源 (2024 H1)
      │  → 首个生产级通用记忆层，快速获得万级 Stars
      ├─ Zep / Graphiti 推出图谱记忆
      │  → 知识图谱路线与向量路线开始分化
      ├─ Letta 平台化（MemGPT → Letta）
      │  → 从研究原型走向商业平台
      ├─ LoCoMo 评测基准发布 (Snap Research)
      │  → 第一个标准化长期记忆评测基准
      │
2025 ─┼─ A-MEM: Zettelkasten 式自组织记忆 (Feb)
      ├─ Mem0 论文发表，向量+图混合架构标准化 (Apr)
      ├─ MemoryOS 获 EMNLP 2025 Oral
      │  → 三层记忆操作系统获学术认可
      ├─ H-MEM 四层层次记忆 (Jul)
      ├─ 记忆评测战：Mem0 vs Zep vs Letta 基准争议
      │  → 暴露当前评测方法论不成熟
      ├─ EverMemOS / MemMachine 刷新 LoCoMo SOTA
      ├─ 认知架构阶段：从工程化转向"记忆操作系统"
      │
2026 ─┼─ Agentic Memory: RL 驱动的统一记忆管理 (Jan)
      │  → 强化学习直接优化记忆操作
      ├─ Memory in the Age of AI Agents 综述
      │  → 标志领域从"碎片探索"进入"系统化"阶段
      └─ 当前状态：记忆已成为 Agent Infra 三大核心组件之一
         （LLM + Tool Use + Memory）
```

---

## 5. 生态现状分析

**当前最活跃项目**：Mem0 以 ~25K Stars 领跑开源社区，提供最易用的 API；Letta（原 MemGPT）聚焦有状态 Agent 平台；Zep/Graphiti 在图谱记忆方向持续深耕。三者构成当前的"第一梯队"。

**主要商业玩家**：Mem0 Inc.（SaaS + 开源）、Letta（平台化服务）、Zep（Cloud + Community）、EverMind AI（EverMemOS）、MemMachine。LangChain 和 LlamaIndex 作为框架层也内置了记忆模块。

**技术热点**：(1) 图谱 vs 向量的路线之争——图谱在关系推理上更强但延迟更高；(2) RL 驱动的自适应记忆管理成为最新研究方向；(3)"记忆操作系统"概念兴起，从单一存储层升级为完整的认知基础设施。

**争议焦点**：LoCoMo 基准的有效性受到质疑——Letta 实验表明简单的文件系统 Agent 就能达到 74%，暴露出当前评测无法有效区分"真正的记忆能力"与"强大的检索能力"。Mem0 的 SOTA 声明也被 Zep 团队公开质疑其评测方法论。
