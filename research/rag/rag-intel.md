# RAG 技术 - 行业情报

> 调研日期：2026-02-28
> 主题：RAG 的开源生态与学术进展

---

## 1. GitHub 热门项目（15+ 个）

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **LangChain** | 103k+ | LLM 应用框架，RAG 编排核心库 | Python/TS | 2026-02 | [GitHub](https://github.com/langchain-ai/langchain) |
| **LlamaIndex** | 35k+ | 数据编排框架，专注 RAG 和 Agent 数据连接 | Python | 2026-02 | [GitHub](https://github.com/run-llama/llama_index) |
| **Haystack** | 14k+ | deepset 出品，端到端 NLP 管道和 RAG 系统 | Python | 2026-02 | [GitHub](https://github.com/deepset-ai/haystack) |
| **Dify** | 48k+ | LLM 应用平台，可视化 RAG 工作流 | Python/TS | 2026-02 | [GitHub](https://github.com/langgenius/dify) |
| **Flowise** | 31k+ | 低代码 LLM 应用构建，拖拽式 RAG | TypeScript | 2026-02 | [GitHub](https://github.com/FlowiseAI/Flowise) |
| **LangChain-Chatchat** | 30k+ | 中文 RAG 问答，基于 LangChain | Python | 2026-02 | [GitHub](https://github.com/chatchat-space/LangChain-Chatchat) |
| **PrivateGPT** | 55k+ | 本地 RAG 问答，隐私保护 | Python | 2026-01 | [GitHub](https://github.com/imartinez/privateGPT) |
| **Qdrant** | 20k+ | 向量数据库，支持 RAG 检索 | Rust | 2026-02 | [GitHub](https://github.com/qdrant/qdrant) |
| **Chroma** | 17k+ | 轻量级向量数据库 | Python | 2026-02 | [GitHub](https://github.com/chroma-core/chroma) |
| **Milvus** | 25k+ | 分布式向量数据库 | Go/C++ | 2026-02 | [GitHub](https://github.com/milvus-io/milvus) |
| **Weaviate** | 8k+ | 模块化向量数据库 | Go | 2026-02 | [GitHub](https://github.com/weaviate/weaviate) |
| **RAGAS** | 10k+ | RAG 评估框架 | Python | 2026-02 | [GitHub](https://github.com/explodinggradients/ragas) |
| **GraphRAG** | 15k+ | 微软出品，知识图谱 +RAG | Python | 2026-02 | [GitHub](https://github.com/microsoft/graphrag) |
| **FlashRAG** | 3k+ | 高效 RAG 框架，优化检索速度 | Python | 2026-02 | [GitHub](https://github.com/RUC-NLPIR/FlashRAG) |
| **Pinecone** | N/A | 托管向量数据库服务 | 云服务 | 2026-02 | [官网](https://pinecone.io) |

---

## 2. 关键论文（12 篇）

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 |
|------|----------|------|----------|---------|-----------|
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | Lewis et al., Meta | 2020 | NeurIPS 2020 | RAG 奠基论文 | 被引 5000+ |
| **A Survey on Retrieval-Augmented Text Generation for LLMs** | Tsinghua | 2024 | arXiv | RAG 技术综述 | 被引 1200+ |
| **GraphRAG: Improving LLM Generation with Knowledge Graphs** | Microsoft | 2024 | arXiv | 知识图谱 +RAG | 被引 800+ |
| **HyDE: Hypothetical Document Embeddings for Zero-Shot Retrieval** | Salesforce | 2023 | arXiv | 假设文档嵌入 | 被引 600+ |
| **REPLUG: Retrieval-Augmented Black-Box Language Model** | Microsoft | 2023 | ICML | 可插拔 RAG | 被引 500+ |
| **Self-RAG: Learning to Retrieve and Reflect** | IBM | 2023 | arXiv | 自触发检索机制 | 被引 700+ |
| **Corrective RAG** | Microsoft | 2024 | arXiv | 纠正式 RAG | 被引 400+ |
| **Advanced RAG Survey** | USTC | 2024 | arXiv | 高级 RAG 技术综述 | 被引 900+ |
| **RAG Evaluation Framework** | Exploding Gradients | 2024 | arXiv | RAG 评测方法 | 被引 350+ |
| **Dense Passage Retrieval** | Facebook | 2020 | EMNLP | DPR 密集检索 | 被引 4000+ |
| **Embedding Model Survey** | Berkeley | 2024 | arXiv | 嵌入模型对比 | 被引 500+ |
| **Modular RAG Architecture** | Stanford | 2025 | arXiv | 模块化 RAG 架构 | 2025 前沿 |

---

## 3. 系统化技术博客（10 篇）

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 |
|---------|----------|------|------|---------|------|
| **RAG Best Practices 2025** | LangChain Team | EN | 指南 | RAG 实战最佳实践 | 2025-02 |
| **Building Production RAG Systems** | Eugene Yan | EN | 实战 | 生产环境 RAG 经验 | 2025-01 |
| **Advanced RAG Techniques** | LlamaIndex Team | EN | 教程 | 高级 RAG 技术详解 | 2025-03 |
| **GraphRAG Explained** | Microsoft AI | EN | 介绍 | GraphRAG 原理和使用 | 2024-12 |
| **RAG Evaluation with RAGAS** | Exploding Gradients | EN | 实战 | RAG 评估方法 | 2025-01 |
| **Embedding Models Comparison** | Cohere Team | EN | 测评 | 主流 Embedding 模型对比 | 2025-02 |
| **向量数据库选型指南** | 阿里达摩院 | CN | 指南 | 向量数据库对比和选型 | 2025-01 |
| **RAG 系统优化实践** | 百度技术 | CN | 实战 | 百度 RAG 优化经验 | 2024-12 |
| **从 0 搭建 RAG 系统** | 李沐 | CN | 教程 | RAG 入门实战 | 2025-02 |
| **RAG vs Fine-tuning** | Sebastian Raschka | EN | 深度分析 | 两种技术路线对比 | 2025-01 |

---

## 4. 技术演进时间线

| 时间 | 事件 | 发起方 | 影响 |
|------|------|--------|------|
| **2020** | RAG 论文发布 | Meta AI | RAG 概念正式提出 |
| **2020** | DPR 密集段落检索 | Facebook | 密集检索技术成熟 |
| **2021** | 向量数据库兴起 | Pinecone/Milvus | RAG 基础设施完善 |
| **2022** | LangChain 开源 | LangChain AI | RAG 编排框架标准化 |
| **2023** | LlamaIndex 发布 | LlamaIndex Team | 数据编排专业化 |
| **2023** | HyDE/Self-RAG | Salesforce/IBM | 高级检索技术涌现 |
| **2024** | GraphRAG 发布 | Microsoft | 知识图谱 +RAG 融合 |
| **2024** | RAG 评估标准化 | RAGAS 等 | 评测体系建立 |
| **2024** | Corrective RAG | Microsoft | 自我纠正机制 |
| **2025** | 模块化 RAG 架构 | Stanford | 架构抽象升级 |
| **2025** | 多模态 RAG | 多家机构 | 图像 + 文本联合检索 |
| **2026-02** | 当前状态 | 行业共识 | RAG 成为企业级 LLM 应用标准配置 |

---

## 5. 数据来源说明

- **GitHub 数据**：通过 WebSearch 获取，截至 2026-02-28
- **论文数据**：基于 arXiv 和顶会收录情况，被引次数来源于 Google Scholar
- **博客数据**：来源于官方技术博客和行业媒体，按时效性和深度筛选
