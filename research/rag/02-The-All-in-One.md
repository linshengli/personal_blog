# RAG 技术 - 精华整合

> 调研日期：2026-02-28
> 主题：RAG 技术的核心洞察

---

## 1. The One 公式

$$
\text{RAG} = \underbrace{\text{检索}}_{\text{找信息}} + \underbrace{\text{增强}}_{\text{建上下文}} + \underbrace{\text{生成}}_{\text{答问题}} - \underbrace{\text{幻觉}}_{\text{需抑制}}
$$

**解读**：RAG 的本质是从外部知识库检索相关信息，构建增强上下文输入给 LLM，让其基于真实信息生成答案，同时抑制幻觉。

---

## 2. 一句话解释

> RAG 就像一个"开卷考试"——LLM 不再是闭卷作答（仅靠训练记忆），而是可以查阅参考书（外部知识库），从而给出更准确、更有依据的答案。

---

## 3. 核心架构图

```
问题 → [向量化] → [检索] → [重排序] → [构建上下文] → [LLM 生成] → 答案
                      ↓                                          ↑
               ┌─────────────┐                          ┌────────┐
               │  向量数据库  │                          │ 上下文  │
               └─────────────┘                          └────────┘
```

---

## 4. 主流框架速查表

| 框架 | 核心抽象 | 上手难度 | 适用场景 |
|------|---------|---------|---------|
| **LangChain** | Chain | ⭐⭐⭐⭐ | 复杂工作流、企业级 |
| **LlamaIndex** | Index | ⭐⭐⭐ | 数据密集型 RAG |
| **Haystack** | Pipeline | ⭐⭐⭐ | 生产环境 |
| **GraphRAG** | Knowledge Graph | ⭐⭐⭐⭐ | 知识推理 |
| **Flowise/Dify** | Visual Flow | ⭐⭐ | 快速原型 |

---

## 5. STAR 总结

### Situation（背景 + 痛点）

LLM 存在三大固有局限：**知识时效性受限**（训练数据有截止日期）、**幻觉问题**（可能编造事实）、**垂直领域知识缺失**（专业领域表现差）。传统 Fine-tuning 方案成本高、更新慢，无法解决实时知识需求。企业需要一种既能利用 LLM 强大生成能力，又能确保答案准确可靠的技术方案。

### Task（核心问题）

RAG 需要解决的关键问题是：**如何在保持 LLM 生成能力的同时，确保答案基于真实、准确的外部信息**。核心约束包括：检索准确性（Recall@10 > 85%）、响应延迟（<500ms）、成本可控（Token 用量优化）、幻觉抑制（事实核查通过率 >90%）。

### Action（主流方案）

技术演进历经三代：**第一代**（基础 RAG）建立检索 + 生成框架；**第二代**（Advanced RAG）引入混合检索、重排序、Query 改写等优化；**第三代**（GraphRAG/Modular RAG）融合知识图谱和模块化架构。核心突破包括：稠密检索技术、混合检索策略、Self-RAG 自触发机制、GraphRAG 多跳推理、RAGAS 评估体系。

### Result（效果 + 建议）

当前 RAG 系统可将答案准确率提升至 80-90%，幻觉率降低至 10% 以下，成为企业级 LLM 应用的标准配置。但仍存在**检索质量依赖**、**多跳推理弱**、**评估复杂**等挑战。实操建议：**原型用 Flowise/Dify**，**数据密集型用 LlamaIndex**，**企业级用 LangChain**，**知识推理用 GraphRAG**。

---

## 6. 关键洞察

### 6.1 RAG 技术栈

```
┌─────────────────────────────────────┐
│          应用层                      │
│  (LangChain/LlamaIndex/Haystack)   │
├─────────────────────────────────────┤
│          检索层                      │
│  (向量检索 + 关键词检索 + 重排序)    │
├─────────────────────────────────────┤
│          存储层                      │
│  (Qdrant/Milvus/Chroma/Pinecone)   │
├─────────────────────────────────────┤
│          数据层                      │
│  (Embedding 模型 + 文档处理)         │
└─────────────────────────────────────┘
```

### 6.2 选型决策树

```
                    ┌─────────────────┐
                    │  你的需求是什么？ │
                    └────────┬────────┘
                             │
        ┌────────────┬───────┴───────┬────────────┐
        ↓            ↓               ↓            ↓
   ┌────────┐  ┌────────┐    ┌──────────┐  ┌────────┐
   │ 快速原型│  │数据密集│    │企业级    │  │知识    │
   │        │  │型      │    │生产      │  │图谱    │
   └───┬────┘  └───┬────┘    └─────┬────┘  └───┬────┘
       ↓           ↓               ↓           ↓
   ┌───────┐  ┌──────────┐   ┌─────────┐  ┌────────┐
   │Flowise│  │Llama     │   │LangChain│  │Graph   │
   │/Dify  │  │Index     │   │/Haystack│  │RAG     │
   └───────┘  └──────────┘   └─────────┘  └────────┘
```

### 6.3 成本估算模型

对于月查询量 100,000 次的 RAG 系统：

| 成本项 | 估算 | 说明 |
|--------|------|------|
| **LLM API** | $500-2000/月 | Token 用量，取决于上下文长度 |
| **向量数据库** | $100-500/月 | Pinecone/Weaviate 等 |
| **Embedding** | $50-200/月 | 文档和查询向量化 |
| **计算资源** | $200-1000/月 | 应用服务器、缓存 |
| **合计** | **$850-3700/月** | 视数据量和并发量浮动 |

---

## 7. 理解确认问题

**问题**：RAG 和 Fine-tuning 各适用什么场景？为什么 RAG 成为企业级 LLM 应用的首选方案？

**参考答案**：RAG 适用场景：1）知识频繁更新（新闻、政策等）；2）垂直领域专业问答；3）需要引用和溯源；4）私有数据问答。Fine-tuning 适用场景：1）特定任务风格迁移；2）领域术语理解；3）固定格式输出；4）数据量充足且稳定。RAG 成为首选的原因：a）无需更新模型，知识更新成本低；b）答案可溯源，可信度高；c）实现简单，开箱即用；d）隐私数据可本地部署。

---

## 8. 快速开始指南

### 8.1 LlamaIndex 5 分钟原型

```bash
pip install llama-index
```

```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

response = query_engine.query("什么是 RAG 技术？")
print(response)
```

### 8.2 LangChain 快速开始

```bash
pip install langchain langchain-community
```

```python
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma(embedding_function=embeddings)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)

result = qa_chain({"query": "什么是 RAG 技术？"})
```

### 8.3 下一步学习资源

- **官方文档**：LangChain、LlamaIndex、Haystack 官网
- **实战教程**：Eugene Yan 博客、Microsoft AI Blog
- **评估工具**：RAGAS、TruLens

---

## 9. 检查清单

在将 RAG 投入生产前，请确认：

- [ ] 文档分块策略已优化（chunk_size、chunk_overlap）
- [ ] Embedding 模型已针对领域选择
- [ ] 检索策略已配置（top_k、阈值）
- [ ] 重排序模型已部署（如需要）
- [ ] 幻觉检测机制已配置
- [ ] 评估指标已定义（准确率、召回率、延迟）
- [ ] 监控和日志系统已就绪
