# RAG 技术 - 概念剖析

> 调研日期：2026-02-28
> 主题：RAG 技术的核心原理与架构

---

## 1. 定义澄清

### 通行定义

**RAG（Retrieval-Augmented Generation，检索增强生成）**是一种将信息检索与大型语言模型生成相结合的技术架构。它通过从外部知识库中检索相关文档或片段，将其作为上下文输入给 LLM，从而增强生成内容的准确性和可信度。

RAG 的核心价值在于解决 LLM 的三大固有局限：**知识时效性**（训练数据截止）、**幻觉问题**（编造事实）和**垂直领域知识缺失**，使 LLM 能够访问最新、最准确的外部信息。

### 常见误解

| 误解 | 正解 |
|------|------|
| "RAG 就是向量搜索" | RAG 包含检索、重排序、上下文构建、生成多个环节，向量搜索仅是检索方式之一 |
| "RAG 能完全消除幻觉" | RAG 可显著降低幻觉，但无法完全消除，仍需其他机制配合 |
| "RAG 只需要向量数据库" | 完整的 RAG 系统还需要文档处理、检索策略、重排序、评估等组件 |
| "切片越小检索越准" | 切片大小需要平衡召回率和精确率，过大过小都会影响效果 |

### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **RAG vs 向量搜索** | 向量搜索是检索技术；RAG 是检索 + 生成的完整架构 |
| **RAG vs Fine-tuning** | Fine-tuning 更新模型权重；RAG 保持模型不变，通过上下文增强 |
| **RAG vs 知识库问答** | 传统 KBQA 依赖结构化知识；RAG 可处理非结构化文档 |

---

## 2. 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                    RAG 系统架构                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  文档接入   │ ──→ │  文档处理   │ ──→ │  向量化     │    │
│  │  (Ingestion)│     │ (切片/清洗)  │     │  (Embedding)│    │
│  └─────────────┘     └─────────────┘     └──────┬──────┘    │
│                                                  ↓         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  生成响应   │ ←── │  重排序     │ ←── │  检索       │    │
│  │  (LLM)      │     │  (Rerank)   │     │  (Retrieve) │    │
│  └─────────────┘     └─────────────┘     └──────┬──────┘    │
│                                                  ↓         │
│                    ┌────────────────────────────┘          │
│                    ↓                                       │
│              ┌─────────────┐                               │
│              │  向量数据库  │                               │
│              │ (Vector DB) │                               │
│              └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

**组件职责说明：**

| 组件 | 职责 |
|------|------|
| **文档接入** | 从多种数据源（文件、API、数据库）获取原始文档 |
| **文档处理** | 文本清洗、分块（Chunking）、元数据提取 |
| **向量化** | 使用 Embedding 模型将文本转换为向量表示 |
| **检索** | 根据查询向量从数据库中检索相关文档 |
| **重排序** | 对检索结果进行精排，提升 Top-K 相关性 |
| **生成** | LLM 结合检索结果生成最终响应 |

---

## 3. 数学形式化

### 3.1 检索相关性评分

对于查询 $q$ 和文档 $d$，相关性评分：

$$\text{score}(q, d) = \frac{\vec{v}_q \cdot \vec{v}_d}{\|\vec{v}_q\| \|\vec{v}_d\|} = \cos(\theta)$$

其中 $\vec{v}_q$ 和 $\vec{v}_d$ 分别为查询和文档的嵌入向量。

**自然语言解释**：相关性由查询和文档向量的余弦相似度衡量。

### 3.2 Top-K 检索

检索最相关的 $K$ 个文档：

$$D_{\text{top}} = \underset{d \in D, |D|=K}{\text{argmax}} \sum_{d} \text{score}(q, d)$$

**自然语言解释**：从文档库中选择与查询最相关的前 K 个文档。

### 3.3 RAG 生成概率

LLM 生成答案 $a$ 的条件概率：

$$P(a | q, D) = \sum_{d \in D_{\text{top}}} P(a | q, d) \cdot P(d | q)$$

**自然语言解释**：最终答案是基于所有检索文档的加权生成概率。

### 3.4 检索召回率与精确率

$$\text{Recall@K} = \frac{|D_{\text{relevant}} \cap D_{\text{retrieved}}|}{|D_{\text{relevant}}|}$$

$$\text{Precision@K} = \frac{|D_{\text{relevant}} \cap D_{\text{retrieved}}|}{K}$$

**自然语言解释**：召回率衡量查全程度，精确率衡量查准程度。

---

## 4. 实现逻辑（Python 伪代码）

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class Document:
    """文档片段"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    score: Optional[float] = None

class EmbeddingModel:
    """文本向量化模型"""

    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    def embed(self, texts: List[str]) -> np.ndarray:
        """将文本转换为向量"""
        pass

class VectorStore:
    """向量数据库"""

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors: List[np.ndarray] = []
        self.documents: List[Document] = []

    def add(self, documents: List[Document]):
        """添加文档到向量库"""
        for doc in documents:
            self.vectors.append(doc.embedding)
            self.documents.append(doc)

    def search(self, query_embedding: np.ndarray,
               top_k: int) -> List[Document]:
        """检索最相关的文档"""
        # 计算余弦相似度
        scores = []
        for vec in self.vectors:
            score = np.dot(query_embedding, vec) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vec)
            )
            scores.append(score)

        # 返回 Top-K
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            doc.score = scores[idx]
            results.append(doc)
        return results

class Reranker:
    """重排序模型"""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def rerank(self, query: str,
               documents: List[Document]) -> List[Document]:
        """对检索结果进行重排序"""
        # 使用交叉编码器计算更精确的相关性
        pass

class TextSplitter:
    """文本分块器"""

    def __init__(self, chunk_size: int = 500,
                 chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> List[str]:
        """将文本分割成块"""
        # 按句子或段落分割，保持语义完整性
        pass

class RAGSystem:
    """RAG 系统核心类"""

    def __init__(self, embedding_model: str,
                 vector_store: VectorStore,
                 reranker: Optional[Reranker] = None,
                 llm=None):
        self.embedder = EmbeddingModel(embedding_model, dimension=1536)
        self.vector_store = vector_store
        self.reranker = reranker
        self.llm = llm

    def ingest_documents(self, texts: List[str],
                         metadata_list: List[Dict] = None):
        """
        文档接入流程
        """
        # 阶段 1: 文本分块
        splitter = TextSplitter()
        chunks = []
        for text in texts:
            chunks.extend(splitter.split(text))

        # 阶段 2: 向量化
        embeddings = self.embedder.embed(chunks)

        # 阶段 3: 存储
        documents = [
            Document(content=chunk,
                    metadata=metadata_list[i] if metadata_list else {},
                    embedding=embeddings[i])
            for i, chunk in enumerate(chunks)
        ]
        self.vector_store.add(documents)

    def query(self, question: str, top_k: int = 5) -> str:
        """
        RAG 查询流程
        检索 → 重排序 → 生成
        """
        # 阶段 1: 检索
        query_embedding = self.embedder.embed([question])[0]
        retrieved_docs = self.vector_store.search(
            query_embedding, top_k=top_k * 2  # 初检多返回一些
        )

        # 阶段 2: 重排序
        if self.reranker:
            retrieved_docs = self.reranker.rerank(
                question, retrieved_docs
            )[:top_k]

        # 阶段 3: 构建上下文
        context = "\n\n".join([doc.content for doc in retrieved_docs])

        # 阶段 4: LLM 生成
        prompt = self._build_prompt(question, context)
        answer = self.llm.generate(prompt)

        return answer
```

---

## 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **检索召回率@10** | > 85% | 标准测试集 | 相关文档被检索到的比例 |
| **检索精确率@10** | > 70% | 标准测试集 | 检索结果中相关的比例 |
| **NDCG@10** | > 0.75 | 标准评测集 | 排序质量指标 |
| **答案准确率** | > 80% | 人工评估/自动评测 | 生成答案的正确性 |
| **检索延迟** | < 100ms | 端到端基准测试 | 单次检索的平均耗时 |
| **幻觉率** | < 10% | 事实核查测试 | 生成内容编造事实的比例 |

---

## 6. 扩展性与安全性

### 水平扩展

- **分布式向量检索**：使用 HNSW、IVF 等近似最近邻算法支持亿级向量
- **分片存储**：按文档类型或时间分片，提升检索效率
- **缓存层**：对热门查询结果进行缓存，减少重复检索

### 垂直扩展

- **Embedding 模型优化**：使用更大、更专业的嵌入模型提升表示质量
- **混合检索**：结合关键词检索（BM25）和向量检索，提升召回率
- **多跳检索**：支持迭代式多轮检索，获取更深层信息

### 安全考量

| 风险 | 防护措施 |
|------|---------|
| **数据泄露** | 访问控制、文档级权限管理、数据加密 |
| **Prompt 注入** | 输入过滤、上下文隔离、系统指令保护 |
| **知识库污染** | 文档来源验证、内容审核、版本控制 |
| **隐私数据** | PII 检测与脱敏、访问审计、最小权限原则 |
| **检索投毒** | 来源白名单、异常检测、多源交叉验证 |
