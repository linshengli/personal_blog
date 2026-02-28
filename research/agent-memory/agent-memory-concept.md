# Agent Memory 概念剖析

## 1. 定义澄清

**通行定义**：Agent Memory（智能体记忆）是赋予 LLM-based Agent 跨交互持久化信息存储与检索能力的系统，使 Agent 能够在多轮对话、多会话甚至多任务间保持上下文连贯性、积累经验并持续学习。它本质上解决了 LLM 的"无状态"问题——让 Agent 从"金鱼记忆"进化为具有"人类式记忆"的智能体。

**常见误解**：

1. **记忆 ≠ 上下文窗口**：上下文窗口是 LLM 单次推理的输入容量限制（如 128K tokens），而记忆是跨会话的持久化信息管理系统。上下文窗口是"桌面大小"，记忆是"书房+图书馆"。
2. **记忆 ≠ RAG**：RAG 从外部知识库检索静态文档片段辅助生成，而 Agent Memory 是动态的——它会根据交互经历不断写入、更新、遗忘和反思。RAG 是"查字典"，记忆是"亲身经历过"。
3. **记忆 ≠ 缓存/KV Cache**：KV Cache 是推理加速的底层优化，缓存的是注意力计算的中间结果；Agent Memory 是语义层面的信息管理，包含抽象、压缩和检索等高级操作。
4. **记忆 ≠ 向量数据库**：向量数据库是记忆的一种**存储后端**，而非记忆本身。记忆系统还包含写入策略、检索策略、遗忘机制、反思与压缩等完整的管理逻辑。

**边界辨析**：

| 概念 | 核心差异 |
|------|---------|
| RAG | 静态知识检索 vs 动态经验管理 |
| KV Cache | 推理层加速 vs 语义层管理 |
| 上下文窗口 | 单次容量限制 vs 持久化存储系统 |
| 向量数据库 | 存储后端 vs 完整管理系统 |
| Fine-tuning | 参数级知识固化 vs 实例级信息存取 |

---

## 2. 核心架构

```
┌───────────────────────────────────────────────────────────────────┐
│                    Agent Memory System                            │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐    ┌──────────────────────────────────────────┐ │
│  │  感知输入层   │    │         Agent 决策/推理层 (LLM)          │ │
│  │  Perception  │───▶│  Planning · Reasoning · Action          │ │
│  │  用户输入     │    │         ▲              │                 │ │
│  │  环境反馈     │    │         │ retrieve     │ store           │ │
│  │  工具返回     │    │         │              ▼                 │ │
│  └─────────────┘    │  ┌──────┴──────────────────┐             │ │
│                      │  │   Memory Controller      │             │ │
│                      │  │   记忆控制器               │             │ │
│                      │  │   · 写入策略 (何时/何物)   │             │ │
│                      │  │   · 检索策略 (相关性+时效)  │             │ │
│                      │  │   · 更新/遗忘调度          │             │ │
│                      │  │   · 反思/压缩触发          │             │ │
│                      │  └──┬────────┬────────┬───┘             │ │
│                      └─────┼────────┼────────┼─────────────────┘ │
│                            │        │        │                    │
│  ┌─────────────────────────┼────────┼────────┼──────────────────┐│
│  │        Memory Store     │        │        │                  ││
│  │  ┌─────────────┐  ┌────┴───┐ ┌──┴────┐ ┌─┴──────────┐      ││
│  │  │ Working Mem  │  │Episodic│ │Semantic│ │ Procedural │      ││
│  │  │ 工作记忆     │  │ Memory │ │Memory  │ │  Memory    │      ││
│  │  │ (上下文窗口) │  │情景记忆 │ │语义记忆 │ │ 程序性记忆  │      ││
│  │  │ 当前对话     │  │交互历史 │ │知识事实 │ │ 技能/模式  │      ││
│  │  └─────────────┘  └────────┘ └───────┘ └────────────┘      ││
│  │                                                              ││
│  │  Storage Backends: Vector DB │ Graph DB │ KV Store │ File    ││
│  └──────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────┘
```

**组件职责**：

- **感知输入层**：接收用户消息、环境观测、工具调用返回等原始信息
- **记忆控制器**：核心调度中心，决定何时写入、如何检索、何时遗忘/压缩
- **工作记忆**：当前上下文窗口内的活跃信息，容量受 LLM 窗口限制
- **情景记忆**：历史交互的具体片段（谁在何时说了什么、做了什么）
- **语义记忆**：从交互中提炼的事实、偏好和知识（用户喜欢 Python、项目用 React）
- **程序性记忆**：学到的操作模式和技能（上次解决类似 bug 的步骤）
- **Agent 决策层**：基于检索到的记忆进行推理、规划和行动

---

## 3. 记忆类型分类

### 3.1 工作记忆 (Working Memory)

| 属性 | 说明 |
|------|------|
| **对应** | LLM 上下文窗口中的活跃信息 |
| **容量** | 受限于模型窗口（4K~2M tokens） |
| **持久性** | 会话内有效，会话结束即丢失 |
| **实现** | 对话历史拼接、System Prompt 注入 |
| **用途** | 当前任务的即时推理上下文 |

### 3.2 情景记忆 (Episodic Memory)

| 属性 | 说明 |
|------|------|
| **对应** | 人类的"自传式回忆" |
| **内容** | 具体事件：时间、参与者、上下文、结果 |
| **存储** | 向量化后存入向量数据库，保留时间戳 |
| **检索** | 相似度搜索 + 时间衰减加权 |
| **用途** | "你上周让我改的那个 API 接口..." |

### 3.3 语义记忆 (Semantic Memory)

| 属性 | 说明 |
|------|------|
| **对应** | 人类的"常识和知识" |
| **内容** | 提炼的事实、用户偏好、实体关系 |
| **存储** | 知识图谱或结构化 KV 存储 |
| **检索** | 实体查询、关系遍历 |
| **用途** | "用户偏好深色主题"、"项目使用 TypeScript" |

### 3.4 程序性记忆 (Procedural Memory)

| 属性 | 说明 |
|------|------|
| **对应** | 人类的"肌肉记忆/技能" |
| **内容** | 成功的操作序列、工具调用模式、问题解决策略 |
| **存储** | 模式模板 + 成功率统计 |
| **检索** | 任务类型匹配 |
| **用途** | "部署流程：先跑测试→构建→推送→验证" |

### 3.5 联想记忆 (Associative Memory)

| 属性 | 说明 |
|------|------|
| **对应** | 人类的"触发式联想" |
| **内容** | 概念之间的关联强度和路径 |
| **存储** | 图结构，边带权重 |
| **检索** | 图遍历、扩散激活 |
| **用途** | "提到 Redis → 联想到缓存策略 → 联想到上次的性能优化方案" |

---

## 4. 数学形式化

### 4.1 记忆检索相似度

$$
\text{relevance}(q, m_i) = \alpha \cdot \text{sim}(E(q), E(m_i)) + \beta \cdot \text{recency}(m_i) + \gamma \cdot \text{importance}(m_i)
$$

检索得分是语义相似度、时间新近性和记忆重要性的加权组合。源自 Generative Agents (Park et al., 2023) 的经典设计。

### 4.2 时间衰减函数（遗忘曲线）

$$
\text{recency}(m_i) = e^{-\lambda \cdot (t_{now} - t_{create}(m_i))}
$$

模拟 Ebbinghaus 遗忘曲线，记忆的新近性得分随时间指数衰减，λ 为衰减速率参数。

### 4.3 记忆重要性评分

$$
\text{importance}(m_i) = \text{LLM\_Score}(m_i) \cdot (1 + \log(1 + \text{access\_count}(m_i)))
$$

重要性由 LLM 对内容的重要性判断乘以访问频率的对数增益决定。被频繁检索的记忆会获得更高的留存优先级。

### 4.4 记忆压缩/反思

$$
M_{reflected} = \text{LLM}(\text{summarize}(M_{raw}[t_1:t_2])) \quad \text{when } |M_{raw}| > \theta
$$

当原始记忆数量超过阈值 θ 时，触发 LLM 对一段时间窗口内的记忆进行反思式摘要，生成更高层次的抽象记忆。

### 4.5 上下文预算分配

$$
C_{total} = C_{system} + C_{working} + C_{retrieved} + C_{output} \leq W_{max}
$$

上下文窗口（W_max）需在系统提示、工作记忆、检索记忆和输出预留之间做预算分配，这是 MemGPT 的核心约束。

---

## 5. 实现逻辑

```python
class AgentMemory:
    """Agent 记忆系统核心类"""

    def __init__(self, config):
        self.working_memory = []            # 当前上下文窗口内容
        self.vector_store = VectorDB(config) # 情景+语义记忆存储
        self.graph_store = GraphDB(config)   # 关联记忆/知识图谱
        self.llm = LLMClient(config)         # 用于重要性评估和反思
        self.max_working = config.max_tokens # 工作记忆容量上限
        self.decay_rate = config.decay_rate  # 遗忘衰减率

    def store(self, experience: str, metadata: dict) -> str:
        """记忆写入：评估重要性后持久化"""
        importance = self.llm.score_importance(experience)
        if importance < self.min_threshold:
            return None  # 过滤低价值信息

        memory_id = self.vector_store.insert(
            text=experience,
            embedding=self.llm.embed(experience),
            metadata={**metadata, "importance": importance,
                      "created_at": now(), "access_count": 0}
        )
        # 提取实体关系写入图谱
        entities = self.llm.extract_entities(experience)
        self.graph_store.add_relations(entities)
        return memory_id

    def retrieve(self, query: str, k: int = 5) -> list:
        """记忆检索：综合相关性、时效性、重要性排序"""
        candidates = self.vector_store.search(
            embedding=self.llm.embed(query), top_k=k * 3
        )
        scored = []
        for mem in candidates:
            score = (
                0.5 * mem.similarity +
                0.3 * exp(-self.decay_rate * (now() - mem.created_at)) +
                0.2 * mem.importance
            )
            scored.append((score, mem))
        scored.sort(reverse=True)

        # 更新访问计数
        top_k = [m for _, m in scored[:k]]
        for m in top_k:
            self.vector_store.increment_access(m.id)
        return top_k

    def reflect(self) -> str:
        """记忆反思：将近期记忆压缩为高阶洞察"""
        recent = self.vector_store.get_recent(hours=24)
        if len(recent) < 10:
            return None
        summary = self.llm.generate(
            f"Synthesize these experiences into 3 key insights:\n"
            f"{[m.text for m in recent]}"
        )
        self.store(summary, {"type": "reflection", "source_count": len(recent)})
        return summary

    def forget(self):
        """记忆遗忘：清理低价值和过期记忆"""
        all_memories = self.vector_store.get_all()
        for mem in all_memories:
            retention_score = (
                mem.importance *
                exp(-self.decay_rate * (now() - mem.created_at)) *
                (1 + log(1 + mem.access_count))
            )
            if retention_score < self.forget_threshold:
                self.vector_store.delete(mem.id)

    def update(self, memory_id: str, new_info: str):
        """记忆更新：用新信息修正已有记忆"""
        old = self.vector_store.get(memory_id)
        merged = self.llm.generate(
            f"Update this memory with new information:\n"
            f"Old: {old.text}\nNew: {new_info}"
        )
        self.vector_store.update(memory_id, text=merged,
                                  embedding=self.llm.embed(merged))
```

---

## 6. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 检索延迟 | < 200 ms | 端到端查询耗时 | 含 Embedding + 向量搜索 + 重排序 |
| 检索准确率 | > 75% (LoCoMo) | 标准评测集 | 跨会话长期记忆问答准确率 |
| 遗忘精度 | > 90% | 人工评估 | 被遗忘记忆中真正低价值的比例 |
| 上下文利用率 | > 80% | Token 利用分析 | 注入上下文中与当前查询相关的比例 |
| 存储效率 | < 1KB/轮对话 | 压缩后存储量 | 经过摘要压缩后的平均存储开销 |
| 写入吞吐 | > 50 ops/s | 批量写入测试 | 每秒可处理的记忆写入操作数 |
| 一致性 | > 95% | 矛盾检测 | 同一实体不同记忆间的一致性比率 |

---

## 7. 扩展性与安全性

### 水平扩展

- **分片策略**：按用户 ID 或会话 ID 分片，不同用户的记忆存储在不同节点
- **多级缓存**：热记忆（近期高频访问）保留在 Redis，冷记忆下沉到持久化向量库
- **异步写入**：记忆存储采用消息队列异步处理，不阻塞主推理链路

### 垂直扩展

- **压缩上限**：单节点向量库可存储约 1000 万条记忆（取决于维度和索引类型）
- **窗口优化**：通过分层摘要将有效记忆容量从上下文窗口的线性限制提升为对数级增长

### 安全考量

- **记忆注入攻击（Memory Poisoning）**：恶意用户通过精心构造的输入污染记忆库，使后续检索返回有害内容。防护手段包括记忆写入校验、来源标注、异常检测
- **隐私泄露**：跨用户记忆隔离不严格可能导致 A 用户的信息被 B 用户的检索召回。需严格的 Tenant 隔离和访问控制
- **记忆幻觉**：LLM 在反思/摘要阶段可能引入不存在的"虚假记忆"，需通过事实校验机制防护
- **数据合规**：记忆中可能存储 PII（个人身份信息），需符合 GDPR 等法规的"被遗忘权"要求
