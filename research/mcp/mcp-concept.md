# MCP (Model Context Protocol) 概念剖析报告

> 调研日期：2026-03-01
> 主题：MCP 技术概念与原理

---

## 1. 定义澄清

### 1.1 通行定义

**MCP (Model Context Protocol)** 是一个开源的、标准化的协议，用于连接 AI 应用程序与外部系统。它提供了一种统一的方式来暴露数据源、工具和预定义交互模板，使 AI 模型能够访问关键信息并执行任务。

MCP 的核心类比是 **"AI 应用的 USB-C 端口"**：正如 USB-C 为电子设备提供标准化连接，MCP 为 AI 应用与外部系统的连接提供了标准化协议。

MCP 包含三个核心层次：
- **数据层**：基于 JSON-RPC 2.0 的协议交换，定义消息格式和语义
- **传输层**：管理通信通道（Stdio/HTTP）和认证机制
- **原语层**：定义 Tools、Resources、Prompts 三大核心原语

### 1.2 常见误解

| 误解 | 正确理解 |
|------|----------|
| **误解 1：MCP 是一个 AI 框架或 SDK** | MCP 是**协议规范**，SDK 只是协议的实现。核心是协议本身，而非具体实现 |
| **误解 2：MCP 等同于 Function Calling** | Function Calling 是 LLM 的内部能力，MCP 是**外部系统与 AI 应用之间的桥梁协议** |
| **误解 3：MCP 只能用于本地工具调用** | MCP 支持**本地 (Stdio)** 和**远程 (HTTP)** 两种传输，可连接云端服务 |
| **误解 4：MCP 管理 LLM 如何调用工具** | MCP **不决定** AI 如何使用上下文，只定义如何提供上下文 |

### 1.3 边界辨析

| 概念 | 核心职责 | 与 MCP 的区别 |
|------|----------|---------------|
| **Function Calling** | LLM 内部机制，用于生成工具调用请求 | MCP 是**协议层**，Function Calling 是**模型层** |
| **Plugin (插件)** | 特定平台 (如 ChatGPT) 的扩展机制 | MCP 是**跨平台开放标准**，不绑定特定厂商 |
| **API 集成** | 点对点的接口调用 | MCP 提供**标准化的发现和协商机制**，支持动态能力发现 |
| **Agent Framework** | 智能体的编排和执行框架 | MCP 专注于**上下文交换协议**，不涉及 Agent 编排逻辑 |

---

## 2. 核心架构

### 2.1 参与者架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP Host (AI Application)                          │
│                        (e.g., Claude Desktop, Claude Code)                   │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐              │
│  │   MCP Client 1  │  │   MCP Client 2  │  │   MCP Client 3  │              │
│  │   (Session A)   │  │   (Session B)   │  │   (Session C)   │              │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘              │
│           │                    │                    │                        │
│           │ Dedicated          │ Dedicated          │ Dedicated              │
│           │ Connection         │ Connection         │ Connection             │
│           ▼                    ▼                    ▼                        │
└─────────────────────────────────────────────────────────────────────────────┘
           │                    │                    │
           │ Stdio              │ Stdio              │ HTTP                   │
           │ Transport          │ Transport          │ Transport              │
           ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌───────────────────────────────┐  │
│   MCP Server A   │  │   MCP Server B   │  │       MCP Server C            │  │
│   (Local)        │  │   (Local)        │  │       (Remote)                │  │
│   Filesystem     │  │   Database       │  │       e.g., Sentry API        │  │
└──────────────────┘  └──────────────────┘  └───────────────────────────────┘  │
```

### 2.2 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
│  (AI Host: Claude Desktop, Claude Code, Custom Apps)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Data Layer (Protocol)                        │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              JSON-RPC 2.0 Message Exchange                │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  Lifecycle Management │  Capability Negotiation           │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  Server Primitives           │  Client Primitives         │  │
│  │  • Tools (list/call)         │  • Sampling (complete)     │  │
│  │  • Resources (list/read)     │  • Elicitation (request)   │  │
│  │  • Prompts (list/get)        │  • Logging                 │  │
│  ├───────────────────────────────────────────────────────────┤  │
│  │  Notifications │  Progress Tracking │  Tasks (Experimental)│  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Transport Layer                               │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐  │
│  │   Stdio Transport       │  │   HTTP Transport            │  │
│  │   (Local IPC)           │  │   (Remote RPC)              │  │
│  │   • stdin/stdout        │  │   • HTTP POST               │  │
│  │   • Low latency         │  │   • SSE Streaming           │  │
│  │   • Single client       │  │   • OAuth/Bearer Auth       │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 数据流向图

```
┌──────────────┐                    ┌──────────────┐
│  MCP Client  │                    │  MCP Server  │
│   (Host)     │                    │   (Provider) │
└──────┬───────┘                    └──────┬───────┘
       │                                   │
       │  1. initialize (capabilities)     │
       │──────────────────────────────────>│
       │                                   │
       │  2. initialize response           │
       │<──────────────────────────────────│
       │                                   │
       │  3. notifications/initialized     │
       │──────────────────────────────────>│
       │                                   │
       │  4. tools/list                    │
       │──────────────────────────────────>│
       │                                   │
       │  5. tools list response           │
       │<──────────────────────────────────│
       │                                   │
       │  6. tools/call (name, args)       │
       │──────────────────────────────────>│
       │                                   │
       │  7. tool result (content)         │
       │<──────────────────────────────────│
       │                                   │
       │  8. notifications/tools/list_changed (async)
       │<──────────────────────────────────│
```

---

## 3. 数学形式化

### 3.1 协议通信模型

MCP 基于 JSON-RPC 2.0 构建，可形式化为以下通信模型：

**定义 1 (MCP 消息三元组)**：每条 MCP 消息可表示为三元组 $M = (method, params, id)$，其中：
- $method \in \mathcal{M}$ 是方法名，来自预定义的方法集合
- $params \in \mathcal{P}$ 是参数对象
- $id \in \mathbb{N} \cup \{\bot\}$ 是请求标识符，$\bot$ 表示通知（无需响应）

**定义 2 (响应映射)**：响应函数 $R: \mathbb{N} \times \mathcal{M} \rightarrow \mathcal{R} \cup \mathcal{E}$ 将请求映射到结果或错误：

$$R(id, method) = \begin{cases}
(result, \bot) & \text{if success} \\
(\bot, error) & \text{if failure}
\end{cases}$$

### 3.2 能力协商形式化

**定义 3 (能力协商)**：初始化握手过程可形式化为能力交集计算：

$$C_{effective} = C_{client} \cap C_{server} = \{(p, f) \mid (p, f) \in C_{client} \land (p, f) \in C_{server}\}$$

其中：
- $C_{client} = \{(p_i, f_i) \mid p_i \in Primitives, f_i \in Features\}$
- $C_{server} = \{(p_j, f_j) \mid p_j \in Primitives, f_j \in Features\}$
- $Primitives = \{tools, resources, prompts, \dots\}$
- $Features = \{listChanged, subscribe, \dots\}$

**协议版本兼容性**：

$$Compatible(v_c, v_s) = \begin{cases}
true & \text{if } |date(v_c) - date(v_s)| \leq \delta \\
false & \text{otherwise}
\end{cases}$$

其中 $\delta$ 是协议版本容忍窗口（通常为 0，要求精确匹配）。

### 3.3 工具调用语义

**定义 4 (工具调用)**：工具调用是一个映射函数 $T: \mathcal{N} \times \mathcal{A} \rightarrow \mathcal{C} \cup \{\bot\}$：

$$T(name, args) = \begin{cases}
content \in \mathcal{C} & \text{if } name \in \mathcal{N}_{available} \land args \models Schema(name) \\
\bot & \text{otherwise}
\end{cases}$$

其中：
- $\mathcal{N}_{available}$ 是可用工具名称集合
- $Schema(name)$ 是工具的输入 Schema
- $\models$ 表示 JSON Schema 验证关系
- $\mathcal{C}$ 是内容对象集合（文本、图像、音频、资源等）

**工具发现操作**：

$$\mathcal{N}_{available} = \bigcup_{s \in Servers} \{t.name \mid t \in tools/list(s)\}$$

### 3.4 资源定位与订阅

**定义 5 (资源 URI 空间)**：资源空间 $\mathcal{R}$ 由 URI 唯一标识：

$$\mathcal{R} = \{r \mid r.uri \in \mathcal{U}, r \text{ conforms to RFC 3986}\}$$

**资源读取操作**：

$$Read: \mathcal{U} \rightarrow \mathcal{Content} \cup \{error\}$$
$$Read(uri) = \begin{cases}
content & \text{if } uri \in \mathcal{R}_{accessible} \\
ResourceNotFound & \text{if } uri \notin \mathcal{R} \\
AccessDenied & \text{if } uri \notin \mathcal{R}_{accessible}
\end{cases}$$

**订阅机制**：订阅关系可表示为二元组集合 $S \subseteq \mathcal{U} \times \mathcal{C}$：

$$Subscribe(uri, client) \iff (uri, client) \in S$$

资源更新通知：

$$Notify(r) = \{(c, r.updated) \mid (r.uri, c) \in S \land r.modified\_at > t_{last\_sync}\}$$

### 3.5 性能约束模型

**定义 6 (吞吐量约束)**：在给定时间窗口内，MCP 服务器的最大请求处理速率：

$$Throughput_{max} = \min\left(\frac{1}{\sum_{i} P_i \cdot T_i}, \text{RateLimit}_{config}\right)$$

其中：
- $P_i$ 是第 $i$ 类请求的概率分布
- $T_i$ 是第 $i$ 类请求的平均处理时间
- $\text{RateLimit}_{config}$ 是配置的速率限制

**延迟模型**：

$$Latency_{total} = Latency_{transport} + Latency_{protocol} + Latency_{processing}$$

对于不同传输方式：
$$Latency_{transport} = \begin{cases}
T_{stdio} \approx O(1) & \text{本地进程间通信} \\
T_{HTTP} \approx O(RTT) + T_{TLS} & \text{远程 HTTP 调用}
\end{cases}$$

---

## 4. 实现逻辑 (Python 伪代码)

### 4.1 MCP Server 实现

```python
class MCPServer:
    """MCP 服务器实现"""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version
        self.protocol_version = "2025-06-18"
        self.capabilities = {
            "tools": {"listChanged": True},
            "resources": {"subscribe": True, "listChanged": True},
            "prompts": {"listChanged": True},
        }
        self.tools: dict[str, ToolDefinition] = {}
        self.resources: dict[str, ResourceDefinition] = {}
        self.prompts: dict[str, PromptDefinition] = {}

    def register_tool(self, tool: ToolDefinition, handler: callable):
        """注册一个工具及其处理函数"""
        self.tools[tool.name] = (tool, handler)

    def register_resource(self, resource: ResourceDefinition, loader: callable):
        """注册一个资源及其加载函数"""
        self.resources[resource.uri] = (resource, loader)

    def register_prompt(self, prompt: PromptDefinition, generator: callable):
        """注册一个提示模板及其生成函数"""
        self.prompts[prompt.name] = (prompt, generator)

    async def handle_initialize(self, params: dict) -> dict:
        """处理初始化请求，进行能力协商"""
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def handle_tools_list(self, params: Optional[dict] = None) -> dict:
        """处理工具列表请求"""
        tools_list = [
            {
                "name": tool.name,
                "title": tool.title,
                "description": tool.description,
                "inputSchema": tool.input_schema,
            }
            for tool, _ in self.tools.values()
        ]
        return {"tools": tools_list}

    async def handle_tools_call(self, params: dict) -> dict:
        """处理工具调用请求"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool, handler = self.tools[tool_name]
        result = await handler(arguments)

        return {
            "content": [{"type": "text", "text": result}],
            "isError": False
        }
```

### 4.2 MCP Client 实现

```python
class MCPClient:
    """MCP 客户端实现"""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.session: Optional[asyncio.StreamRW] = None
        self._request_id = 0
        self.server_capabilities: dict = {}
        self.server_info: dict = {}

    async def connect_stdio(self, process: asyncio.subprocess.Process) -> bool:
        """通过 stdio 连接到本地 MCP 服务器"""
        self.session = (process.stdin, process.stdout)
        return await self._initialize()

    async def connect_http(self, url: str, auth_token: Optional[str] = None) -> bool:
        """通过 HTTP 连接到远程 MCP 服务器"""
        self.url = url
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        return await self._initialize()

    async def _initialize(self) -> bool:
        """执行初始化握手"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"elicitation": {}},
                "clientInfo": {"name": "example-client", "version": "1.0.0"}
            }
        }

        response = await self._send_request(request)
        result = response.get("result", {})
        self.protocol_version = result.get("protocolVersion")
        self.server_capabilities = result.get("capabilities", {})
        self.server_info = result.get("serverInfo", {})

        await self._send_notification("notifications/initialized")
        return True

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """调用工具"""
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments}
        })
        return response.get("result", {}).get("content")
```

### 4.3 MCP Host 集成

```python
class MCPHost:
    """MCP Host (AI 应用) 实现"""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.clients: dict[str, MCPClient] = {}
        self.tool_registry: dict[str, str] = {}

    async def add_server(self, config: ServerConfig) -> bool:
        """添加并连接 MCP 服务器"""
        client = MCPClient(config.name)
        # 连接逻辑...
        self.clients[config.name] = client
        return True

    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """执行工具调用"""
        server_name = self.tool_registry[tool_name]
        client = self.clients[server_name]
        return await client.call_tool(tool_name, arguments)
```

---

## 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **连接延迟 (Stdio)** | < 10ms | 端到端基准测试 | 本地进程间通信 |
| **连接延迟 (HTTP)** | 50-200ms | 端到端基准测试 | 包含 TLS 握手 |
| **消息延迟 P50 (Stdio)** | 1-5ms | 负载测试 | 单次请求响应 |
| **消息延迟 P50 (HTTP)** | 100-500ms | 负载测试 | 取决于网络 RTT |
| **吞吐量 (Stdio)** | 1000-5000 req/s | 压力测试 | 本地高并发 |
| **吞吐量 (HTTP)** | 100-500 req/s | 压力测试 | 远程 API |
| **并发连接数** | ~100 (Stdio) / ~1000+ (HTTP) | 连接测试 | HTTP 更适合高并发 |
| **内存占用** | ~5-10MB (Stdio) / ~2-5MB (HTTP) | 资源监控 | 单客户端消耗 |

---

## 6. 扩展性与安全性

### 6.1 扩展性分析

**水平扩展能力**：
- **服务器实例扩展**：HTTP 传输支持多实例部署，可通过负载均衡扩展
- **客户端连接扩展**：单 Host 可连接多个独立 Server，无理论上限
- **协议版本扩展**：支持版本协商，但需向后兼容
- **原语扩展**：支持自定义原语（如实验性 Tasks）

**架构扩展点**：
1. 自定义 URI Schemes：支持 `file://`, `git://`, `https://` 及自定义方案
2. 工具输出 Schema：支持结构化输出验证
3. 任务原语 (实验性)：支持长时间运行的异步任务
4. 自定义通知类型：服务器可定义领域特定通知

### 6.2 安全性分析

**安全威胁模型**：

| 威胁类型 | 可能性 | 影响 | 缓解措施 |
|---------|--------|------|---------|
| 恶意工具注入 | 中 | 高 | 人工确认提示 |
| 资源越权访问 | 中 | 高 | URI 验证+ACL |
| 工具参数注入 | 高 | 中 | Schema 验证 |
| 服务器仿冒 | 低 | 高 | OAuth 认证 |
| 中间人攻击 | 低 | 高 | TLS 加密 |
| DoS 攻击 | 中 | 中 | 速率限制 |
| 数据泄露 | 中 | 高 | 最小权限原则 |

**安全控制措施**：

| 机制 | 适用场景 | 安全级别 |
|------|----------|---------|
| **OAuth 2.0** | 远程 HTTP 服务器 | 高 |
| **Bearer Token** | API 密钥认证 | 中 |
| **本地 Stdio** | 本地进程通信 | 中 (依赖 OS 权限) |
| **mTLS** | 企业内网部署 | 高 |

**人工确认机制 (Human-in-the-Loop)**：
MCP 强烈推荐对敏感操作实施人工确认，包括删除、修改、执行、访问等类别。

---

## 参考文献

1. Model Context Protocol Official Documentation. https://modelcontextprotocol.io/
2. MCP Specification (Revision 2025-06-18). https://spec.modelcontextprotocol.io/
3. MCP Python SDK. https://github.com/modelcontextprotocol/python-sdk
4. MCP Reference Servers. https://github.com/modelcontextprotocol/servers
5. JSON-RPC 2.0 Specification. https://www.jsonrpc.org/

---

*报告生成日期：2026-03-01*
