# MCP (Model Context Protocol) 概念剖析报告

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
"""
MCP Server 实现伪代码
展示如何暴露 Tools、Resources、Prompts
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"


@dataclass
class JsonRpcMessage:
    jsonrpc: str = "2.0"
    method: Optional[str] = None
    params: Optional[dict] = None
    id: Optional[int] = None
    result: Optional[dict] = None
    error: Optional[dict] = None

    def is_request(self) -> bool:
        return self.method is not None and self.id is not None

    def is_notification(self) -> bool:
        return self.method is not None and self.id is None

    def is_response(self) -> bool:
        return self.id is not None and (self.result is not None or self.error is not None)

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in self.__dict__.items() if v is not None})

    @classmethod
    def from_json(cls, data: str) -> "JsonRpcMessage":
        obj = json.loads(data)
        return cls(**obj)


@dataclass
class ToolDefinition:
    name: str
    title: str
    description: str
    input_schema: dict
    output_schema: Optional[dict] = None
    annotations: dict = field(default_factory=dict)


@dataclass
class ResourceDefinition:
    uri: str
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    mime_type: Optional[str] = None


@dataclass
class PromptDefinition:
    name: str
    title: Optional[str] = None
    description: Optional[str] = None
    arguments: list = field(default_factory=list)


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
        self.subscriptions: set[str] = set()
        self._request_id = 0

    # ========== 注册原语 ==========

    def register_tool(self, tool: ToolDefinition, handler: callable):
        """注册一个工具及其处理函数"""
        self.tools[tool.name] = (tool, handler)

    def register_resource(self, resource: ResourceDefinition, loader: callable):
        """注册一个资源及其加载函数"""
        self.resources[resource.uri] = (resource, loader)

    def register_prompt(self, prompt: PromptDefinition, generator: callable):
        """注册一个提示模板及其生成函数"""
        self.prompts[prompt.name] = (prompt, generator)

    # ========== 生命周期管理 ==========

    async def handle_initialize(self, params: dict) -> dict:
        """处理初始化请求，进行能力协商"""
        client_caps = params.get("capabilities", {})
        client_info = params.get("clientInfo", {})
        protocol_version = params.get("protocolVersion")

        # 版本协商
        if protocol_version != self.protocol_version:
            # 简单版本协商逻辑，实际实现可能更复杂
            pass

        return {
            "protocolVersion": self.protocol_version,
            "capabilities": self.capabilities,
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    # ========== 工具原语 ==========

    async def handle_tools_list(self, params: Optional[dict] = None) -> dict:
        """处理工具列表请求"""
        cursor = params.get("cursor") if params else None

        # 分页逻辑（简化）
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

        # 验证输入 Schema（简化版）
        if not self._validate_schema(arguments, tool.input_schema):
            raise ValueError("Invalid arguments for tool")

        # 执行工具处理函数
        result = await handler(arguments)

        return {
            "content": [{"type": "text", "text": result}],
            "isError": False
        }

    # ========== 资源原语 ==========

    async def handle_resources_list(self, params: Optional[dict] = None) -> dict:
        """处理资源列表请求"""
        resources_list = [
            {
                "uri": res.uri,
                "name": res.name,
                "title": res.title,
                "description": res.description,
                "mimeType": res.mime_type,
            }
            for res, _ in self.resources.values()
        ]
        return {"resources": resources_list}

    async def handle_resources_read(self, params: dict) -> dict:
        """处理资源读取请求"""
        uri = params.get("uri")

        if uri not in self.resources:
            raise ValueError(f"Resource not found: {uri}")

        resource, loader = self.resources[uri]
        content = await loader(uri)

        return {
            "contents": [
                {
                    "uri": uri,
                    "mimeType": resource.mime_type,
                    "text": content
                }
            ]
        }

    async def handle_resources_subscribe(self, params: dict) -> dict:
        """处理资源订阅请求"""
        uri = params.get("uri")
        self.subscriptions.add(uri)
        return {}

    # ========== 提示原语 ==========

    async def handle_prompts_list(self, params: Optional[dict] = None) -> dict:
        """处理提示列表请求"""
        prompts_list = [
            {
                "name": prompt.name,
                "title": prompt.title,
                "description": prompt.description,
                "arguments": prompt.arguments,
            }
            for prompt, _ in self.prompts.values()
        ]
        return {"prompts": prompts_list}

    async def handle_prompts_get(self, params: dict) -> dict:
        """处理提示获取请求"""
        prompt_name = params.get("name")
        arguments = params.get("arguments", {})

        if prompt_name not in self.prompts:
            raise ValueError(f"Prompt not found: {prompt_name}")

        prompt, generator = self.prompts[prompt_name]
        messages = await generator(arguments)

        return {
            "messages": messages
        }

    # ========== 通知机制 ==========

    async def notify_tools_list_changed(self) -> JsonRpcMessage:
        """发送工具列表变更通知"""
        return JsonRpcMessage(
            method="notifications/tools/list_changed",
            id=None  # 通知不需要 ID
        )

    async def notify_resources_updated(self, uri: str) -> JsonRpcMessage:
        """发送资源更新通知"""
        return JsonRpcMessage(
            method="notifications/resources/updated",
            params={"uri": uri},
            id=None
        )

    # ========== 辅助方法 ==========

    def _validate_schema(self, data: dict, schema: dict) -> bool:
        """简化版 JSON Schema 验证"""
        required = schema.get("required", [])
        properties = schema.get("properties", {})

        # 检查必填字段
        for field in required:
            if field not in data:
                return False

        # 检查类型（简化）
        for key, value in data.items():
            if key in properties:
                expected_type = properties[key].get("type")
                if expected_type == "string" and not isinstance(value, str):
                    return False
                if expected_type == "number" and not isinstance(value, (int, float)):
                    return False
                if expected_type == "boolean" and not isinstance(value, bool):
                    return False

        return True

    async def process_message(self, message: JsonRpcMessage) -> Optional[JsonRpcMessage]:
        """处理传入的 JSON-RPC 消息"""
        if not message.is_request():
            return None

        method = message.method
        params = message.params or {}

        # 方法路由
        handlers = {
            "initialize": self.handle_initialize,
            "tools/list": self.handle_tools_list,
            "tools/call": self.handle_tools_call,
            "resources/list": self.handle_resources_list,
            "resources/read": self.handle_resources_read,
            "resources/subscribe": self.handle_resources_subscribe,
            "prompts/list": self.handle_prompts_list,
            "prompts/get": self.handle_prompts_get,
        }

        if method not in handlers:
            return JsonRpcMessage(
                id=message.id,
                error={"code": -32601, "message": f"Method not found: {method}"}
            )

        try:
            result = await handlers[method](params)
            return JsonRpcMessage(id=message.id, result=result)
        except Exception as e:
            return JsonRpcMessage(
                id=message.id,
                error={"code": -32603, "message": str(e)}
            )
```

### 4.2 MCP Client 实现

```python
"""
MCP Client 实现伪代码
展示如何连接服务器、发现能力、调用工具
"""
import asyncio
import json
from typing import Optional, Any


class MCPClient:
    """MCP 客户端实现"""

    def __init__(self, server_name: str):
        self.server_name = server_name
        self.session: Optional[asyncio.StreamRW] = None
        self._request_id = 0
        self._pending_requests: dict[int, asyncio.Future] = {}
        self.server_capabilities: dict = {}
        self.server_info: dict = {}
        self.protocol_version: str = ""

    # ========== 连接管理 ==========

    async def connect_stdio(self, process: asyncio.subprocess.Process) -> bool:
        """通过 stdio 连接到本地 MCP 服务器"""
        self.session = (process.stdin, process.stdout)
        return await self._initialize()

    async def connect_http(self, url: str, auth_token: Optional[str] = None) -> bool:
        """通过 HTTP 连接到远程 MCP 服务器"""
        # HTTP 连接实现（简化）
        self.url = url
        self.headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else {}
        return await self._initialize()

    async def disconnect(self):
        """断开连接"""
        if self.session:
            stdin, stdout = self.session
            await stdin.drain()
            stdin.close()

    # ========== 生命周期管理 ==========

    async def _initialize(self) -> bool:
        """执行初始化握手"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {
                    "elicitation": {}
                },
                "clientInfo": {
                    "name": "example-client",
                    "version": "1.0.0"
                }
            }
        }

        response = await self._send_request(request)

        if "error" in response:
            raise Exception(f"Initialization failed: {response['error']}")

        result = response.get("result", {})
        self.protocol_version = result.get("protocolVersion")
        self.server_capabilities = result.get("capabilities", {})
        self.server_info = result.get("serverInfo", {})

        # 发送已初始化通知
        await self._send_notification("notifications/initialized")

        return True

    # ========== 原语操作 ==========

    async def list_tools(self) -> list[dict]:
        """获取可用工具列表"""
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list"
        })
        return response.get("result", {}).get("tools", [])

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """调用工具"""
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        })

        if "error" in response:
            raise Exception(f"Tool call failed: {response['error']}")

        result = response.get("result", {})
        if result.get("isError"):
            raise Exception(f"Tool execution error: {result.get('content')}")

        return result.get("content")

    async def list_resources(self) -> list[dict]:
        """获取可用资源列表"""
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/list"
        })
        return response.get("result", {}).get("resources", [])

    async def read_resource(self, uri: str) -> dict:
        """读取资源内容"""
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "resources/read",
            "params": {"uri": uri}
        })
        return response.get("result", {}).get("contents", [])

    async def subscribe_resource(self, uri: str):
        """订阅资源更新"""
        if self.server_capabilities.get("resources", {}).get("subscribe"):
            await self._send_request({
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "resources/subscribe",
                "params": {"uri": uri}
            })

    async def list_prompts(self) -> list[dict]:
        """获取可用提示列表"""
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/list"
        })
        return response.get("result", {}).get("prompts", [])

    async def get_prompt(self, name: str, arguments: Optional[dict] = None) -> list[dict]:
        """获取提示内容"""
        response = await self._send_request({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "prompts/get",
            "params": {"name": name, "arguments": arguments or {}}
        })
        return response.get("result", {}).get("messages", [])

    # ========== 内部方法 ==========

    def _next_id(self) -> int:
        """生成下一个请求 ID"""
        self._request_id += 1
        return self._request_id

    async def _send_request(self, request: dict) -> dict:
        """发送请求并等待响应"""
        request_id = request.get("id")
        future = asyncio.Future()
        self._pending_requests[request_id] = future

        # 发送请求
        await self._send_message(request)

        # 等待响应
        response = await asyncio.wait_for(future, timeout=30.0)
        del self._pending_requests[request_id]

        return response

    async def _send_notification(self, method: str, params: Optional[dict] = None):
        """发送通知（不需要响应）"""
        notification = {
            "jsonrpc": "2.0",
            "method": method
        }
        if params:
            notification["params"] = params
        await self._send_message(notification)

    async def _send_message(self, message: dict):
        """发送 JSON-RPC 消息"""
        data = json.dumps(message).encode() + b"\n"
        if self.session:
            stdin, _ = self.session
            stdin.write(data)
            await stdin.drain()

    async def _receive_loop(self):
        """接收消息循环"""
        _, stdout = self.session
        buffer = b""

        while True:
            chunk = await stdout.read(4096)
            if not chunk:
                break

            buffer += chunk

            # 按行解析 JSON-RPC 消息
            while b"\n" in buffer:
                line, buffer = buffer.split(b"\n", 1)
                if line.strip():
                    await self._handle_message(json.loads(line))

    async def _handle_message(self, message: dict):
        """处理接收到的消息"""
        if "id" in message:
            # 响应消息
            request_id = message["id"]
            if request_id in self._pending_requests:
                self._pending_requests[request_id].set_result(message)
        elif "method" in message:
            # 通知消息
            method = message["method"]
            params = message.get("params", {})
            await self._handle_notification(method, params)

    async def _handle_notification(self, method: str, params: dict):
        """处理通知"""
        if method == "notifications/tools/list_changed":
            await self._on_tools_changed()
        elif method == "notifications/resources/updated":
            await self._on_resources_updated(params.get("uri"))

    async def _on_tools_changed(self):
        """工具列表变更回调"""
        print("Tools list changed, refreshing...")
        tools = await self.list_tools()
        # 更新本地缓存

    async def _on_resources_updated(self, uri: str):
        """资源更新回调"""
        print(f"Resource {uri} updated, refreshing...")
        content = await self.read_resource(uri)
        # 更新本地缓存


# ========== 使用示例 ==========

async def main():
    """MCP 客户端使用示例"""
    client = MCPClient("weather-server")

    # 连接到 stdio 服务器
    process = await asyncio.create_subprocess_exec(
        "python", "-m", "mcp_weather_server",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE
    )

    await client.connect_stdio(process)

    # 发现工具
    tools = await client.list_tools()
    print(f"Available tools: {[t['name'] for t in tools]}")

    # 调用工具
    result = await client.call_tool("get_forecast", {"location": "San Francisco"})
    print(f"Weather forecast: {result}")

    # 读取资源
    resources = await client.list_resources()
    if resources:
        content = await client.read_resource(resources[0]["uri"])
        print(f"Resource content: {content}")

    await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```

### 4.3 MCP Host (AI 应用) 集成

```python
"""
MCP Host 实现伪代码
展示 AI 应用如何协调多个 MCP 客户端
"""
import asyncio
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    transport: str = "stdio"  # or "http"
    url: Optional[str] = None
    auth_token: Optional[str] = None


class MCPHost:
    """MCP Host (AI 应用) 实现"""

    def __init__(self, app_name: str):
        self.app_name = app_name
        self.clients: dict[str, MCPClient] = {}
        self.tool_registry: dict[str, str] = {}  # tool_name -> server_name
        self.resource_registry: dict[str, str] = {}  # resource_uri -> server_name

    async def add_server(self, config: ServerConfig) -> bool:
        """添加并连接 MCP 服务器"""
        client = MCPClient(config.name)

        if config.transport == "stdio":
            process = await asyncio.create_subprocess_exec(
                config.command, *config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE
            )
            success = await client.connect_stdio(process)
        elif config.transport == "http":
            success = await client.connect_http(config.url, config.auth_token)
        else:
            raise ValueError(f"Unknown transport: {config.transport}")

        if success:
            self.clients[config.name] = client
            await self._refresh_registries(config.name)

        return success

    async def remove_server(self, name: str):
        """移除 MCP 服务器"""
        if name in self.clients:
            await self.clients[name].disconnect()
            del self.clients[name]
            # 清理注册表

    async def _refresh_registries(self, server_name: str):
        """刷新工具和资源注册表"""
        client = self.clients[server_name]

        # 更新工具注册
        tools = await client.list_tools()
        for tool in tools:
            self.tool_registry[tool["name"]] = server_name

        # 更新资源注册
        resources = await client.list_resources()
        for res in resources:
            self.resource_registry[res["uri"]] = server_name

    async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
        """执行工具调用"""
        if tool_name not in self.tool_registry:
            raise ValueError(f"Unknown tool: {tool_name}")

        server_name = self.tool_registry[tool_name]
        client = self.clients[server_name]

        return await client.call_tool(tool_name, arguments)

    async def get_available_tools(self) -> list[dict]:
        """获取所有可用工具"""
        all_tools = []
        for client in self.clients.values():
            tools = await client.list_tools()
            all_tools.extend(tools)
        return all_tools

    async def read_resource(self, uri: str) -> dict:
        """读取资源"""
        if uri not in self.resource_registry:
            raise ValueError(f"Unknown resource: {uri}")

        server_name = self.resource_registry[uri]
        client = self.clients[server_name]

        return await client.read_resource(uri)


# ========== 完整使用示例 ==========

async def run_ai_assistant():
    """AI 助手主循环示例"""
    host = MCPHost("My AI Assistant")

    # 连接多个 MCP 服务器
    await host.add_server(ServerConfig(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/home/user"]
    ))

    await host.add_server(ServerConfig(
        name="weather",
        command="python",
        args=["-m", "mcp_weather_server"]
    ))

    await host.add_server(ServerConfig(
        name="database",
        transport="http",
        url="https://api.example.com/mcp",
        auth_token="secret_token"
    ))

    # 获取所有可用工具
    tools = await host.get_available_tools()
    print("Available tools:")
    for tool in tools:
        print(f"  - {tool['name']}: {tool['description']}")

    # AI 助手决策调用工具
    # 在实际实现中，这部分由 LLM 决策
    weather_result = await host.execute_tool("get_forecast", {
        "location": "San Francisco"
    })
    print(f"Weather: {weather_result}")

    # 读取文件资源
    file_content = await host.read_resource("file:///home/user/project/README.md")
    print(f"File content: {file_content}")


if __name__ == "__main__":
    asyncio.run(run_ai_assistant())
```

---

## 5. 性能指标

| 指标 | 描述 | Stdio 传输 | HTTP 传输 | 备注 |
|------|------|-----------|-----------|------|
| **连接延迟** | 建立连接的时间 | < 10ms | 50-200ms | HTTP 包含 TLS 握手 |
| **消息延迟 (P50)** | 单次请求响应时间 | 1-5ms | 100-500ms | 取决于网络 RTT |
| **消息延迟 (P99)** | 长尾延迟 | < 50ms | < 2s | 网络波动影响 |
| **吞吐量** | 每秒处理请求数 | 1000-5000 req/s | 100-500 req/s | 取决于服务器实现 |
| **并发连接数** | 单服务器最大连接 | ~100 | ~1000+ | HTTP 更适合高并发 |
| **消息大小限制** | 单次消息最大尺寸 | ~1MB | ~10MB | 取决于传输配置 |
| **内存占用** | 单客户端内存消耗 | ~5-10MB | ~2-5MB | 进程 vs 网络连接 |
| **CPU 开销** | 每请求处理开销 | < 0.1ms | 1-5ms | TLS 加解密开销 |

### 5.1 性能优化建议

| 优化方向 | 具体措施 | 预期收益 |
|----------|----------|----------|
| **连接池** | HTTP 复用连接，避免重复 TLS 握手 | 减少 50-100ms/请求 |
| **批处理** | 批量获取工具/资源列表 | 减少网络往返次数 |
| **缓存** | 缓存工具列表、资源元数据 | 减少 80% 重复请求 |
| **流式响应** | 大资源分块传输 | 降低内存峰值 |
| **压缩** | GZIP 压缩大消息体 | 减少 60-80% 带宽 |

---

## 6. 扩展性与安全性分析

### 6.1 扩展性分析

#### 6.1.1 水平扩展能力

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Host                                 │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ Client  │  │ Client  │  │ Client  │  │ Client  │  ...       │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │           │           │           │                     │
│       ▼           ▼           ▼           ▼                     │
│  ┌─────────────────────────────────────────────────┐            │
│  │           Load Balancer / API Gateway           │            │
│  └─────────────────────────────────────────────────┘            │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐                 │
│   │ Server 1 │    │ Server 2 │    │ Server N │                 │
│   └──────────┘    └──────────┘    └──────────┘                 │
└─────────────────────────────────────────────────────────────────┘
```

**扩展能力评估**：

| 维度 | 扩展能力 | 说明 |
|------|----------|------|
| **服务器实例扩展** | ★★★★★ | HTTP 传输支持多实例部署 |
| **客户端连接扩展** | ★★★★☆ | 单 Host 可连接多 Server |
| **协议版本扩展** | ★★★☆☆ | 版本协商机制存在但需兼容 |
| **原语扩展** | ★★★★☆ | 支持自定义原语 (如 Tasks) |

#### 6.1.2 架构扩展点

1. **自定义 URI Schemes**: 支持 `file://`, `git://`, `https://` 及自定义方案
2. **工具输出 Schema**: 支持结构化输出验证
3. **任务原语 (实验性)**: 支持长时间运行的异步任务
4. **自定义通知类型**: 服务器可定义领域特定通知

### 6.2 安全性分析

#### 6.2.1 安全威胁模型

```
┌─────────────────────────────────────────────────────────────────┐
│                      安全威胁矩阵                                │
├──────────────────┬──────────────┬───────────────┬───────────────┤
│     威胁类型      │    可能性     │     影响      │    缓解措施    │
├──────────────────┼──────────────┼───────────────┼───────────────┤
│ 恶意工具注入     │     中        │      高       │ 人工确认提示   │
│ 资源越权访问     │     中        │      高       │ URI 验证+ACL   │
│ 工具参数注入     │     高        │      中       │ Schema 验证    │
│ 服务器仿冒       │     低        │      高       │ OAuth 认证     │
│ 中间人攻击       │     低        │      高       │ TLS 加密       │
│ DoS 攻击         │     中        │      中       │ 速率限制       │
│ 数据泄露         │     中        │      高       │ 最小权限原则   │
└──────────────────┴──────────────┴───────────────┴───────────────┘
```

#### 6.2.2 安全控制措施

**认证与授权**：

| 机制 | 适用场景 | 安全级别 |
|------|----------|----------|
| **OAuth 2.0** | 远程 HTTP 服务器 | 高 |
| **Bearer Token** | API 密钥认证 | 中 |
| **本地 Stdio** | 本地进程通信 | 中 (依赖 OS 权限) |
| **mTLS** | 企业内网部署 | 高 |

**输入验证**：
```python
# MCP 推荐的安全实践
def validate_tool_input(arguments: dict, schema: dict) -> bool:
    # 1. JSON Schema 验证
    if not jsonschema.validate(arguments, schema):
        return False

    # 2. 长度限制
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > MAX_INPUT_LENGTH:
            return False

    # 3. 注入检测
    dangerous_patterns = [';', '|', '`', '$(', '${']
    for value in arguments.values():
        if isinstance(value, str):
            for pattern in dangerous_patterns:
                if pattern in value:
                    log_suspicious_input(value)
                    return False

    return True
```

**人工确认机制 (Human-in-the-Loop)**：

MCP 强烈推荐对敏感操作实施人工确认：

```python
SENSITIVE_OPERATIONS = {
    "delete": ["file_delete", "database_drop"],
    "modify": ["file_write", "config_update"],
    "execute": ["shell_command", "code_execution"],
    "access": ["credentials_read", "pii_access"],
}

async def require_user_confirmation(tool_name: str, arguments: dict) -> bool:
    """要求用户确认敏感操作"""
    for category, operations in SENSITIVE_OPERATIONS.items():
        if tool_name in operations:
            confirmation = await prompt_user(
                f"⚠️ {category.upper()} operation requested:\n"
                f"Tool: {tool_name}\n"
                f"Arguments: {arguments}\n"
                f"Do you want to proceed? (yes/no)"
            )
            return confirmation.lower() == "yes"
    return True  # 非敏感操作自动通过
```

#### 6.2.3 安全最佳实践清单

| 角色 | 安全措施 |
|------|----------|
| **Server 开发者** | • 所有输入进行 Schema 验证<br>• 实现速率限制<br>• 记录审计日志<br>• 最小权限原则 |
| **Client 开发者** | • 敏感操作人工确认<br>• 工具调用前向用户展示<br>• 验证服务器证书<br>• 安全存储凭证 |
| **Host 开发者** | • 服务器来源验证<br>• 工具权限分级<br>• 异常行为检测<br>• 操作审计追踪 |
| **终端用户** | • 仅信任已知服务器<br>• 定期审查权限<br>• 监控异常行为 |

### 6.3 扩展性评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **协议可扩展性** | 8/10 | 支持自定义原语和通知 |
| **传输可扩展性** | 9/10 | 支持 Stdio/HTTP，可扩展更多 |
| **服务器部署扩展** | 9/10 | 支持本地和云端部署 |
| **客户端并发** | 8/10 | 单 Host 多 Client 架构 |
| **安全控制扩展** | 7/10 | 基础认证完善，细粒度授权待加强 |

---

## 参考文献

1. Model Context Protocol Official Documentation. https://modelcontextprotocol.io/
2. MCP Specification (Revision 2025-06-18). https://spec.modelcontextprotocol.io/
3. MCP Python SDK. https://github.com/modelcontextprotocol/python-sdk
4. MCP Reference Servers. https://github.com/modelcontextprotocol/servers
5. JSON-RPC 2.0 Specification. https://www.jsonrpc.org/

---

*报告生成日期：2026-03-01*
