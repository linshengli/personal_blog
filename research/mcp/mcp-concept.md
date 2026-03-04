# MCP 概念剖析

> 调研日期：2026-03-04
> 调研主题：Model Context Protocol (MCP)

---

## 1. 定义澄清

### 通行定义

**Model Context Protocol (MCP)** 是一个开源标准协议，用于连接 AI 应用程序与外部系统。它提供了一种标准化的方式，让 AI 助手（如 Claude、ChatGPT）能够访问外部数据源（本地文件、数据库）、工具（搜索引擎、计算器）和工作流（专用提示词模板）。

MCP 的核心定位类似于 **AI 应用程序的 USB-C 接口**——正如 USB-C 为电子设备提供统一的连接标准，MCP 为 AI 应用程序与外部系统的连接提供统一协议。

### 常见误解

| 误解 | 正确理解 |
|------|----------|
| "MCP 是一个具体的软件产品" | MCP 是一个协议标准 + 一组 SDK 和参考实现，不是单一软件 |
| "MCP 只支持本地工具调用" | MCP 支持 STDIO（本地）和 Streamable HTTP（远程）两种传输层，可连接远程服务 |
| "MCP 包含 LLM 调用逻辑" | MCP 只负责上下文交换协议，不规定 AI 应用如何使用 LLM |
| "MCP Server 必须本地运行" | MCP Server 可以本地执行（STDIO 传输）或远程部署（HTTP 传输） |
| "MCP 是 Anthropic 专有技术" | MCP 是开源标准，有跨语言 SDK 和多厂商支持 |

### 边界辨析

| MCP | 相邻概念 | 核心区别 |
|-----|----------|----------|
| MCP | OpenAPI/Swagger | OpenAPI 定义 REST API 规范；MCP 定义 AI 上下文交换协议，支持双向通信 |
| MCP | LangChain Tools | LangChain 是 Python/JS 框架，绑定特定语言；MCP 是语言无关的协议标准 |
| MCP | gRPC | gRPC 是通用 RPC 框架；MCP 专注于 AI 场景，内置 Tools/Resources/Prompts 语义 |
| MCP | Function Calling | Function Calling 是 LLM 原生能力；MCP 是独立的协议层，可跨模型复用 |

---

## 2. 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                    MCP 系统架构                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │              MCP Host (AI Application)                  │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│  │  │ MCP Client 1│  │ MCP Client 2│  │ MCP Client 3│     │ │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │ │
│  └─────────┼────────────────┼────────────────┼────────────┘ │
│            │                │                │              │
│            │ Dedicated      │ Dedicated      │ Dedicated    │
│            │ Connection     │ Connection     │ Connection   │
│            ▼                ▼                ▼              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌────────────────┐ │
│  │ MCP Server A    │ │ MCP Server B    │ │ MCP Server C   │ │
│  │ (Local - STDIO) │ │ (Local - STDIO) │ │ (Remote - HTTP)│ │
│  │ Filesystem      │ │ Database        │ │ Sentry         │ │
│  └─────────────────┘ └─────────────────┘ └────────────────┘ │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                      传输层 (Transport)                       │
│  ┌─────────────────────────┬──────────────────────────────┐ │
│  │ STDIO Transport         │ Streamable HTTP Transport    │ │
│  │ - 本地进程通信           │ - 远程 HTTP 通信               │ │
│  │ - 无网络开销             │ - 支持 OAuth 认证             │ │
│  └─────────────────────────┴──────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                      数据层 (Data Layer)                      │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  JSON-RPC 2.0 协议层                                     │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │ Tools        │  │ Resources    │  │ Prompts      │  │ │
│  │  │ (执行函数)    │  │ (数据源)      │  │ (模板)        │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │
│  │  │ Sampling     │  │ Elicitation  │  │ Logging      │  │ │
│  │  │ (LLM 采样)    │  │ (用户输入)    │  │ (日志)        │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 职责 |
|------|------|
| **MCP Host** | AI 应用程序（如 Claude Code、VS Code），协调和管理一个或多个 MCP 客户端 |
| **MCP Client** | 维护与 MCP Server 的连接，获取上下文数据供 Host 使用 |
| **MCP Server** | 提供上下文数据的程序，可本地或远程部署 |
| **Tools** | 可执行函数，AI 应用可调用以执行操作（文件操作、API 调用、数据库查询） |
| **Resources** | 数据源，提供上下文信息（文件内容、数据库记录、API 响应） |
| **Prompts** | 可复用模板，结构化与 LLM 的交互（系统提示词、few-shot 示例） |

---

## 3. 数学形式化

### 3.1 MCP 通信模型

$$
\text{MCP} = (H, \{C_i\}_{i=1}^n, \{S_j\}_{j=1}^m, \mathcal{T}, \mathcal{D})
$$

其中：
- $H$：MCP Host 集合
- $C_i$：第 $i$ 个 MCP Client
- $S_j$：第 $j$ 个 MCP Server
- $\mathcal{T}$：传输层协议（STDIO 或 HTTP）
- $\mathcal{D}$：数据层协议（JSON-RPC 2.0）

### 3.2 能力协商公式

初始化阶段的能力协商：

$$
\text{Capabilities}(C, S) = \text{Capabilities}_C \cap \text{Capabilities}_S
$$

$$
\text{Capabilities}_C = \{elicitation, sampling, \dots\}
$$

$$
\text{Capabilities}_S = \{tools, resources, prompts, notifications, \dots\}
$$

### 3.3 工具调用语义

$$
\text{ToolCall}(t, args) \rightarrow \text{Content}[]
$$

其中：
- $t \in \text{Tools}$：工具名称
- $args \models \text{inputSchema}(t)$：参数符合输入模式
- $\text{Content} = \{text, image, resource, \dots\}$：返回内容类型

### 3.4 通知效率增益

相较于轮询（Polling），通知（Notification）的带宽节省：

$$
\text{Efficiency} = \frac{N_{poll} \times T}{N_{notify}} = \frac{\text{轮询次数} \times \text{周期}}{\text{实际通知次数}}
$$

典型场景下，$\text{Efficiency} \approx 10-100\times$

---

## 4. 实现逻辑（Python 伪代码）

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum

class TransportType(Enum):
    STDIO = "stdio"
    STREAMABLE_HTTP = "streamable_http"

@dataclass
class ServerInfo:
    name: str
    version: str

@dataclass
class ClientInfo:
    name: str
    version: str

@dataclass
class Tool:
    name: str
    title: str
    description: str
    input_schema: Dict[str, Any]

class MCPServer:
    """MCP 服务器 - 提供上下文数据"""
    def __init__(self, name: str, version: str):
        self.info = ServerInfo(name, version)
        self.tools: Dict[str, Tool] = {}
        self.resources: Dict[str, Any] = {}
        self.prompts: Dict[str, Any] = {}
        self.clients: List[MCPClient] = []

    def register_tool(self, tool: Tool):
        """注册工具"""
        self.tools[tool.name] = tool
        self._notify_clients("tools/list_changed")

    def call_tool(self, name: str, args: Dict) -> List[Dict]:
        """执行工具调用"""
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        # 执行工具逻辑...
        return [{"type": "text", "text": "result"}]

class MCPClient:
    """MCP 客户端 - 连接服务器获取上下文"""
    def __init__(self, transport: TransportType, config: Dict):
        self.transport = transport
        self.config = config
        self.session: Optional[MCPSession] = None
        self.server_capabilities: Dict = {}

    async def connect(self) -> MCPSession:
        """建立与服务器的连接"""
        self.session = MCPSession(self.transport, self.config)
        await self.session.initialize()
        return self.session

class MCPSession:
    """MCP 会话 - 管理单次连接的完整生命周期"""
    def __init__(self, transport: TransportType, config: Dict):
        self.transport = transport
        self.config = config
        self.protocol_version = "2025-06-18"
        self.initialized = False

    async def initialize(self):
        """初始化连接 - 能力协商"""
        # 发送 initialize 请求
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {"elicitation": {}},
                "clientInfo": {"name": "example-client", "version": "1.0.0"}
            }
        }
        response = await self._send(request)
        self.server_capabilities = response["result"]["capabilities"]
        self.initialized = True

        # 发送 initialized 通知
        await self._notify("notifications/initialized")

    async def list_tools(self) -> List[Tool]:
        """工具发现"""
        request = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        response = await self._send(request)
        return [Tool(**t) for t in response["result"]["tools"]]

    async def call_tool(self, name: str, args: Dict) -> List[Dict]:
        """工具执行"""
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": name, "arguments": args}
        }
        response = await self._send(request)
        return response["result"]["content"]

class MCPHost:
    """MCP Host - AI 应用程序"""
    def __init__(self):
        self.sessions: Dict[str, MCPSession] = {}
        self.available_tools: List[Tool] = []

    async def add_server(self, name: str, client: MCPClient):
        """添加 MCP 服务器"""
        session = await client.connect()
        self.sessions[name] = session

        # 发现工具
        tools = await session.list_tools()
        self.available_tools.extend(tools)

    async def execute_tool(self, name: str, args: Dict) -> List[Dict]:
        """执行工具调用"""
        # 找到提供该工具的 session
        for session_name, session in self.sessions.items():
            tools = await session.list_tools()
            if any(t.name == name for t in tools):
                return await session.call_tool(name, args)
        raise ValueError(f"Tool not found: {name}")

# 使用示例
async def main():
    host = MCPHost()

    # 添加本地文件系统服务器
    fs_client = MCPClient(TransportType.STDIO, {"command": "mcp-server-filesystem"})
    await host.add_server("filesystem", fs_client)

    # 添加远程 Sentry 服务器
    sentry_client = MCPClient(TransportType.STREAMABLE_HTTP, {
        "url": "https://sentry.io/mcp",
        "token": "xxx"
    })
    await host.add_server("sentry", sentry_client)

    # 执行工具调用
    result = await host.execute_tool("read_file", {"path": "/path/to/file.txt"})
    print(result)
```

---

## 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 连接建立延迟 | < 100ms (STDIO) | 端到端基准测试 | 本地 STDIO 传输无网络开销 |
| 连接建立延迟 | < 500ms (HTTP) | 端到端基准测试 | 远程 HTTP 传输含网络延迟 |
| 工具调用延迟 | < 200ms | 单次调用 P95 | 不含工具本身执行时间 |
| 吞吐量 | > 100 req/s (单连接) | 负载测试 | 取决于传输层和网络条件 |
| 通知延迟 | < 50ms | 事件触发到接收 | 内部进程通信 |
| 消息大小上限 | 16MB | 协议限制 | JSON-RPC 消息体 |

---

## 6. 扩展性与安全性

### 水平扩展

| 扩展方式 | 说明 | 限制 |
|----------|------|------|
| **多服务器连接** | MCP Host 可同时连接多个 MCP Server | 受客户端资源限制 |
| **远程 HTTP 服务器** | 单 MCP Server 可服务多个 Client | 需处理并发和认证 |
| **负载均衡** | 远程服务器可部署多实例 + 负载均衡 | 需会话粘性支持 |

### 垂直扩展

| 优化点 | 上限 |
|--------|------|
| 单服务器工具数量 | 无协议限制，建议 < 1000 |
| 单连接并发请求 | 取决于传输层（HTTP/2 支持多路复用） |
| 单次响应大小 | 16MB（协议限制） |

### 安全考量

| 风险 | 防护措施 |
|------|----------|
| **本地命令注入** | STDIO 传输需验证子进程参数 |
| **远程认证** | HTTP 传输使用 OAuth 2.0 / Bearer Token |
| **数据泄露** | 敏感 Resources 需访问控制 |
| **工具滥用** | 工具调用需权限检查和速率限制 |
| **中间人攻击** | HTTPS 加密传输，证书验证 |

---

## 7. 参考资源

- [MCP 官方文档](https://modelcontextprotocol.io/)
- [MCP 规范](https://modelcontextprotocol.io/specification/latest/)
- [MCP GitHub 组织](https://github.com/modelcontextprotocol)
