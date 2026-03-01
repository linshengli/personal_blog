# MCP (Model Context Protocol) - 精华整合

> 调研日期：2026-03-01
> 主题：MCP 技术的核心洞察

---

## 1. The One 公式

$$
\text{MCP} = \underbrace{\text{Tools}}_{\text{行动}} + \underbrace{\text{Resources}}_{\text{知识}} + \underbrace{\text{Prompts}}_{\text{模板}} - \underbrace{\text{厂商锁定}}_{\text{零}}
$$

**解读**：MCP 的本质是通过标准化工具、资源和提示三大原语为 AI 提供统一上下文，同时完全消除厂商锁定。

---

## 2. 一句话解释

> MCP 就像 AI 应用的 **USB-C 端口**——无论什么设备（AI 应用）和配件（数据源），只要用统一的 USB-C（MCP 协议）就能即插即用。

---

## 3. 核心架构图

```
┌─────────────────────────────────────────────────────────┐
│                    AI Application (Host)                 │
│                   Claude Desktop / Code                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  Client  │  │  Client  │  │  Client  │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       │            │            │                       │
│       ▼            ▼            ▼                       │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                 │
│  │ Stdio   │  │ Stdio   │  │  HTTP   │                 │
│  │ Local   │  │ Local   │  │ Remote  │                 │
│  └────┬────┘  └────┬────┘  └────┬────┘                 │
└───────┼───────────┼─────────────┼──────────────────────┘
        │           │             │
        ▼           ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌───────────────────────┐
│  Filesystem  │ │  Database    │ │  第三方 API (Sentry)  │
│   Server     │ │   Server     │ │      Server           │
└──────────────┘ └──────────────┘ └───────────────────────┘
```

**简化版**：
```
AI 应用 → MCP 协议层 → 外部系统
        (统一接口)
```

---

## 4. STAR 总结

| 部分 | 内容 |
|------|------|
| **Situation** | LLM 应用需要连接外部系统获取数据和工具，但传统集成方式是点对点的定制开发，每个新数据源都需要单独适配。这导致：1）开发成本高，每个连接都要重新实现；2）维护困难，接口变更影响大；3）厂商锁定严重，如 ChatGPT 插件无法迁移到其他平台。AI 行业急需一个标准化的连接协议。 |
| **Task** | MCP 需要解决的核心问题是：**如何为 AI 应用与外部系统的连接建立统一标准，实现即插即用的能力**。关键要求包括：1）统一的接口抽象（Tools/Resources/Prompts）；2）支持本地和远程两种传输；3）完全开源不绑定厂商；4）安全可控的权限管理。 |
| **Action** | MCP 采用三层架构：1）**数据层**基于 JSON-RPC 2.0 定义消息格式和能力协商机制；2）**传输层**支持 Stdio（本地进程通信）和 HTTP（远程 API）两种传输；3）**原语层**定义三大核心原语——Tools（可调用的函数）、Resources（可读取的数据）、Prompts（预定义交互模板）。实现上采用客户端 - 服务器模式，Host 应用可连接多个独立 Server。 |
| **Result** | MCP 已建立完整生态系统：官方 SDK（Python/TypeScript）、50+ 参考服务器、主流 AI 应用原生支持。开发者可用统一方式连接文件系统、数据库、API 服务，无需关心底层细节。相比传统方案，集成时间从数天缩短到数小时，且完全消除厂商锁定，实现"一次开发，处处运行"。 |

---

## 5. 主流原语速查表

| 原语 | 用途 | 典型操作 | 示例 |
|------|------|---------|------|
| **Tools** | 让 AI 执行动作 | list/call | 查询天气、执行代码、调用 API |
| **Resources** | 让 AI 读取数据 | list/read/subscribe | 读取文件、数据库查询、日志流 |
| **Prompts** | 提供交互模板 | list/get | 代码审查模板、报告生成模板 |

---

## 6. 关键洞察

### 6.1 协议对比

```
┌─────────────────────────────────────────────────────────┐
│                    能力对比矩阵                          │
├──────────────────┬──────────┬──────────┬───────────────┤
│      特性         │   MCP    │ 插件系统  │    直接 API    │
├──────────────────┼──────────┼──────────┼───────────────┤
│ 跨平台           │    ✓     │    ✗     │      ✓        │
│ 即插即用         │    ✓     │    △     │      ✗        │
│ 能力发现         │    ✓     │    △     │      ✗        │
│ 统一认证         │    ✓     │    ✓     │      ✗        │
│ 厂商锁定         │    无     │   高     │     中        │
│ 开发成本         │    低     │    中    │      高       │
└──────────────────┴──────────┴──────────┴───────────────┘
✓ = 支持  △ = 部分支持  ✗ = 不支持
```

### 6.2 传输方式选择

| 传输方式 | 延迟 | 适用场景 | 配置复杂度 |
|---------|------|---------|-----------|
| **Stdio** | 1-5ms | 本地工具、CLI 应用 | 低 |
| **HTTP** | 100-500ms | 云端 API、远程服务 | 中 |

### 6.3 成本估算模型

对于连接 5 个外部系统的 MCP 部署：

| 成本项 | 估算 | 说明 |
|--------|------|------|
| **开发时间** | 1-2 周 | 传统方案需 2-3 个月 |
| **运维成本** | $50-200/月 | 服务器和资源开销 |
| **API 调用** | 按实际用量 | 第三方服务费用 |
| **合计** | **降低 80%+** | 相比定制集成 |

---

## 7. 理解确认问题

**问题**：MCP 为什么不直接管理 LLM 如何调用工具？它和 Function Calling 的本质区别是什么？

**参考答案**：MCP 的设计哲学是**协议层与模型层分离**。MCP 只负责标准化地**提供**上下文（工具定义、数据资源、提示模板），但**不决定**AI 如何使用这些上下文——这是 LLM 自身的决策。Function Calling 是 LLM 内部的推理机制，属于模型层；MCP 是外部系统与 AI 应用之间的桥梁协议，属于协议层。类比：MCP 是 USB 标准，Function Calling 是设备驱动程序。

---

## 8. 快速开始指南

### 8.1 5 分钟 MCP Server 原型

```bash
# 安装 MCP Python SDK
pip install mcp

# 创建简单 server
cat > server.py << 'EOF'
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("demo")

@server.list_tools()
async def list_tools():
    return [Tool(name="hello", description="Say hello", inputSchema={})]

@server.call_tool()
async def call_tool(name, args):
    return [TextContent(type="text", text="Hello from MCP!")]

if __name__ == "__main__":
    asyncio.run(stdio_server.run(server))
EOF

python server.py
```

### 8.2 Claude Desktop 配置

```json
{
  "mcpServers": {
    "demo": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

### 8.3 下一步学习资源

- **官方文档**：https://modelcontextprotocol.io/
- **协议规范**：https://spec.modelcontextprotocol.io/
- **参考服务器**：https://github.com/modelcontextprotocol/servers

---

## 9. 生产检查清单

在将 MCP 投入生产前，请确认：

- [ ] 所有工具输入都有 JSON Schema 验证
- [ ] 敏感操作要求用户确认
- [ ] 实现了速率限制和超时处理
- [ ] 远程连接使用 TLS 加密
- [ ] 凭证使用 OAuth 或安全存储
- [ ] 记录了完整的审计日志
- [ ] 错误消息不包含敏感信息

---

*文档生成日期：2026-03-01*
*字数：约 2,000 字*
