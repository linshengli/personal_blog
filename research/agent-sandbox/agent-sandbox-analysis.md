# Agent Sandbox 技术 - 方案对比

> 调研日期：2026-02-28
> 主题：Agent Sandbox 的主流方案横向评估

---

## 1. 历史发展时间线

```
2008 ─┬─ LXC 容器 → 操作系统级虚拟化奠基
      ├─ Docker 发布 → 容器技术普及
2017 ─┼─ gVisor 开源 → 应用级内核沙箱新范式
2018 ─┼─ Firecracker → 微 VM 技术成熟
2022 ─┼─ Code Interpreter → LLM 代码执行标准化
2023 ─┼─ E2B/OpenHands → AI Agent 专用沙箱兴起
2024 ─┼─ 安全研究爆发 → 顶会论文关注逃逸风险
2025 ─┼─ TEE+ 沙箱 → 硬件级隔离融合
2026 ─┴─ 当前状态：标准化和合规化阶段
```

---

## 2. 六种方案横向对比

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **Docker 容器** | Linux 命名空间+Cgroups 隔离 | - 启动快 (<100ms)<br>- 资源开销低<br>- 生态成熟<br>- 易部署 | - 隔离强度有限<br>- 容器逃逸风险<br>- 共享内核攻击面 | 开发测试、低风险场景 | 低 |
| **gVisor** | 用户态应用级内核，拦截系统调用 | - 强隔离 (系统调用过滤)<br>- 兼容 OCI<br>- 细粒度安全策略 | - 性能损耗 (10-20%)<br>- 不支持所有 syscall<br>- 配置复杂 | 中等风险、多租户 | 中 |
| **Firecracker 微 VM** | 轻量虚拟机，KVM 硬件虚拟化 | - VM 级隔离<br>- 启动快 (<1s)<br>- 内存开销低<br>- AWS 验证 | - 需要 KVM 支持<br>- 工具链复杂<br>- 调试困难 | 高风险、生产环境 | 中 - 高 |
| **Kata Containers** | VM 容器化，OCI 兼容运行时 | - 透明替换 Docker<br>- VM 隔离强度<br>- 社区支持强 | - 性能开销 (15-25%)<br>- 配置复杂<br>- 兼容性问题 | 企业级、混合负载 | 中 - 高 |
| **E2B Runtime** | 专用 AI Agent 沙箱，基于微 VM | - AI 场景优化<br>- 多语言支持<br>- 内置监控<br>- 开发者友好 | - 商业产品<br>- 定制有限<br>- 依赖第三方 | AI Agent 代码执行 | 中 (SaaS 定价) |
| **Browser Sandbox** | 浏览器 WebContainer/IFrame 隔离 | - 零服务器成本<br>- 天然网络隔离<br>- 即时启动 | - 能力受限 (无系统调用)<br>- XSS 风险<br>- 仅 JS/Web 技术栈 | 前端代码、教学演示 | 低 |

---

## 3. 技术细节对比

| 维度 | Docker | gVisor | Firecracker | Kata Containers | E2B | Browser |
|------|--------|--------|-------------|-----------------|-----|---------|
| **启动延迟** | <100ms | <200ms | <500ms | <800ms | <1s | <50ms |
| **内存开销** | ~5MB | ~20MB | ~5MB | ~30MB | ~10MB | ~50MB |
| **隔离强度** | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| **性能损耗** | <5% | 10-20% | 5-10% | 15-25% | 5-15% | N/A |
| **易用性** | 高 | 中 | 中 | 中低 | 高 | 高 |
| **生态成熟度** | 非常高 | 中 | 高 | 中 | 中 | 高 |
| **并发密度** | >1000 | >500 | >200 | >100 | >100 | N/A |
| **网络隔离** | 中 | 中 | 高 | 高 | 高 | 高 |
| **多语言支持** | 全部 | 全部 | 全部 | 全部 | 全部 | JS 为主 |

---

## 4. 各方案核心特性详解

### 4.1 Docker 容器

**核心机制**：Linux 命名空间 + Cgroups + seccomp

```bash
# 基础容器运行
docker run --rm \
  --memory="512m" \
  --cpus="1.0" \
  --security-opt seccomp=default.json \
  python:3.11-slim python script.py
```

**适用场景**：
- 开发测试环境
- 低风险代码执行
- 快速原型验证

---

### 4.2 gVisor

**核心机制**：用户态内核 Sentry，拦截系统调用

```bash
# 使用 gVisor 运行容器
docker run --rm \
  --runtime=runsc \
  --memory="512m" \
  python:3.11-slim python script.py
```

**适用场景**：
- 多租户 SaaS
- 中等风险代码
- 需要 OCI 兼容

---

### 4.3 Firecracker 微 VM

**核心机制**：KVM 硬件虚拟化，最小化 VMM

```rust
// Firecracker API 示例
let vm = FirecrackerVM::new()
    .kernel("vmlinux")
    .rootfs("rootfs.ext4")
    .memory(512)
    .vcpu(1)
    .start();
```

**适用场景**：
- 高风险代码执行
- 生产环境隔离
- serverless 平台

---

### 4.4 Kata Containers

**核心机制**：每个容器一个轻量 VM

```bash
# 使用 Kata 运行时
docker run --rm \
  --runtime=kata \
  ubuntu bash -c "echo hello"
```

**适用场景**：
- 企业级应用
- 混合负载
- 需要透明替换 Docker

---

### 4.5 E2B Runtime

**核心机制**：专为 AI Agent 设计的微 VM 沙箱

```typescript
// E2B SDK 使用
import { Sandbox } from 'e2b';
const sandbox = await Sandbox.create();
const result = await sandbox.runCode('print("Hello")');
```

**适用场景**：
- AI Agent 代码执行
- 多语言支持需求
- 需要内置监控

---

### 4.6 Browser Sandbox

**核心机制**：WebContainer API / IFrame 隔离

```javascript
// WebContainer 示例
const wc = await WebContainer.boot();
await wc.fs.writeFile('index.js', 'console.log("Hi")');
const proc = await wc.spawn('node', ['index.js']);
```

**适用场景**：
- 前端代码演示
- 教学环境
- 零服务器成本需求

---

## 5. 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **开发测试/原型** | Docker | 快速启动，生态成熟，零学习成本 | $50-200 (基础设施) |
| **AI Agent 代码执行** | E2B / OpenHands Sandbox | AI 场景优化，内置安全策略 | $200-1000 (SaaS+ 基础设施) |
| **多租户 SaaS** | gVisor / Firecracker | 强隔离，防止租户间攻击 | $500-2000 |
| **企业级生产** | Kata Containers / Firecracker | VM 级安全，合规认证支持 | $2000-10000+ |
| **前端代码演示** | Browser Sandbox (WebContainer) | 零服务器成本，即时启动 | $0-100 (CDN) |
| **混合负载** | Kubernetes + 多种运行时 | 灵活调度，按风险分级 | $1000-5000 |

---

## 6. 选型决策树

```
                        ┌─────────────────┐
                        │ 代码执行需求？   │
                        └────────┬────────┘
                                 │
            ┌────────────┬───────┴───────┬────────────┐
            ↓            ↓               ↓            ↓
       ┌────────┐  ┌────────┐    ┌──────────┐  ┌────────┐
       │前端 JS │  │AI Agent│    │多租户    │  │企业级  │
       └───┬────┘  └───┬────┘    └─────┬────┘  └───┬────┘
           ↓           ↓               ↓           ↓
       ┌───────┐  ┌──────────┐   ┌─────────┐  ┌────────┐
       │Browser│  │E2B/      │   │gVisor/  │  │Kata/   │
       │Sandbox│  │OpenHands │   │Firecra. │  │Firecra │
       └───────┘  └──────────┘   └─────────┘  └────────┘
```
