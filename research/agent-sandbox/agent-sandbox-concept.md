# Agent Sandbox 技术 - 概念剖析

> 调研日期：2026-02-28
> 主题：Agent Sandbox 的核心原理与架构

---

## 1. 定义澄清

### 通行定义

**Agent Sandbox**（Agent 沙箱）是一种为 AI Agent 提供安全代码执行环境的隔离技术。它通过容器化、虚拟化或系统级隔离机制，限制 Agent 可访问的系统资源（文件、网络、进程、系统调用），防止恶意或错误的代码执行对主机系统造成损害。

Agent Sandbox 的核心价值在于在赋予 LLM 代码执行能力的同时，建立可控的安全边界，使 Agent 能够"安全地犯错"。

### 常见误解

| 误解 | 正解 |
|------|------|
| "Sandbox 就是 Docker 容器" | Docker 提供基础隔离，但 Agent Sandbox 需要更细粒度的权限控制和资源限制 |
| "Sandbox 能 100% 防止逃逸" | 任何沙箱都有被绕过的风险，需要多层防御和持续监控 |
| "Sandbox 只防恶意代码" | Sandbox 同样防止意外错误（如 rm -rf、无限循环、资源耗尽） |
| "Browser Sandbox 足够安全" | 浏览器沙箱有 XSS、CSRF 等特有风险，且能力受限 |

### 边界辨析

| 相邻概念 | 核心区别 |
|---------|---------|
| **Agent Sandbox vs 传统 Sandbox** | 传统沙箱防病毒/恶意软件；Agent 沙箱防 LLM 幻觉导致的危险操作 |
| **Agent Sandbox vs CI/CD 环境** | CI/CD 是预定义流程；Agent Sandbox 支持动态、开放式的代码执行 |
| **容器 vs 虚拟机沙箱** | 容器轻量但隔离弱；VM 隔离强但开销大；微 VM 是折中方案 |

---

## 2. 核心架构

```
┌──────────────────────────────────────────────────────────────┐
│                  Agent Sandbox 系统架构                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  LLM 生成   │ ──→ │  代码解析   │ ──→ │  安全审查   │    │
│  │  代码       │     │  & 验证     │     │  (Pre-exe)  │    │
│  └─────────────┘     └─────────────┘     └──────┬──────┘    │
│                                                  ↓         │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐    │
│  │  结果返回   │ ←── │  执行监控   │ ←── │  沙箱执行   │    │
│  │  & 清理     │     │  & 日志     │     │  (隔离环境)  │    │
│  └─────────────┘     └─────────────┘     └──────┬──────┘    │
│                                                  ↓         │
│                    ┌────────────────────────────┘          │
│                    ↓                                       │
│              ┌─────────────┐                               │
│              │  资源限制   │                               │
│              │ - CPU/内存  │                               │
│              │ - 文件系统  │                               │
│              │ - 网络访问  │                               │
│              │ - 系统调用  │                               │
│              └─────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

**组件职责说明：**

| 组件 | 职责 |
|------|------|
| **代码解析** | 解析 LLM 生成的代码，识别潜在危险操作 |
| **安全审查** | 执行前检查，阻止已知的危险模式（如 rm -rf /） |
| **沙箱执行** | 在隔离环境中运行代码，限制资源访问 |
| **执行监控** | 实时监控 CPU、内存、网络使用，检测异常行为 |
| **资源限制** | 定义和执行资源配额（CPU 时间、内存上限、文件访问范围） |

---

## 3. 数学形式化

### 3.1 沙箱逃逸概率模型

对于沙箱防护强度 $P$ 和攻击复杂度 $C$，逃逸概率：

$$P_{\text{escape}} = \frac{1}{1 + e^{k(P - C)}}$$

其中 $k$ 为敏感系数。

**自然语言解释**：防护强度越高、攻击复杂度越高，逃逸概率越低，呈 S 型曲线下降。

### 3.2 资源消耗模型

对于任务 $T$，资源消耗上限：

$$\text{Cost}(T) = \alpha \cdot \text{CPU}_{\text{time}} + \beta \cdot \text{Memory}_{\text{peak}} + \gamma \cdot \text{IO}_{\text{ops}}$$

**自然语言解释**：总成本是 CPU 时间、内存峰值和 IO 操作的加权和。

### 3.3 隔离强度量化

沙箱隔离强度 $S$ 的多维度评估：

$$S = w_1 \cdot S_{\text{fs}} + w_2 \cdot S_{\text{net}} + w_3 \cdot S_{\text{proc}} + w_4 \cdot S_{\text{sys}}$$

其中：
- $S_{\text{fs}}$：文件系统隔离强度
- $S_{\text{net}}$：网络隔离强度
- $S_{\text{proc}}$：进程隔离强度
- $S_{\text{sys}}$：系统调用隔离强度

**自然语言解释**：总体隔离强度是各维度隔离强度的加权和。

### 3.4 执行延迟模型

$$\text{Latency} = T_{\text{setup}} + T_{\text{exec}} + T_{\text{teardown}} + T_{\text{security}}$$

**自然语言解释**：总延迟包括沙箱启动、代码执行、清理和安全检查的时间。

---

## 4. 实现逻辑（Python 伪代码）

```python
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import subprocess
import tempfile
import os

class SecurityLevel(Enum):
    LOW = "low"       # 仅基础容器隔离
    MEDIUM = "medium" # 系统调用过滤 + 资源限制
    HIGH = "high"     # 微 VM + 网络隔离

@dataclass
class ResourceLimits:
    """资源限制配置"""
    cpu_time: int = 30        # CPU 秒
    memory_mb: int = 512      # 内存上限 (MB)
    disk_mb: int = 100        # 磁盘上限 (MB)
    network: bool = False     # 是否允许网络访问
    max_processes: int = 10   # 最大进程数

@dataclass
class ExecutionResult:
    """执行结果"""
    success: bool
    stdout: str
    stderr: str
    exit_code: int
    execution_time: float
    error_type: Optional[str] = None

class SandboxManager:
    """沙箱管理器，体现关键抽象"""

    def __init__(self, security_level: SecurityLevel = SecurityLevel.MEDIUM):
        self.security_level = security_level
        self.resource_limits = ResourceLimits()
        self.allowed_syscalls = self._get_allowed_syscalls()

    def execute(self, code: str, language: str = "python",
                timeout: int = 30) -> ExecutionResult:
        """
        在沙箱中执行代码
        核心流程：审查 → 创建 → 执行 → 监控 → 清理
        """
        # 阶段 1: 安全审查
        if not self._security_review(code):
            return ExecutionResult(
                success=False, stdout="", stderr="Security check failed",
                exit_code=-1, error_type="SECURITY_VIOLATION"
            )

        # 阶段 2: 创建沙箱环境
        sandbox = self._create_sandbox()

        try:
            # 阶段 3: 执行代码 (带监控)
            result = self._execute_in_sandbox(
                sandbox=sandbox,
                code=code,
                language=language,
                timeout=timeout
            )
            return result

        finally:
            # 阶段 4: 清理沙箱
            self._cleanup_sandbox(sandbox)

    def _security_review(self, code: str) -> bool:
        """
        执行前安全审查
        检测危险模式：系统命令、文件操作、网络请求等
        """
        dangerous_patterns = [
            r'rm\s+-rf\s+/',           # 危险删除
            r'sudo\s+',                # 提权
            r'chmod\s+777',            # 开放权限
            r'/etc/passwd',            # 敏感文件
            r'import\s+os\s*;',        # OS 模块 (可选)
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                logger.warning(f"Dangerous pattern detected: {pattern}")
                return False
        return True

    def _create_sandbox(self) -> Any:
        """
        创建隔离环境
        根据安全级别选择不同策略
        """
        if self.security_level == SecurityLevel.HIGH:
            # 使用微 VM (如 Firecracker)
            return self._create_microvm()
        elif self.security_level == SecurityLevel.MEDIUM:
            # 使用 Docker + seccomp
            return self._create_container()
        else:
            # 仅使用临时目录
            return self._create_temp_env()

    def _execute_in_sandbox(self, sandbox: Any, code: str,
                            language: str, timeout: int) -> ExecutionResult:
        """
        在沙箱中执行代码，带资源监控
        """
        start_time = time.time()

        # 准备执行命令
        cmd = self._build_command(sandbox, code, language)

        # 应用资源限制
        limits = self._build_limits(self.resource_limits)

        try:
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=limits,  # UNIX: 设置资源限制
                **sandbox.kwargs
            )

            execution_time = time.time() - start_time

            return ExecutionResult(
                success=process.returncode == 0,
                stdout=process.stdout,
                stderr=process.stderr,
                exit_code=process.returncode,
                execution_time=execution_time
            )

        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False, stdout="", stderr="Timeout",
                exit_code=-2, error_type="TIMEOUT"
            )
        except Exception as e:
            return ExecutionResult(
                success=False, stdout="", stderr=str(e),
                exit_code=-3, error_type="EXECUTION_ERROR"
            )

    def _get_allowed_syscalls(self) -> list:
        """
        获取允许的系统调用列表 (seccomp 配置)
        """
        return [
            'read', 'write', 'exit', 'exit_group',
            'mmap', 'munmap', 'mprotect',
            'brk', 'rt_sigreturn',
        ]

    def _cleanup_sandbox(self, sandbox: Any):
        """清理沙箱环境，释放资源"""
        pass
```

---

## 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| **启动延迟** | < 100ms (容器) / < 1s (微 VM) | 端到端基准测试 | 沙箱创建到就绪的时间 |
| **执行开销** | < 10% 性能损耗 | 对比裸机执行 | 沙箱带来的额外开销 |
| **隔离强度** | 阻止 99%+ 已知逃逸 | CVE 测试集 | 对已知漏洞的防护能力 |
| **资源限制精度** | ±5% | 压力测试 | CPU/内存限制的准确性 |
| **并发支持** | > 100 实例/节点 | 负载测试 | 单节点支持的并发沙箱数 |
| **清理完整性** | 100% | 残留检测 | 执行后无文件/进程残留 |

---

## 6. 扩展性与安全性

### 水平扩展

- **沙箱池**：预创建沙箱池，减少启动延迟
- **调度器**：基于 Kubernetes 的沙箱编排，自动扩缩容
- **状态持久化**：支持沙箱快照，快速恢复执行上下文

### 垂直扩展

- **资源动态调整**：根据任务类型自动调整资源配额
- **多级别隔离**：支持从轻量到重度的多级安全配置
- **硬件加速**：利用 Intel SGX 等可信执行环境 (TEE)

### 安全考量

| 风险 | 防护措施 |
|------|---------|
| **容器逃逸** | seccomp 过滤、rootless 容器、微 VM |
| **侧信道攻击** | 资源隔离、噪声注入、时间限制 |
| **网络攻击** | 网络命名空间隔离、出站流量过滤 |
| **持久化恶意代码** | 执行后彻底清理、只读文件系统 |
| **资源耗尽** | Cgroups 限制、配额管理、超时机制 |
| **供应链攻击** | 镜像签名验证、最小化基础镜像 |
