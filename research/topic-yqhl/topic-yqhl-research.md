# 大模型推理框架多硬件统一适配与调度技术 — 深度调研报告

> **调研主题**：大模型推理框架多硬件统一适配与调度技术
> **所属领域**：大模型框架
> **调研日期**：2026-05-25
> **调研方法**：Web 检索（GitHub / arXiv / 技术博客）+ 系统化分析

---

# 第一部分：概念剖析

## 1.1 定义澄清

### 通行定义

**大模型推理框架多硬件统一适配与调度技术**，是指在大语言模型（LLM）推理服务场景下，通过统一的软件抽象层和智能调度系统，使同一套推理框架能够无缝运行于不同厂商、不同架构的 AI 加速芯片（包括 NVIDIA GPU、AMD GPU、华为昇腾 NPU、寒武纪 MLU、摩尔线程 MUSA、海光 DCU、Apple Silicon、Intel GPU 等），实现算子适配、内存管理、任务调度和通信协同的跨平台统一。其核心目标是解决当前 AI 芯片生态碎片化带来的 "M 种模型 × N 种芯片" 适配困境，将其降维为 "M + N" 的线性复杂度。

### 常见误解

| 误解 | 纠正 |
|------|------|
| "只要用标准 Python 框架就能自动跨芯片运行" | 不同芯片的算子库、编译器后端和内存模型差异巨大，需要专门的硬件抽象层（HAL）和代码生成 |
| "硬件统一适配 = 性能一致" | 不同芯片的算力特性差异显著（如 GPU 擅长并行计算、NPU 擅长固定流水线），统一适配的目标是"可用"而非"同速" |
| "统一调度只是负载均衡" | 真正的异构调度需要感知算力特性、显存层级、通信拓扑、KV Cache 亲和性等多维信息，远不止分发请求 |

### 边界辨析

与相邻概念的核心区别：

- **CUDA 生态兼容层（如 ZLUDA、AMD ROCm 翻译层）**：目标是让 CUDA 代码不经修改在其他芯片上运行，而多硬件适配框架是原生支持多后端，各后端皆有独立优化的算子实现。
- **云原生推理平台（如 KServe、BentoML）**：关注的是部署编排而非芯片级抽象，不涉及底层算子移植。
- **单一厂商推理引擎（如 TensorRT-LLM）**：深度绑定特定硬件（NVIDIA），不做跨厂商适配；而本主题强调的正是跨厂商统一。

## 1.2 核心架构

```
┌────────────────────────────────────────────────────────────┐
│              大模型推理多硬件统一适配系统架构                      │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  [模型层]  DeepSeek / Qwen / LLaMA / GLM / MoE ...          │
│       │                                                      │
│  [推理引擎层]  SGLang / vLLM / xLLM / TensorRT-LLM          │
│       │            (统一 API 接口层)                         │
│       ▼                                                      │
│  [硬件抽象层 HAL]  ├─ 统一张量描述 ─ 算子注册表 ─ 内存池       │
│       │            ├─ 设备发现/拓扑感知/能力查询               │
│       │            └─ 编译时/运行时 dispatch                  │
│       ▼                                                      │
│  [编译层]  FlagTree / Triton / MLIR / TVM                    │
│       │      (统一 IR → 后端代码生成)                         │
│       ▼                                                      │
│  [算子层]  FlagGems / FlashAttention / cuDNN / ATB           │
│       │      (各芯片原生或 DSL 生成算子)                      │
│       ▼                                                      │
│  [芯片层]  NVIDIA ─ AMD ─ Ascend ─ Cambricon ─ Moore ...    │
│                                                             │
│  ═══════════════════════════════════════════════════════     │
│                                                             │
│  [调度系统]  ┌─ 请求调度器 (KV-aware Routing)                │
│              ├─ PD 分离调度 (Prefill/Decode 独立扩缩容)       │
│              ├─ 异构资源调度 (CPU-GPU-NPU 协同)               │
│              ├─ 弹性扩缩容 (SLA-driven Planner)               │
│              └─ 缓存调度 (KV Cache tiering)                   │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

**各组件说明：**

- **推理引擎层**：对外暴露统一推理 API，屏蔽底层硬件差异。SGLang 和 xLLM 已原生支持多后端
- **硬件抽象层（HAL）**：核心适配层，通过设备描述文件和编译时宏开关实现 "一个代码库，多芯片编译"。xLLM 通过 `ops_api.h` 统一内核接口，编译器用预处理器宏（`USE_CUDA`、`USE_NPU`、`USE_MLU`）选择后端
- **编译层**：Triton/FlagTree 将 DSL 级算子描述编译为各芯片原生代码，新设备适配仅需添加后端代码生成
- **调度系统**：跨硬件的智能任务分发，感知芯片算力差异，实现 Prefill/Decode 分离和 KV Cache 亲和性路由

## 1.3 数学形式化

### 核心公式 1：推理延迟模型

对于包含 $L$ 层、每层隐藏维度 $d$ 的 Transformer 模型，单 Token 推理延迟为：

$$
t_{\text{infer}} = \sum_{l=1}^{L} \left( \frac{4 d^2}{\text{TFLOPS}_{\text{chip}}} + \frac{2 d \cdot \text{seq\_len} \cdot \text{bit\_width}}{\text{BW}_{\text{mem}}} \right)
$$

- 第一项为计算时间（Compute-bound），第二项为访存时间（Memory-bound）
- Prefill 阶段以第一项为主，Decode 阶段以第二项为主
- $\text{TFLOPS}_{\text{chip}}$ 和 $\text{BW}_{\text{mem}}$ 随芯片类型不同而差异巨大（如 H100：989 TFLOPS/3.35 TB/s vs 昇腾 910B：320 TFLOPS/1.6 TB/s）

### 核心公式 2：异构调度开销模型

在异构集群中，将请求 $r$ 分配到芯片 $i$ 的调度代价函数：

$$
C(r, i) = \alpha \cdot \frac{\text{FLOPs}_r}{\text{TFLOPS}_i} + \beta \cdot \frac{\text{KV\_size}_r}{\text{BW}_i} + \gamma \cdot \text{cache\_miss\_penalty}_i + \delta \cdot \text{comm\_cost}(r, i)
$$

- 调度器在运行时最小化 $C(r, i)$，综合考虑计算能力、内存带宽、缓存命中率和通信开销
- 权重 $\alpha, \beta, \gamma, \delta$ 通过在线学习动态调整

### 核心公式 3：PD 分离性能增益

将 Prefill 和 Decode 分离到独立 GPU 池后，系统吞吐量增益为：

$$
S_{\text{speedup}} = \frac{T_{\text{combined}}}{T_{\text{PD}}} = \frac{1}{\frac{T_{\text{P}}}{T_{\text{P}}+T_{\text{D}}} \cdot \frac{1}{k_P} + \frac{T_{\text{D}}}{T_{\text{P}}+T_{\text{D}}} \cdot \frac{1}{k_D}}
$$

其中 $k_P$ 和 $k_D$ 分别为 Prefill 和 Decode 阶段的独立加速比，$T_P$ 和 $T_D$ 为两阶段的原始耗时。NVIDIA Dynamo 的实测表明 $S_{\text{speedup}}$ 可达 2-7x。

### 核心公式 4：适配效率

衡量从模型发布到多芯片可用的适配效率：

$$
E_{\text{adapt}} = \frac{N_{\text{chips}}}{t_{\text{adapt}} \cdot \text{effort}}
$$

- FlagOS 实现的 "Day 0 适配" 使 $t_{\text{adapt}}$ 从数周缩短至 24-36 小时
- 通过 KernelGen 2.0 自动算子生成，$E_{\text{adapt}}$ 相比手工适配提升 10-50x

### 核心公式 5：KV Cache 层级卸载收益

多层级 KV Cache 卸载的等效可用显存：

$$
V_{\text{eff}} = V_{\text{GPU}} + V_{\text{CPU}} \cdot \eta_{\text{CPU}} + V_{\text{SSD}} \cdot \eta_{\text{SSD}}
$$

其中 $\eta_{\text{CPU}}$ 和 $\eta_{\text{SSD}}$ 为跨层级传输的效率因子（通常为 0.3-0.7）。NVIDIA Dynamo 的四级缓存（Hot/Warm/Cool/Cold）可将有效上下文窗口扩展 10x 以上。

## 1.4 实现逻辑（Python 伪代码）

```python
class MultiHardwareInferenceEngine:
    """多硬件统一推理引擎的核心抽象"""
    def __init__(self, config):
        self.device_registry = DeviceRegistry()    # 发现并注册所有可用芯片
        self.hardware_hal = HardwareAbstractionLayer(config)  # 硬件抽象层
        self.scheduler = UnifiedScheduler(config)   # 统一调度器
        self.engine_registry = EngineRegistry()     # 后端引擎注册表（vLLM/SGLang/TRT-LLM）
        self.kv_cache_manager = TieredKVCacheManager(config)  # 多级 KV Cache 管理

    def discover_hardware(self):
        """发现异构芯片，建立拓扑和性能基线"""
        devices = []
        for backend in ['cuda', 'npu', 'musa', 'mlu', 'rocm']:
            d = self.device_registry.probe(backend)
            if d:
                d.benchmark()  # 实测 TFLOPS、带宽、延迟
                devices.append(d)
        return devices

    def serve(self, model_name, request_stream):
        """主推理服务入口 — 统一调度"""
        model = self.hardware_hal.load_model(model_name)
        # 启动物理引擎（自动选择后端）
        engine = self.engine_registry.create(
            model=model,
            devices=self.devices,
            backend=model.preferred_backend
        )
        # 启动调度循环
        for request in request_stream:
            target_device = self.scheduler.route_request(
                request,
                devices=self.devices,
                kv_cache=self.kv_cache_manager
            )
            # KV-aware 路由：优先分配到缓存命中率最高的设备
            result = engine.execute(request, device=target_device)
            yield result

    def route_request(self, request, devices, kv_cache):
        """KV-aware 异构路由决策"""
        scores = []
        for device in devices:
            cache_hit = kv_cache.compute_overlap(
                request.prefix_hash, device.device_id
            )
            compute_cost = request.flops / device.tflops
            comm_cost = self.topology.comm_latency(request.source, device)
            scores.append(
                self.alpha * compute_cost +
                self.beta * cache_hit.penalty() +
                self.gamma * comm_cost
            )
        return devices[argmin(scores)]


class HardwareAbstractionLayer:
    """硬件抽象层 — 统一内存和算子接口"""
    def __init__(self, config):
        self.backends = self._init_backends(config.supported_devices)
        self.memory_pool = UnifiedMemoryPool()

    def _init_backends(self, devices):
        # 每种后端都有独立的 Kernel 实现和内存管理
        return {d: Backend(d) for d in devices}

    def dispatch_op(self, op_name, tensor, backend_type):
        """运行时派发算子到对应硬件后端"""
        kernel = self.backends[backend_type].get_kernel(op_name)
        # 支持编译时和运行时两种 dispatch 模式
        return kernel(tensor)
```

## 1.5 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| Prefill 延迟 (TTFT) | < 200ms / 4K tokens | 单请求端到端测量 | 受芯片计算能力主导，GPU > NPU |
| Decode 延迟 (TPOT) | < 30ms / token | 逐 token 测量 | 受显存带宽主导，依赖 KV Cache 优化 |
| 跨芯片吞吐比 | > 80% of CUDA 原生 | 同模型多芯片对比 | 衡量适配质量：FlagOS 可达 83-94% |
| 适配周期 | < 48 小时（Day 0） | 从模型发布到全量适配 | FlagOS / xLLM 目标值 |
| 异构集群利用率 | > 85% | 算力利用率监控 | NVIDIA Dynamo / CANN Next 实测值 |
| KV Cache 命中率 | > 95% | 缓存请求比例 | SGLang RadixAttention |
| 跨设备通信带宽 | > 90% 线速 | 基准测试 | 依赖 RDMA / NVLink 拓扑 |

## 1.6 扩展性与安全性

### 水平扩展

- **PD 独立扩缩容**：Prefill 和 Decode 节点可独立增加 GPU 实例，Prefill 池按输入负载弹性伸缩（NVIDIA Dynamo Planner 实现 SLA 驱动的自动扩缩容）
- **异构集群混合扩展**：FlagOS 已在千卡级异构集群上完成验证（沐曦+英伟达混合训练，扩展效率超 90%）
- **跨节点 KV Cache 共享**：通过分布式 KV 缓存系统，多节点共享缓存状态，SGLang + NVIDIA Dynamo 已实现全局缓存感知路由

### 垂直扩展

- **单节点多卡张量并行（TP）**：xLLM 支持 8 卡 Tensor Parallelism，SGLang 支持 3D 并行（TP+PP+DP）
- **单卡利用优化**：FP4 量化（Blackwell B200）+ 自适应 grid sizing（SGLang 批大小 1 时比 vLLM 快 1.78x）
- **CPU + GPU/NPU 算力套利**：KTransformers 利用 AMD AMX 优化 CPU 内核做 MoE 专家延迟计算，释放 GPU 显存

### 安全考量

- **多租户隔离**：异构集群中不同租户的请求可能共享同一物理芯片，需确保显存隔离和计算隔离（NVIDIA Dynamo 的 Grove 调度器安全性）
- **数据面安全**：跨芯片通信（NVLink、PCIe、RDMA）需加密保护，防止模型权重泄露
- **供应链安全**：国产芯片可能使用私有通信协议，统一适配层需进行安全审计——FlagOS 的开源策略降低了闭源固件风险

---

# 第二部分：行业情报

## 2.1 GitHub 热门项目

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|
| **vLLM** | ~50k | PagedAttention 推理引擎，V1 引擎默认，支持 TP/PP/EP | Python + CUDA + Triton | 2026-05 | https://github.com/vllm-project/vllm |
| **llama.cpp** | ~75k | CPU/边缘推理，GGUF 量化，支持 Vulkan/SYCL/Metal | C/C++ + GGML | 2026-05 | https://github.com/ggerganov/llama.cpp |
| **SGLang** | ~15k | RadixAttention，原生多硬件（NVIDIA/AMD/Ascend/TPU），FP4 优化 | Python + CUDA + Rust | 2026-05 | https://github.com/sgl-project/sglang |
| **NVIDIA Dynamo** | ~5k | 分布式推理编排层，引擎无关，KV-aware 路由 | Rust + Python | 2026-05 | https://github.com/ai-dynamo/dynamo |
| **FlagOS (众智)** | ~8k | 多芯片统一软件栈，含 FlagGems/FlagTree/FlagScale | Python + Triton | 2026-04 | https://github.com/flagos-ai |
| **xLLM** | ~3k | 京东开源，支持 5 种国产芯片 + NVIDIA，PD 分离 | C++ + Python + CUDA | 2025-11 | https://github.com/jd-opensource/xllm |
| **KTransformers** | ~8k | CPU-GPU 异构 MoE 推理，单卡运行 671B 模型 | Python + C++/CUDA | 2026-05 | https://github.com/kvcache-ai/ktransformers |
| **TensorRT-LLM** | ~10k | NVIDIA 官方推理引擎，FP4/FP8 量化，NVLink 优化 | C++ + CUDA | 2026-05 | https://github.com/NVIDIA/TensorRT-LLM |
| **OpenLM** | ~1.5k | 多平台学术推理框架，PagedAttention + 动态批处理 | Python + CUDA | 2025 | https://github.com/ |
| **FlagGems** | ~2k | 基于 Triton 的跨平台算子库，497 算子 | Python + Triton | 2026-03 | https://github.com/flagos-ai/FlagGems |
| **vLLM-plugin-FL** | ~500 | vLLM 多芯片推理插件，零代码切换后端 | Python | 2026-04 | https://github.com/flagos-ai/vllm-plugin-FL |
| **LMDeploy** | ~5k | 上海 AI Lab 推理框架，支持 NVIDIA + Ascend | Python + C++ | 2026-05 | https://github.com/InternLM/lmdeploy |
| **CANN** | ~3k | 华为昇腾异构计算框架，2026 年全面开源 | C/C++ | 2026-02 | https://gitcode.com/ascend/cann |
| **Text Generation Inference (TGI)** | ~12k | HuggingFace 推理框架，多模型支持 | Rust + Python | 2026-05 | https://github.com/huggingface/text-generation-inference |
| **Prima.cpp** | ~1.2k | 异构家庭集群推理，70B 模型在 4 台消费设备运行 | C++ | 2025-04 | https://github.com/ |

## 2.2 关键论文

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力 | 链接 |
|------|----------|------|----------|---------|-------|------|
| **Efficient Memory Management for Large Language Model Serving with PagedAttention** | Kwon et al. (Stanford) | 2023 | SOSP'23 | 提出 PagedAttention，奠定多硬件推理内存管理基础 | 被引 1500+ | ACM Digital Library |
| **SGLang: Efficient Execution of Structured Language Model Programs** | Zheng et al. (Stanford/LMSYS) | 2024 | OSDI'24 | RadixAttention + 结构化生成，6.4x 吞吐提升 | 被引 500+ | arXiv:2312.07104 |
| **Disaggregated Prefill-Decode Separation** | Patel et al. (Stanford) | 2024 | OSDI'24 (DistServe) | PD 分离架构，4.48x goodput 提升，20x 延迟方差降低 | 被引 300+ | OSDI'24 |
| **KTransformers: CPU-GPU Heterogeneous MoE Inference** | MadSys (清华) | 2025 | SOSP'25 | Expert Deferral + NUMA 感知张量并行，671B 模型单机运行 | 顶会 | arXiv |
| **APEX: Asynchronous CPU-GPU Execution for Constrained GPUs** | 多机构 | 2025/2026 | arXiv | 配置文件驱动的 CPU-GPU 并行调度，T4 上吞吐提升 84-96% | 新论文 | arXiv:2506.03296v4 |
| **SiPipe: CPU-GPU Pipeline Parallel LLM Inference** | 多机构 | 2025 | arXiv | 利用 CPU 处理流水线并行中的辅助计算，提升 2.1x 吞吐 | 新论文 | arXiv:2506.22033 |
| **HeteroScale: Coordinated Autoscaling for Heterogeneous LLM Inference** | 多机构 | 2025 | arXiv | 生产级 PD 分离自动扩缩容，利用率提升 26.6pp | 新论文 | arXiv:2508.19559 |
| **Hyperion: Hierarchical Scheduling in Multi-tier Networks** | 多机构 | 2025 | arXiv | 边缘-云两级调度，延迟降低 52.1% | 新论文 | arXiv:2511.14450 |
| **Hetis: Fine-grained Dynamic Parallelism for Heterogeneous GPU** | 多机构 | 2025 | ACM | 按算子粒度分配异构 GPU 资源，吞吐提升 2.25x | 新论文 | ACM |
| **Agent.xpu: Scheduling Agentic LLM on Heterogeneous SoC** | 多机构 | 2025 | arXiv | 消费级 SoC 上异构执行图编排，延迟降低 91% | 新论文 | arXiv:2506.24045 |
| **Ghidorah: Speculative Decoding on Hetero-Core Edge** | 多机构 | 2025 | arXiv | 投机解码 + 异核模型并行，边缘 7.6x 加速 | 新论文 | arXiv:2505.23219 |
| **Efficient VRAM-Constrained xLM Inference on Clients** | 多机构 | 2026 | MLSys'26 Industry | CPU-GPU 混合分片推理，客户端 TTFT 提升 6.7x | 录用 | arXiv:2604.26334 |

## 2.3 系统化技术博客

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|
| **NVIDIA Dynamo 1.0: Production-Ready Distributed Inference** | NVIDIA 技术博客 | EN | 官方技术 | Dynamo 架构详解：PD 分离、KV-aware 路由、Planner 自动扩缩容 | 2026-03 | https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/ |
| **SGLang + KTransformers CPU Kernel Hybrid Inference** | LMSYS Blog | EN | 深度教程 | KTransformers 集成 SGLang 实现 CPU-GPU 混合 MoE 推理 | 2025-10 | https://www.lmsys.org/blog/2025-10-22-KTransformers/ |
| **SMG: The Case for Disaggregating CPU from GPU** | PyTorch Blog / LightSeek | EN | 深度教程 | Rust 实现的纯 CPU 推理网关，gRPC 通信提升 3.5x 吞吐 | 2025-12 | https://pytorch.org/blog/lightseek-smg/ |
| **FP4 MoE Kernel Engineering on Blackwell** | HuggingFace Blog (apsys) | EN | 技术深度 | SGLang FP4 内核优化：Kernel Fusion + CUTLASS + Adaptive Grid Sizing | 2026-01 | https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison |
| **Disaggregation in LLMs: Evolution of AI Infrastructure** | InfoQ | EN | 深度分析 | PD 分离技术演进全景，成本节约 15-40% | 2025 | https://www.infoq.com/articles/llms-evolution-ai-infrastructure/ |
| **NVIDIA Dynamo: 分布式推理框架的革新者** | 百度开发者 | CN | 深度分析 | Dynamo 架构中文深度解析，PD 分离 + KV-aware 路由 | 2026 | https://developer.baidu.com/article/detail.html?id=6623941 |
| **众智FlagOS 2.0正式发布：32款AI芯片全面适配** | 智源社区/BAAI | CN | 官方发布 | 497 算子、18 家厂商、统一 IR + 算子自动生成 | 2026-03 | https://hub.baai.ac.cn/view/53601 |
| **多芯异构推理新标杆：千亿参数全栈适配** | 百度开发者 | CN | 行业分析 | 国产 OS 完成八类 AI 加速卡适配，延迟波动 ±3% 以内 | 2026 | https://developer.baidu.com/article/detail.html?id=6898947 |
| **2025大语言模型推理框架选型指南** | 百度开发者 | CN | 选型指南 | 三维评估模型（性能/成本/生态），12 框架横评 | 2025 | https://developer.baidu.com/article/detail.html?id=7079182 |
| **大模型推理引擎：主流框架深度解析与选型指南** | CSDN/业内人士 | CN | 深度教程 | vLLM/SGLang/TensorRT-LLM/llama.cpp 四框架对比 | 2025 | 中国通信网 |

## 2.4 技术演进时间线

```
2017 ── Transformer 架构提出 → 奠定大模型基础
2019 ── NVIDIA TensorRT 引入 GPU 推理优化 → 单一硬件推理引擎雏形
2022 ── ChatGPT 发布 → 大模型推理服务需求爆发
     ├── vLLM (PagedAttention) 发布 → 高效显存管理的新范式
2023 ── ├── llama.cpp 发布 → CPU/边缘推理走向大众
     ├── TensorRT-LLM 发布 → NVIDIA 官方推理栈
     └── PagedAttention SOSP'23 → 学术界确认新方向
2024 ── ├── SGLang (RadixAttention) → 结构化生成 + 多硬件支持起步
     ├── DistServe (PD 分离) → 推理架构重大变革
     ├── KTransformers → CPU-GPU 异构推理的突破
     └── 国产芯片适配启动（vLLM-Ascend / MindIE）
2025 ── ├── NVIDIA Dynamo 发布 → 分布式推理编排标准化
     ├── SGLang 支持 Ascend / AMD / Intel → 多硬件原生支持
     ├── xLLM 开源（京东）→ 国产芯片企业级推理
     ├── KTransformers SOSP'25 + 昇腾适配
     ├── SiPipe / APEX / Hyperion → 异构调度论文爆发
     └── FlagOS 统一软件栈初步完成
2026 ── 当前状态：
     ├── FlagOS 2.0：32 款芯片 × 497 算子，Day 0 适配
     ├── NVIDIA Dynamo 1.0 GA：7x 吞吐提升，大规模部署
     ├── CANN 全面开源：华为打通 CUDA 生态
     ├── SGLang BlackWell FP4：1.78x 小批延迟优化
     ├── 中国电信 Triton 统一跨架构：一套代码三芯片
     └── 行业共识：多硬件统一适配 = AI 基础设施必选项
```

---

# 第三部分：方案对比

## 3.1 历史发展时间线

```
2022 ─┬─ 单硬件推理时代：vLLM (NVIDIA only)、TensorRT-LLM (NVIDIA only)
       影响：每个芯片需要独立开发工具链
2023 ─┼─ CPU 推理平民化：llama.cpp 让推理脱离 GPU 依赖
       影响：边缘/消费级硬件推理成为可能
2024 ─┼─ 多硬件探索期：SGLang 开始原生支持多后端，FlagOS 项目启动
       影响：从 CUDA 锁定走向多芯片开放
2025 ─┼─ 爆发之年：NVIDIA Dynamo 统一编排 + KTransformers CPU-GPU 协同 + xLLM 多国产芯片
       ├─ PD 分离成为标配，异构调度论文 10+ 篇
       └─ "Day 0 适配"概念诞生，适配周期从月级→天级
2026 ─┴─ 当前状态：多硬件统一适配进入生产就绪阶段。
        两大阵营成型：
          │ 国际阵营：SGLang + NVIDIA Dynamo (引擎无关编排)
          │ 国产阵营：FlagOS 2.0 (32芯片统一栈) + CANN 开源
        核心矛盾从"能不能适配"转向"性能能否超越原生"
```

## 3.2 7 种方案横向对比

| 方案 | 原理 | 优点 | 缺点 | 适用场景 | 成本量级 |
|------|------|------|------|---------|---------|
| **vLLM + 社区后端** | PagedAttention + V1 引擎，社区维护各芯片后端（vLLM-Ascend, vLLM-ROCm 等） | ① 社区最大（50k stars），生态成熟<br>② 文档完善，案例丰富<br>③ V1 引擎架构灵活 | ① 多硬件支持碎片化<br>② 非 NVIDIA 后端性能差异大<br>③ 无原生 PD 分离调度 | 需要快速上手中型部署 | $0（开源），GPU 按需 |
| **SGLang** | RadixAttention + 原生多硬件 + FP4 内核工程 | ① 原生多后端（NVIDIA/AMD/Ascend/TPU）<br>② KV Cache 效率（6.4x 提升）<br>③ 小批延迟最优（FP4 1.78x 优于 vLLM） | ① 社区规模小于 vLLM<br>② 部分国产芯片支持需要额外插件<br>③ 生态周边工具较少 | 追求极致性能的多硬件部署 | $0（开源），GPU/NPU 按需 |
| **FlagOS (智源)** | 统一软件栈（算子/编译/通信/框架）+ vLLM 插件 | ① 芯片支持最广（32 款）<br>② Day 0 适配（24-36 小时）<br>③ 算子自动生成，性能超原生 | ① 项目较新，成熟度待验证<br>② 学习曲线陡峭（整套栈）<br>③ 国际社区影响力有限 | 国产芯片为主的企业部署 | $0（开源），国产芯片部署 |
| **xLLM (京东)** | 五层架构 + 四种调度器 + 统一算子接口 | ① 国产芯片覆盖最全的单一框架<br>② 企业级生产验证（京东 11.11）<br>③ 动态 Shape + Mooncake 缓存 | ① 非 NVIDIA 芯片为设计重心<br>② 技术报告少，透明度有限<br>③ 框架较新（v0.7） | 国产芯片+中美混合部署 | $0（开源），生产级部署需工程团队 |
| **NVIDIA Dynamo** | 引擎无关分布式编排层 + KV-aware 路由 + Planner | ① 引擎无关（兼容 3 大引擎）<br>② PD 分离调度最优（7x 吞吐）<br>③ Rust 核心 + 生产级成熟度 | ① 需 NVIDIA 原生运行环境<br>② 对国产芯片不支持<br>③ 架构复杂，运维要求高 | NVIDIA 大规模多节点集群 | $0（开源），NVIDIA GPU 集群 |
| **KTransformers** | CPU-GPU 异构 MoE 推理（Expert Deferral） | ① 单卡运行 671B 模型<br>② CPU 算力利用率极高（AMX）<br>③ 支持多厂商 GPU | ① 仅适配 MoE 模型<br>② 依赖 CPU 高带宽内存<br>③ 非 MoE 模型收益有限 | 受限于显存的 MoE 推理场景 | $0（开源），消费级硬件即可 |
| **llama.cpp + GGUF** | CPU 优先推理 + GGUF 量化 + Vulkan/SYCL/Metal | ① 最广泛的硬件支持（CPU/GPU/NPU）<br>② 社区最大（75k stars）<br>③ 量化标准（GGUF）业界通用 | ① 无连续批处理（单请求引擎）<br>② 吞吐量远低于专用框架<br>③ 无 PD 分离/多节点支持 | 个人/边缘/嵌入式场景 | $0（开源），普通硬件即可 |

## 3.3 技术细节对比

| 维度 | vLLM | SGLang | FlagOS | xLLM | NVIDIA Dynamo | KTransformers | llama.cpp |
|------|------|--------|--------|------|---------------|---------------|-----------|
| **多硬件支持** | NVIDIA(原生)/AMD/部分Intel | NVIDIA/AMD/Ascend/Intel/TPU | 32款芯片 | NVIDIA/昇腾/寒武纪/摩尔线程/沐曦 | 仅NVIDIA | NVIDIA/AMD/昇腾/Intel | CPU/Metal/Vulkan/SYCL |
| **PD 分离** | 社区实验 | 原生支持 | 通过FlagScale | 原生4种调度器 | 核心设计 | 不支持 | 不支持 |
| **KV Cache 管理** | PagedAttention | RadixAttention + tiering | FlagGems统一 | 3层Mooncake | 4层tiering + 全局路由 | 标准管理 | 标准管理 |
| **连续批处理** | ✅ | ✅ | ✅ | ✅ | 编排层 | ❌ | ❌ |
| **最大部署规模** | 数百GPU | 400k+ GPU | 千卡级 | 百卡级 | 万卡级 | 单机多卡 | 单机 |
| **性能(NVIDIA)** | 中（基准） | 高（1.1-1.3x vs vLLM） | 83-94% of CUDA | 2.2x vs vLLM-Ascend | 2-7x 提升 | 有限(GPU不足时) | 低 |
| **学习曲线** | 低 | 中 | 高 | 中高 | 高 | 中 | 极低 |
| **生态成熟度** | 极高 | 高 | 中 | 中低 | 高 | 中高 | 极高 |
| **运维复杂度** | 低 | 中 | 高 | 中 | 高 | 低 | 极低 |

## 3.4 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| **个人研究/原型验证** | llama.cpp 或 vLLM（单卡） | 零配置成本，社区支持极佳，快速验证想法 | $0-500（消费级 GPU 或纯 CPU） |
| **小型团队生产部署（单一硬件 NVIDIA）** | vLLM + 可选 SGLang | 生态最成熟，文档/社区完善，运维简单 | $2,000-10,000（1-4 张 H100/A100） |
| **中型生产环境（追求性能）** | SGLang + NVIDIA Dynamo | PD 分离 + KV-aware 路由，6.4x 吞吐提升，延迟最优 | $10,000-50,000（8-32 张 GPU） |
| **国产芯片企业部署** | FlagOS + xLLM 双轨 | FlagOS 统一栈（32 芯片）+ xLLM 企业生产验证，覆盖国产芯片全场景 | $5,000-30,000（国产 NPU/MLU 集群） |
| **大规模分布式推理（万卡级）** | NVIDIA Dynamo + SGLang/TensorRT-LLM | Dynamo 编排层 SLA 驱动 Planner，万卡级弹性调度，7x 吞吐提升 | $50,000-500,000+（NVIDIA 大规模集群） |
| **显存受限 + MoE 模型推理** | KTransformers | 单消费级 GPU + CPU DRAM 运行 671B 模型，Expert Deferral 极致压榨 | $1,000-5,000（1 张 RTX 5090 + 高配 CPU） |
| **混合硬件/跨云部署** | SGLang（原生多后端）+ FlagOS（国产芯片） | 唯一支持国际 + 国产芯片原生统一后端的方案组合 | $10,000-100,000（混合 GPU/NPU 集群） |
| **边缘/嵌入式推理** | llama.cpp（Vulkan/SYCL） | 最广泛的边缘设备覆盖，量化标准统一 | $0-1,000（边缘设备） |

---

# 第四部分：精华整合

## 4.1 The One 公式

$$
\text{多硬件统一推理} = \underbrace{\text{硬件抽象层（HAL）}}_{\text{屏蔽芯片差异}} + \underbrace{\text{统一调度器}}_{\text{感知异构最优路由}} - \underbrace{\text{适配性能损耗}}_{\text{跨芯片算子非原生代价}}
$$

这个公式说明：多硬件统一推理的本质是用软件抽象换取硬件自由度，核心挑战是将适配性能损耗从 "不可接受" 压缩到 "可接受" 甚至 "超越原生"。

## 4.2 一句话解释

**让同一个大模型推理程序，不需要改代码，就能在英伟达 GPU、华为昇腾 NPU、摩尔线程 GPU、甚至你的笔记本电脑上运行，并自动选择最合适的芯片来完成计算。**

## 4.3 核心架构图

```
模型（DeepSeek / Qwen / LLaMA...）
   │
   ▼
统一推理接口（vLLM / SGLang API）
   │
   ▼┌──────────────────────────────────────┐
    │       硬件抽象层（HAL）                │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │
    │  │NVIDIA   │ │Ascend   │ │Moore    │  │
    │  │CUDA后端 │ │NPU后端  │ │MUSA后端 │  │
    │  └────┬────┘ └────┬────┘ └────┬────┘  │
    └──────┼────────────┼────────────┼───────┘
           ▼            ▼            ▼
   ┌───────────────────────────────┐
   │    统一调度器                   │
   │  ├─ KV-aware 路由              │
   │  ├─ PD 分离调度                 │
   │  ├─ 异构资源编排                │
   │  └─ SLA 驱动弹性扩缩容          │
   └───────────────────────────────┘
           │
           ▼
   Prefill池(计算优化)     Decode池(访存优化)
    NVIDIA H100───┐       AMD MI355───┐
    Ascend 910B───┤       Ascend 910B─┤
    Intel Gaudi───┘       Intel Gaudi─┘
```

## 4.4 STAR 总结

### Situation（背景+痛点）

> 大模型进入万亿参数时代，推理计算需求以年增 10x 的速度膨胀。与此同时，AI 芯片生态极度碎片化：NVIDIA 占据高端但供应受限且成本高昂，华为昇腾、寒武纪、摩尔线程、海光等国产芯片性能快速追赶但各自使用私有工具链。企业和云厂商面临 "被单一芯片绑定" 的风险——每种模型都需要为每个芯片单独适配，M×N 的适配组合数呈指数级增长。适配一个模型到一款新芯片通常需要数周到数月，成为 AI 基础设施的"堵点"。

### Task（核心问题）

> 如何构建一套统一的推理软件栈，让大模型在发布当天即完成多芯片适配（Day 0 适配）？如何在异构芯片间实现智能调度，使 Prefill（计算密集）和 Decode（访存密集）两个阶段各自在最优硬件上执行？如何在降低适配成本的同时，使跨芯片推理性能达到原生水平的 80-100%？

### Action（主流方案）

> 业界形成了三条技术演进路径：（1）**国际阵营**以 SGLang + NVIDIA Dynamo 为代表，SGLang 原生支持 NVIDIA/AMD/Ascend/Intel 多后端，RadixAttention 实现极致 KV Cache 效率，Dynamo 提供万卡级 PD 分离编排；（2）**国产阵营**以 FlagOS 2.0 为核心，通过统一算子库（FlagGems，497 算子）+ 统一编译器（FlagTree）+ 自动算子生成（KernelGen 2.0），将适配周期压缩至 24-36 小时，同时使跨芯片性能超越原生方案 6-22%；（3）**异构调度**方向上，Agent.xpu 提出消费级 SoC 上的异构执行图，APEX 和 SiPipe 实现 CPU-GPU 异步并行，KTransformers 通过 Expert Deferral 把 CPU 算力也利用起来，跑出单消费卡上 671B 模型的效果。

### Result（效果+建议）

> 截至 2026 年 5 月，多硬件统一适配已从 "能不能适配" 进入 "性能反超原生" 的新阶段。FlagOS 2.0 覆盖 32 款芯片，xLLM 实现 5 种国产芯片企业级推理，NVIDIA Dynamo 1.0 在万卡集群上取得 7x 吞吐提升。实操建议：以 NVIDIA 为主力算力的团队优先选择 SGLang + Dynamo；以国产芯片为主力或需要混合部署的团队选择 FlagOS + xLLM；个人开发者从 llama.cpp 或 vLLM 入手最快捷。

## 4.5 理解确认问题

**问题**：如果一个推理框架声称支持了 20 款芯片，但每款新芯片适配需要 3 个月手工开发，那么当第 21 款芯片发布时，实际支持时间是多少？真正衡量多硬件适配能力的关键指标应该是什么？

**参考答案**：如果仍需 3 个月手工适配，说明该框架的 "支持" 只是逐个芯片独立完成算子移植，没有真正的硬件抽象层和统一编译器。衡量多硬件适配能力的关键指标应当是 **增量适配周期**（新芯片从发布到可用的时间）和 **自动生成算子覆盖率**（KernelGen 自动生成的算子占所需算子的比例）。FlagOS 2.0 的 Day 0 适配（24-36 小时）和 KernelGen 2.0 的 95%+ 自动生成正确率，才是真正衡量多硬件适配能力的标杆。

---

## 附录：参考资料索引

### GitHub 仓库
1. vLLM: https://github.com/vllm-project/vllm
2. SGLang: https://github.com/sgl-project/sglang
3. NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo
4. FlagOS: https://github.com/flagos-ai
5. xLLM: https://github.com/jd-opensource/xllm
6. KTransformers: https://github.com/kvcache-ai/ktransformers
7. llama.cpp: https://github.com/ggerganov/llama.cpp
8. TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
9. TGI: https://github.com/huggingface/text-generation-inference

### 关键论文
10. PagedAttention: SOSP'23
11. SGLang: OSDI'24
12. DistServe (PD 分离): OSDI'24
13. KTransformers: SOSP'25
14. APEX: arXiv:2506.03296
15. Agent.xpu: arXiv:2506.24045
16. Hyperion: arXiv:2511.14450
17. SiPipe: arXiv:2506.22033
18. HeteroScale: arXiv:2508.19559

### 技术博客与官方文档
19. NVIDIA Dynamo Technical Blog: https://developer.nvidia.com/blog/nvidia-dynamo-1-production-ready/
20. SGLang + KTransformers: https://www.lmsys.org/blog/2025-10-22-KTransformers/
21. SMG Blog: https://pytorch.org/blog/lightseek-smg/
22. Blackwell FP4 Blog: https://huggingface.co/blog/apsys/blackwell-nvfp4-comparison
23. FlagOS 2.0 发布: https://hub.baai.ac.cn/view/53601
24. CANN 开源: https://www.stdaily.com/web/gdxw/2026-02/13/content_474097.html
25. xLLM Technical Report: arXiv:2510.14686

---

> **报告完 | 调研日期：2026-05-25**
