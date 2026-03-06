/**
 * Research Runner
 *
 * Step 2 of the daily pipeline:
 * - Load pending topics from topic-state.json
 * - Run research for each topic in parallel (up to config.research.maxConcurrency)
 * - Each session uses claude-agent-sdk query() with WebSearch/WebFetch/Write tools
 * - Claude autonomously searches, fetches data, and writes the report file
 * - Update topic status in topic-state.json on completion
 *
 * RESEARCH_FRAMEWORK is the body of .claude/technical-research/SKILL.md (frontmatter stripped),
 * embedded directly to avoid runtime file-path dependency in CI.
 */

import fs from 'fs';
import path from 'path';
import { query } from '@anthropic-ai/claude-agent-sdk';
import { config } from '../config.js';
import { updateTopicStatus } from './topic-selector.js';
import type { TopicEntry } from './topic-selector.js';

// ---------------------------------------------------------------------------
// Research framework prompt — sourced from .claude/technical-research/SKILL.md
// (YAML frontmatter stripped; backticks and ${...} placeholders escaped)
// ---------------------------------------------------------------------------
const RESEARCH_FRAMEWORK = `# 技术领域深度调研

对任意技术领域进行结构化、可复现的深度调研，产出四个维度的专业报告和一份精华总结。

## 调研框架

一次完整调研包含四个维度 + 最终整合，按依赖关系分为两个层级：

\`\`\`
第一层（独立维度，可同时开展）:
  ├── 概念剖析  —— 技术原理和架构的深度理解
  ├── 行业情报  —— 最新开源项目、论文、博客的信息收集
  └── 方案对比  —— 多种实现方案的横向评估

第二层（依赖第一层的结果）:
  ├── 精华整合  —— 将三份报告浓缩为可传播的总结
  └── 最终整合  —— 将所有内容（三个维度 + 精华整合）合并为一份完整 md 文件
\`\`\`

以下逐一定义每个维度的目标、方法和输出规范。

---

## 维度一：概念剖析

### 目标

建立对目标技术的深层理解——不仅知道"是什么"，还要理解"为什么这样设计"和"如何实现"。

### 输出规范

#### 1. 定义澄清（约 200 字）

- **通行定义**：该领域最被广泛接受的定义
- **常见误解**：至少列出 3 个容易混淆的认知偏差
- **边界辨析**：与相邻概念（易混淆技术）的核心区别

#### 2. 核心架构

使用 ASCII 图呈现系统架构，标注各组件的职责和数据流向：

\`\`\`
┌──────────────────────────────────────┐
│         {领域} 系统架构               │
├──────────────────────────────────────┤
│  输入 → [处理层] → [存储层] → [输出层] │
│           ↓          ↓                │
│        [辅助组件]  [监控组件]          │
└──────────────────────────────────────┘
\`\`\`

每个组件附带一句话说明其功能。

#### 3. 数学形式化（3-5 个公式）

用 LaTeX 公式描述核心机制，覆盖以下方面：
- 核心算法的数学定义
- 关键性能指标的计算方式
- 效率/成本的量化模型

公式应当反映技术本质，而非堆砌符号。每个公式附带一行自然语言解释。

#### 4. 实现逻辑（Python 伪代码）

\`\`\`python
class CoreSystem:
    """核心类，体现该领域的关键抽象"""
    def __init__(self, config):
        self.component_a = ...  # 说明职责
        self.component_b = ...  # 说明职责

    def core_operation(self, input):
        """核心操作，体现关键算法逻辑"""
        intermediate = self.component_a.process(input)
        result = self.component_b.transform(intermediate)
        return result
\`\`\`

伪代码应当体现架构思想，而非纠缠于实现细节。

#### 5. 性能指标

| 指标 | 典型目标值 | 测量方式 | 说明 |
|------|-----------|---------|------|
| 延迟 | < X ms | 端到端基准测试 | ... |
| 吞吐 | > Y req/s | 负载测试 | ... |
| 准确率 | > Z% | 标准评测集 | ... |
| ... | ... | ... | ... |

#### 6. 扩展性与安全性

- **水平扩展**：如何通过增加节点提升容量
- **垂直扩展**：单节点的优化上限
- **安全考量**：该领域特有的安全风险和防护要点

---

## 维度二：行业情报

### 目标

收集该领域最新的开源项目、学术论文和技术博客，建立对当前生态的全景认知。

### 数据新鲜度要求

所有情报数据必须标注来源和日期。使用以下脚本模板进行实时信息采集：

#### GitHub 项目采集

**必须使用 WebSearch/WebFetch 工具获取最新 GitHub 数据**，不能依赖训练数据中的过时信息。

搜索策略（使用 WebSearch 执行以下查询）：
\`\`\`
"{topic} github stars 2025 2026"
"best {topic} open source libraries github"
"awesome {topic} github"
"site:github.com {topic} stars"
\`\`\`

对搜索结果中发现的重要项目，使用 WebFetch 访问其 GitHub 页面，获取以下实时数据：
- 当前 Stars 数量
- 最近提交日期
- 项目描述和核心功能
- 技术栈信息

筛选标准：
- 最近 6 个月有活跃提交
- Stars > 1000（优先）或 > 500（补充）
- 官方维护或知名团队维护

#### 学术论文采集

搜索查询模板：
\`\`\`
"site:arxiv.org {topic} {current_year-5} {current_year}"
"{topic} NeurIPS OR ICML OR ACL OR AAAI {current_year-5}"
\`\`\`

#### 技术博客采集

搜索查询模板：
\`\`\`
"{topic} tutorial best practices {current_year}"
"{topic} medium.com OR dev.to {current_year-5} {current_year}"
\`\`\`

### 输出规范

#### 1. GitHub 热门项目（15+ 个）

| 项目 | Stars | 核心功能 | 技术栈 | 最后更新 | 链接 |
|------|-------|---------|--------|---------|------|

#### 2. 关键论文（12 篇）

选择策略（按优先级）：
1. **影响力优先**：被引次数高、GitHub 实现多、社区讨论热
2. **时效性次之**：近两年的最新研究
3. **来源权威**：顶级会议 > arXiv 顶会投稿 > arXiv 预印本

推荐比例：
- 经典高影响力论文（奠基性工作）：约 40%
- 最新 SOTA 论文（前沿进展）：约 60%

| 论文 | 作者/机构 | 年份 | 会议/期刊 | 核心贡献 | 影响力指标 | 链接 |
|------|----------|------|----------|---------|-----------|------|

#### 3. 系统化技术博客（10 篇）

选择标准：
- **内容深度**：系列文章、深度教程、架构解析（排除碎片化新闻）
- **作者权威**：官方团队博客、知名专家、一线工程师实践
- **语言平衡**：英文约 70%，中文约 30%

英文推荐来源：OpenAI Blog、Google AI Blog、Anthropic Blog、LangChain Blog、个人专家（Eugene Yan、Chip Huyen、Sebastian Raschka 等）

中文推荐来源：大厂技术博客（美团、阿里、字节等）、知乎专栏、机器之心、PaperWeekly

| 博客标题 | 作者/来源 | 语言 | 类型 | 核心内容 | 日期 | 链接 |
|---------|----------|------|------|---------|------|------|

#### 4. 技术演进时间线

按时间顺序列出该领域的关键里程碑事件，标注事件、发起方和影响。

---

## 维度三：方案对比

### 目标

对该领域的主流实现方案进行系统化横向对比，给出可操作的选型建议。

### 输出规范

#### 1. 历史发展时间线

\`\`\`
{年份} ─┬─ {技术/事件} → {对行业的影响}
{年份} ─┼─ {技术/事件} → {对行业的影响}
{年份} ─┼─ {技术/事件} → {对行业的影响}
{年份} ─┴─ 当前状态：{一句话总结}
\`\`\`

#### 2. N 种方案横向对比（5-7 种）

| 方案 | 原理 | 优点（3+） | 缺点（3+） | 适用场景 | 成本量级 |
|------|------|-----------|-----------|---------|---------|

每个方案的优缺点至少各列 3 条，避免泛泛而谈。

#### 3. 技术细节对比

| 维度 | 方案A | 方案B | 方案C | 方案D | 方案E |
|------|------|------|------|------|------|
| 性能 | | | | | |
| 易用性 | | | | | |
| 生态成熟度 | | | | | |
| 社区活跃度 | | | | | |
| 学习曲线 | | | | | |

#### 4. 选型建议

| 场景 | 推荐方案 | 核心理由 | 预估月成本 |
|------|---------|---------|-----------|
| 小型项目/原型验证 | | | |
| 中型生产环境 | | | |
| 大型分布式系统 | | | |

选型建议应当基于当年最新的技术趋势和生态状况。

---

## 精华整合

### 目标

将三份维度报告浓缩为可快速传播的精华内容。精华整合依赖前三个维度的输出。

### 输出规范

#### 1. The One 公式

用一个"悖论式等式"概括该领域的核心本质：

$$
\\text{领域} = \\underbrace{\\text{核心组件1}}_{\\text{功能}} + \\underbrace{\\text{核心组件2}}_{\\text{功能}} - \\underbrace{\\text{核心损耗}}_{\\text{说明}}
$$

这个公式的目的是帮助读者用一个心智模型记住整个领域。

#### 2. 一句话解释

用费曼技巧写一句话，让非技术背景的人也能理解该领域是做什么的。

#### 3. 核心架构图

\`\`\`
输入 → [层1] → [层2] → [层3] → 输出
        ↓       ↓       ↓
      指标1   指标2   指标3
\`\`\`

#### 4. STAR 总结

| 部分 | 内容要求 | 字数 |
|------|---------|------|
| **Situation**（背景+痛点） | 行业现状和核心挑战 | 100-150 字 |
| **Task**（核心问题） | 技术要解决的关键问题和约束 | 80-120 字 |
| **Action**（主流方案） | 技术演进的关键阶段和核心突破 | 120-180 字 |
| **Result**（效果+建议） | 当前成果、现存局限、实操建议 | 80-120 字 |

#### 5. 理解确认问题

提出 1 个能检验是否真正理解该领域的问题，并给出参考答案。

---

## 质量标准

1. **数据新鲜度**：情报维度的所有数据必须标注来源和日期，优先使用近两年的信息
2. **内容完整性**：每份报告生成后需验证字符数 > 100，确认包含有效内容
3. **格式规范性**：所有报告使用 Markdown 格式，表格对齐、代码块标注语言
4. **总字数**：全部产出合计约 6000 字以上
5. **可操作性**：选型建议需包含具体场景和成本估算，而非笼统的"各有优缺点"`;

// ---------------------------------------------------------------------------

export interface ResearchResult {
  topic: string;
  domain: string;
  success: boolean;
  filePath?: string;
  error?: string;
}

// Keep the topic as-is for the directory name so it matches what the Claude agent
// naturally writes when it sees the topic string in the prompt.
export function topicToDirName(topic: string): string {
  return topic.trim();
}

// Worker-pool concurrency limiter
async function runWithConcurrency<T>(
  items: T[],
  limit: number,
  fn: (item: T, index: number) => Promise<any>
): Promise<{ item: T; result?: any; error?: Error }[]> {
  const results: { item: T; result?: any; error?: Error }[] = new Array(items.length);
  let cursor = 0;

  async function worker() {
    while (cursor < items.length) {
      const i = cursor++;
      try {
        results[i] = { item: items[i], result: await fn(items[i], i) };
      } catch (err) {
        results[i] = { item: items[i], error: err as Error };
      }
    }
  }

  const workerCount = Math.min(limit, items.length);
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}

function ts() {
  return new Date().toTimeString().slice(0, 8);
}

function logTool(tag: string, turn: number, name: string, input: Record<string, any>) {
  let detail = '';
  switch (name) {
    case 'WebSearch': detail = `"${input.query ?? ''}"`.slice(0, 80); break;
    case 'WebFetch':  detail = (input.url ?? '').slice(0, 80); break;
    case 'Write':     detail = input.file_path ?? input.path ?? ''; break;
    case 'Read':      detail = input.file_path ?? input.path ?? ''; break;
    case 'Glob':      detail = input.pattern ?? ''; break;
    case 'Grep':      detail = `"${input.pattern ?? ''}" in ${input.path ?? '.'}`.slice(0, 80); break;
    default:          detail = JSON.stringify(input).slice(0, 60);
  }
  console.log(`  [${ts()}][${tag}] turn ${turn} | ${name} ${detail}`);
}

async function researchOneTopic(entry: TopicEntry, index: number, total: number): Promise<void> {
  const { topic, domain } = entry;
  const dirName = topicToDirName(topic);
  const today = new Date().toISOString().split('T')[0];
  const tag = `${index + 1}/${total} ${domain}`;

  const researchDir = path.join(process.cwd(), config.files.researchDir, dirName);
  const researchFile = path.join(researchDir, `${dirName}-research.md`);

  if (!fs.existsSync(researchDir)) {
    fs.mkdirSync(researchDir, { recursive: true });
  }

  const prompt = `你是一个技术调研专家。请按照以下调研框架对指定主题进行深度调研。

${RESEARCH_FRAMEWORK}

---

## 当前任务

**调研主题：${topic}**
**所属域：${domain}**
**调研日期：${today}**

### 执行步骤

1. 使用 WebSearch 和 WebFetch 工具搜集最新数据（GitHub 项目/stars、论文、博客等）
2. 按照上述调研框架，完成以下四个部分（总字数 6000 字以上）：
   - 第一部分：概念剖析
   - 第二部分：行业情报
   - 第三部分：方案对比
   - 第四部分：精华整合
3. 调研完成后，**使用 Write 工具**将完整报告保存到以下路径：
   \`research/${dirName}/${dirName}-research.md\`

**注意：只保存上述一个文件，无需创建其他附属文件。报告使用中文撰写。**`;

  console.log(`[${ts()}][${tag}] START "${topic}" (timeout ${config.research.timeoutMs / 60000}min)`);

  const abortController = new AbortController();
  let turn = 0;

  const runQuery = async () => {
    for await (const message of query({
      prompt,
      abortController,
      options: {
        allowedTools: ['WebSearch', 'WebFetch', 'Read', 'Write', 'Glob', 'Grep'],
        maxTurns: config.research.maxTurns,
      },
    })) {
      const msg = message as any;

      // Tool calls are in assistant message content blocks
      if (msg?.type === 'assistant' && Array.isArray(msg?.message?.content)) {
        for (const block of msg.message.content) {
          if (block?.type === 'tool_use') {
            turn++;
            logTool(tag, turn, block.name, block.input ?? {});
          }
        }
      }

      // Some SDK versions emit tool_use at the top level
      if (msg?.type === 'tool_use') {
        turn++;
        logTool(tag, turn, msg.name, msg.input ?? {});
      }
    }
  };

  const timeoutGuard = new Promise<never>((_, reject) => {
    setTimeout(() => {
      abortController.abort();
      reject(new Error(`Timeout: exceeded ${config.research.timeoutMs / 60000}min`));
    }, config.research.timeoutMs);
  });

  try {
    await Promise.race([runQuery(), timeoutGuard]);
  } catch (err: any) {
    const reason = abortController.signal.aborted ? `TIMEOUT (${config.research.timeoutMs / 60000}min)` : err.message;
    console.log(`[${ts()}][${tag}] ABORT "${topic}" — ${reason}`);
    throw err;
  }

  if (!fs.existsSync(researchFile)) {
    throw new Error(`Research file not written by agent: ${researchFile}`);
  }

  const bytes = fs.statSync(researchFile).size;
  console.log(`[${ts()}][${tag}] DONE  "${topic}" — ${(bytes / 1024).toFixed(1)} KB, ${turn} tool calls`);
}

export async function runParallelResearch(entries: TopicEntry[]): Promise<ResearchResult[]> {
  // Include 'failed' so re-running the script retries them
  const runnable = entries.filter(e => e.status === 'pending' || e.status === 'failed');

  if (runnable.length === 0) {
    console.log('No topics to research (all completed)');
    return [];
  }

  console.log(
    `Starting parallel research: ${runnable.length} topics, max concurrency ${config.research.maxConcurrency}`
  );

  const rawResults = await runWithConcurrency(
    runnable,
    config.research.maxConcurrency,
    (entry, i) => researchOneTopic(entry, i, runnable.length)
  );

  const results: ResearchResult[] = rawResults.map(({ item, error }) => {
    const dirName = topicToDirName(item.topic);
    const filePath = path.join(
      process.cwd(),
      config.files.researchDir,
      dirName,
      `${dirName}-research.md`
    );
    const success = !error && fs.existsSync(filePath);
    updateTopicStatus(item.topic, success ? 'completed' : 'failed');

    if (!success) {
      console.log(`  [FAIL] [${item.domain}] ${item.topic}`);
      console.log(`         ${error?.message ?? 'unknown error'}`);
    }

    return {
      topic: item.topic,
      domain: item.domain,
      success,
      filePath: success ? filePath : undefined,
      error: error?.message,
    };
  });

  return results;
}
