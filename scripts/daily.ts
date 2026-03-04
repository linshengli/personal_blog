/**
 * 每日定时任务脚本
 * 功能：自动选择领域进行调研并更新博客内容
 */

import fs from 'fs';
import path from 'path';

// 预定义的调研领域列表
const RESEARCH_DOMAINS = [
  'AI Agent',
  'MCP (Model Context Protocol)',
  'RAG (Retrieval-Augmented Generation)',
  'LLM Fine-tuning',
  'Prompt Engineering',
  'Vector Database',
  'AI Workflow',
  'Code Generation',
  'Multi-Agent System',
  'AI Memory',
  'Function Calling',
  'Semantic Search',
  'Knowledge Graph',
  'AI Evaluation',
  'Edge AI',
];

// 研究子主题模板
const RESEARCH_TOPICS: Record<string, string[]> = {
  'AI Agent': ['Agent 架构设计', 'Agent 规划能力', 'Agent 工具使用', 'Agent 评估方法'],
  'MCP (Model Context Protocol)': ['MCP 协议规范', 'MCP Server 开发', 'MCP 应用场景', 'MCP 安全性'],
  'RAG (Retrieval-Augmented Generation)': ['RAG 检索优化', 'RAG 索引策略', 'RAG 生成优化', 'RAG 评估指标'],
  'LLM Fine-tuning': ['LoRA 微调', '全量微调', '指令微调', 'RLHF 优化'],
  'Prompt Engineering': ['思维链提示', 'few-shot prompting', 'prompt 优化技巧', 'prompt 安全'],
  'Vector Database': ['向量索引算法', '相似度搜索', '向量压缩', '分布式向量检索'],
  'AI Workflow': ['工作流编排', '任务分解', '并行执行', '错误处理'],
  'Code Generation': ['代码理解', '代码补全', '代码重构', '测试生成'],
  'Multi-Agent System': ['Agent 协作', 'Agent 通信协议', '任务分配', '冲突解决'],
  'AI Memory': ['短期记忆', '长期记忆', '记忆检索', '记忆更新'],
  'Function Calling': ['函数定义', '参数提取', '结果解析', '错误处理'],
  'Semantic Search': ['语义理解', '查询扩展', '结果排序', '多模态搜索'],
  'Knowledge Graph': ['知识抽取', '知识融合', '知识推理', '图神经网络'],
  'AI Evaluation': ['评估指标', '基准测试', '人工评估', '自动化评估'],
  'Edge AI': ['模型压缩', '边缘推理', '联邦学习', '隐私保护'],
};

console.log('执行每日定时任务...');
console.log(`当前时间：${new Date().toISOString()}`);

/**
 * 根据日期选择一个领域（确定性选择，基于日期种子）
 */
function selectDailyTopic(): string {
  const today = new Date();
  const dayOfYear = Math.floor(
    (today.getTime() - new Date(today.getFullYear(), 0, 0).getTime()) / (1000 * 60 * 60 * 24)
  );
  const domainIndex = dayOfYear % RESEARCH_DOMAINS.length;
  return RESEARCH_DOMAINS[domainIndex];
}

/**
 * 根据领域选择具体的研究主题
 */
function selectResearchTopics(domain: string): string[] {
  const topics = RESEARCH_TOPICS[domain] || ['基础概念', '技术原理', '应用场景', '最佳实践'];
  const today = new Date();
  const dayOfMonth = today.getDate();
  // 每天选择 2 个主题
  const startIndex = dayOfMonth % topics.length;
  const endIndex = (startIndex + 2) % topics.length;

  if (endIndex > startIndex) {
    return topics.slice(startIndex, endIndex);
  } else {
    return [topics[startIndex], topics[(startIndex + 1) % topics.length]];
  }
}

/**
 * 检查研究目录是否存在
 */
function ensureResearchDir(domain: string): string {
  const researchDir = path.join(process.cwd(), 'research');
  const domainDir = path.join(researchDir, domain.toLowerCase().replace(/[^a-z0-9]/g, '-'));

  if (!fs.existsSync(domainDir)) {
    fs.mkdirSync(domainDir, { recursive: true });
    console.log(`创建研究目录：${domainDir}`);
  }

  return domainDir;
}

/**
 * 生成调研日志文件名
 */
function getLogFileName(domain: string): string {
  const today = new Date();
  const dateStr = today.toISOString().split('T')[0];
  return `daily-log-${dateStr}.md`;
}

/**
 * 生成每日调研日志内容
 */
function generateDailyLog(domain: string, topics: string[]): string {
  const today = new Date();
  const dateStr = today.toISOString().split('T')[0];

  return `# 每日调研日志 - ${domain}

> 调研日期：${dateStr}
> 调研主题：${topics.join(', ')}

---

## 今日调研概览

### 领域：${domain}
### 聚焦主题：
${topics.map(t => `- ${t}`).join('\n')}

---

## 调研内容

### 1. 概念剖析

（待填充：核心概念定义、常见误解、边界辨析）

### 2. 行业情报

#### GitHub 热门项目
（待填充：相关开源项目）

#### 关键论文
（待填充：重要学术论文）

#### 技术博客
（待填充：高质量技术文章）

### 3. 方案对比

（待填充：不同方案的技术对比）

### 4. 精华整合

（待填充：核心公式、一句话解释、架构图、STAR 总结）

---

## 明日计划

- [ ] 继续深入研究 ${topics[0]}
- [ ] 探索 ${topics[1]} 的实际应用
- [ ] 整理调研笔记

---

*日志生成日期：${dateStr}*
`;
}

/**
 * 主函数
 */
async function runDailyTask() {
  try {
    // 1. 选择今日调研领域
    const domain = selectDailyTopic();
    console.log(`\n📌 今日调研领域：${domain}`);

    // 2. 选择具体研究主题
    const topics = selectResearchTopics(domain);
    console.log(`📋 聚焦主题：${topics.join(', ')}`);

    // 3. 确保研究目录存在
    const domainDir = ensureResearchDir(domain);

    // 4. 生成调研日志
    const logFileName = getLogFileName(domain);
    const logFilePath = path.join(domainDir, logFileName);
    const logContent = generateDailyLog(domain, topics);

    // 5. 写入日志文件（如果已存在则跳过）
    if (fs.existsSync(logFilePath)) {
      console.log(`\n⏭️  今日日志已存在：${logFilePath}`);
      console.log('跳过生成，避免重复覆盖');
    } else {
      fs.writeFileSync(logFilePath, logContent, 'utf8');
      console.log(`\n✅ 已生成调研日志：${logFilePath}`);
    }

    // 6. 输出总结
    console.log('\n===========================================');
    console.log('每日定时任务执行完成！');
    console.log('===========================================');
    console.log(`领域：${domain}`);
    console.log(`主题：${topics.join(', ')}`);
    console.log(`日志：${logFilePath}`);
    console.log('===========================================\n');

  } catch (error) {
    console.error('❌ 执行每日任务时出错:', error);
    process.exit(1);
  }
}

// 执行
runDailyTask();
