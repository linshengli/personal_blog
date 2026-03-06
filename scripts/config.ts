export const config = {
  research: {
    maxConcurrency: 5,       // max parallel research tasks
    maxTurns: 20,            // max agent turns per research session
    timeoutMs: 30 * 60 * 1000, // 30 minutes per topic, then abort
    minContentLength: 1000,    // minimum chars for a valid research file
    requiredSections: ['概念剖析', '行业情报', '方案对比', '精华整合'],
  },
  topics: {
    totalCount: { min: 3, max: 5 },
    domains: ['agent', '大模型训练', 'quant+agent', '大模型框架'] as const,
  },
  files: {
    topicState: 'topic-state.json',
    researchDir: 'research',
  },
} as const;
