/**
 * Topic Selector
 *
 * Step 1 of the daily pipeline:
 * - Read topic-state.json for dedup history
 * - Scan research/ dir for already-completed topics
 * - Use claude-agent-sdk query() to select 3-5 non-duplicate topics
 *   from 4 domains: agent, 大模型训练, quant+agent, 大模型框架
 * - Save result to topic-state.json
 */

import fs from 'fs';
import path from 'path';
import { query } from '@anthropic-ai/claude-agent-sdk';
import { config } from '../config.js';

export interface TopicEntry {
  domain: string;
  topic: string;
  status: 'pending' | 'completed' | 'failed';
}

export interface TopicState {
  date: string;
  topics: TopicEntry[];
  history: string[]; // all-time dedup pool across days
}

const STATE_PATH = path.join(process.cwd(), config.files.topicState);

export function readTopicState(): TopicState {
  if (!fs.existsSync(STATE_PATH)) {
    return { date: '', topics: [], history: [] };
  }
  try {
    return JSON.parse(fs.readFileSync(STATE_PATH, 'utf8')) as TopicState;
  } catch {
    return { date: '', topics: [], history: [] };
  }
}

export function writeTopicState(state: TopicState): void {
  fs.writeFileSync(STATE_PATH, JSON.stringify(state, null, 2), 'utf8');
}

export function updateTopicStatus(topic: string, status: TopicEntry['status']): void {
  const state = readTopicState();
  const entry = state.topics.find(t => t.topic === topic);
  if (entry) {
    entry.status = status;
    writeTopicState(state);
  }
}

function getCompletedTopicsFromFS(): string[] {
  const researchDir = path.join(process.cwd(), config.files.researchDir);
  if (!fs.existsSync(researchDir)) return [];
  return fs
    .readdirSync(researchDir, { withFileTypes: true })
    .filter(e => e.isDirectory())
    .filter(e => {
      const dir = path.join(researchDir, e.name);
      return fs.readdirSync(dir).some(f => f.endsWith('-research.md'));
    })
    .map(e => e.name);
}

export async function selectTodayTopics(): Promise<TopicEntry[]> {
  const today = new Date().toISOString().split('T')[0];
  const state = readTopicState();

  // If today's topics are already selected, return them directly
  if (state.date === today && state.topics.length > 0) {
    console.log(`Today's topics already exist (${state.topics.length} total)`);
    return state.topics;
  }

  // Build dedup list: long-term history + filesystem scan
  const fsCompleted = getCompletedTopicsFromFS();
  const allExisting = Array.from(new Set([...state.history, ...fsCompleted]));

  const { min, max } = config.topics.totalCount;

  const prompt = `You are a technical research assistant. Select ${min} to ${max} specific research topics in total from the following 4 technology domains for today's research.

**Technology Domains:**
1. agent (AI Agent frameworks, architecture, tool use, Multi-Agent systems, RAG, etc.)
2. 大模型训练 (LLM pre-training, SFT fine-tuning, RLHF, alignment, etc.)
3. quant+agent (quantitative trading combined with AI Agents, strategy generation, risk control, etc.)
4. 大模型框架 (LLM inference frameworks, deployment optimization, vLLM, TGI, etc.)

**Selection Rules:**
- Select ${min}-${max} topics TOTAL across all domains (not required to cover every domain)
- Each topic must be a specific technical direction (e.g., "ReAct框架工具调用最新进展" NOT "Agent技术")
- Topics must be in Chinese, 8-30 characters long
- Strictly avoid conceptual overlap with already-researched topics:
  ${allExisting.length > 0 ? allExisting.join('\n  ') : 'none'}

**Output ONLY a JSON array, no other text:**
[
  {"domain": "agent", "topic": "具体topic名称"},
  {"domain": "大模型训练", "topic": "具体topic名称"},
  {"domain": "大模型框架", "topic": "具体topic名称"}
]`;

  console.log('Generating today\'s topics via claude-agent-sdk...');

  let output = '';
  for await (const message of query({
    prompt,
    options: { allowedTools: [], maxTurns: 1 },
  })) {
    if ('result' in message && message.result) {
      output += String(message.result);
    }
  }

  // Parse JSON array from LLM response (may have surrounding prose)
  const jsonMatch = output.match(/\[[\s\S]*?\]/);
  if (!jsonMatch) {
    throw new Error(`Cannot parse topics JSON from LLM response: ${output.slice(0, 300)}`);
  }

  let parsed: Array<{ domain: string; topic: string }>;
  try {
    parsed = JSON.parse(jsonMatch[0]);
  } catch (e) {
    throw new Error(`JSON parse failed: ${jsonMatch[0].slice(0, 300)}`);
  }

  if (parsed.length < min) {
    throw new Error(`LLM returned only ${parsed.length} topics, minimum is ${min}`);
  }

  const topics: TopicEntry[] = parsed.slice(0, max).map(t => ({
    domain: t.domain,
    topic: t.topic,
    status: 'pending' as const,
  }));

  const newState: TopicState = {
    date: today,
    topics,
    history: Array.from(new Set([...allExisting, ...topics.map(t => t.topic)])),
  };
  writeTopicState(newState);

  console.log(`Selected ${topics.length} topics:`);
  topics.forEach((t, i) => console.log(`  ${i + 1}. [${t.domain}] ${t.topic}`));

  return topics;
}
