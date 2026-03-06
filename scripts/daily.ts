/**
 * Daily Research Pipeline
 *
 * Step 1: Select 3-5 non-duplicate topics from 4 domains via claude-agent-sdk
 *         Save to topic-state.json
 * Step 2: Run parallel research for each topic via claude-agent-sdk
 *         (WebSearch, WebFetch, Write, Read, Glob, Grep)
 * Step 3: Validate format of generated files
 *         Results committed to git by GitHub Actions
 */

import { selectTodayTopics, readTopicState, writeTopicState } from './lib/topic-selector.js';
import type { TopicEntry } from './lib/topic-selector.js';
import { runParallelResearch } from './lib/research-runner.js';
import { validateAllResearch, printValidationReport } from './lib/format-validator.js';

console.log('===========================================');
console.log('Daily Research Pipeline');
console.log(`Time: ${new Date().toISOString()}`);
console.log('===========================================\n');

/**
 * Parse custom topics from:
 *   CLI:  npm run daily -- --topics "topic1,topic2,topic3"
 *   ENV:  RESEARCH_TOPICS="topic1,topic2" npm run daily   (set by workflow_dispatch input)
 */
function parseCustomTopics(): TopicEntry[] | null {
  let raw = '';

  const cliIdx = process.argv.indexOf('--topics');
  if (cliIdx !== -1 && process.argv[cliIdx + 1]) {
    raw = process.argv[cliIdx + 1];
  } else if (process.env.RESEARCH_TOPICS) {
    raw = process.env.RESEARCH_TOPICS;
  }

  if (!raw.trim()) return null;

  return raw
    .split(',')
    .map(t => t.trim())
    .filter(Boolean)
    .map(topic => ({ domain: 'custom', topic, status: 'pending' as const }));
}

async function runDailyTask() {
  try {
    // Step 1: Topic selection
    console.log('--- Step 1: Topic Selection ---');

    const customTopics = parseCustomTopics();
    let topics: TopicEntry[];

    if (customTopics && customTopics.length > 0) {
      console.log(`Custom topics provided (${customTopics.length}), skipping auto-selection`);
      topics = customTopics;

      // Persist to topic-state.json for dedup tracking
      const today = new Date().toISOString().split('T')[0];
      const prev = readTopicState();
      writeTopicState({
        date: today,
        topics,
        history: Array.from(new Set([...prev.history, ...topics.map(t => t.topic)])),
      });
    } else {
      topics = await selectTodayTopics();
    }

    console.log(`Selected ${topics.length} topics\n`);

    // Step 2: Parallel research
    console.log('--- Step 2: Parallel Research ---');
    const researchResults = await runParallelResearch(topics);

    const succeeded = researchResults.filter(r => r.success);
    const failed = researchResults.filter(r => !r.success);

    console.log(`\nResearch complete: ${succeeded.length} succeeded, ${failed.length} failed`);
    // Errors are already printed inside runParallelResearch per-topic

    // Step 3: Format validation
    console.log('\n--- Step 3: Format Validation ---');
    const state = readTopicState();
    const validationResults = validateAllResearch(state.topics);
    const allPassed = printValidationReport(validationResults);

    // Summary
    console.log('\n===========================================');
    console.log('Pipeline complete');
    console.log('===========================================');
    topics.forEach(t => console.log(`  [${t.domain}] ${t.topic}`));
    console.log(`Research: ${succeeded.length}/${topics.length} succeeded`);
    console.log(`Validation: ${allPassed ? 'all passed' : 'some failed'}`);
    console.log('===========================================\n');

    const runnableCount = researchResults.length;
    if (runnableCount > 0 && succeeded.length === 0) {
      throw new Error('All topic research failed');
    }
  } catch (error) {
    console.error('\nDaily pipeline error:', error);
    process.exit(1);
  }
}

runDailyTask();
