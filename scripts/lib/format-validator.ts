/**
 * Format Validator
 *
 * Step 3 of the daily pipeline:
 * - For each completed topic, verify the research file exists
 * - Check minimum content length
 * - Check all required sections are present
 * - Print a validation report
 */

import fs from 'fs';
import path from 'path';
import { config } from '../config.js';
import { topicToDirName } from './research-runner.js';
import type { TopicEntry } from './topic-selector.js';

export interface ValidationResult {
  topic: string;
  passed: boolean;
  filePath: string;
  issues: string[];
  charCount?: number;
}

export function validateResearchFile(entry: TopicEntry): ValidationResult {
  const dirName = topicToDirName(entry.topic);
  const filePath = path.join(
    process.cwd(),
    config.files.researchDir,
    dirName,
    `${dirName}-research.md`
  );
  const issues: string[] = [];

  if (!fs.existsSync(filePath)) {
    return { topic: entry.topic, passed: false, filePath, issues: ['research file missing'] };
  }

  const content = fs.readFileSync(filePath, 'utf8');

  if (content.length < config.research.minContentLength) {
    issues.push(
      `content too short: ${content.length} chars (min ${config.research.minContentLength})`
    );
  }

  for (const section of config.research.requiredSections) {
    if (!content.includes(section)) {
      issues.push(`missing required section: ${section}`);
    }
  }

  return {
    topic: entry.topic,
    passed: issues.length === 0,
    filePath,
    issues,
    charCount: content.length,
  };
}

export function validateAllResearch(entries: TopicEntry[]): ValidationResult[] {
  const completed = entries.filter(e => e.status === 'completed');
  return completed.map(entry => validateResearchFile(entry));
}

export function printValidationReport(results: ValidationResult[]): boolean {
  if (results.length === 0) {
    console.log('No completed topics to validate');
    return true;
  }

  console.log('\nValidation report:');
  let allPassed = true;

  for (const r of results) {
    if (r.passed) {
      console.log(`  PASS  ${r.topic} (${r.charCount?.toLocaleString()} chars)`);
    } else {
      allPassed = false;
      console.log(`  FAIL  ${r.topic}`);
      r.issues.forEach(issue => console.log(`        - ${issue}`));
    }
  }

  return allPassed;
}
