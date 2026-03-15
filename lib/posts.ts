import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { unified } from 'unified';
import remarkParse from 'remark-parse';
import remarkRehype from 'remark-rehype';
import rehypeStringify from 'rehype-stringify';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import remarkGfm from 'remark-gfm';

const postsDirectory = path.join(process.cwd(), 'research');

// 内存缓存：存储已转换的 Markdown 内容
const htmlCache = new Map<string, { html: string; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 分钟缓存过期时间

/**
 * 从 Markdown 内容中提取研究范围（标题行）
 * 格式：**研究范围**: XXX 或 **研究范围:** XXX
 */
function extractResearchScope(content: string): string | null {
  const lines = content.split('\n');
  for (const line of lines) {
    // 支持多种格式：**研究范围**、**调研主题**、**研究主题**
    const match = line.match(/\*\*(?:研究范围|调研主题|研究主题)\*\*:?\s*(.+)/);
    if (match) {
      return match[1].trim();
    }
  }
  return null;
}

/**
 * 从 Markdown 内容中提取文章主标题（从第一个 # 标题行）
 */
function extractMainTitle(content: string): string | null {
  const lines = content.split('\n');
  for (const line of lines) {
    const match = line.match(/^#\s+(.+)/);
    if (match) {
      return match[1].trim();
    }
  }
  return null;
}

/**
 * 从内容中提取调研日期
 * 格式：> 调研日期：YYYY-MM-DD  或  **调研日期**：YYYY-MM-DD
 */
function extractResearchDate(content: string): string | null {
  // Try blockquote format first: > 调研日期：YYYY-MM-DD
  const blockquoteMatch = content.match(/>\s*调研日期：(\d{4}-\d{2}-\d{2})/);
  if (blockquoteMatch) {
    return blockquoteMatch[1];
  }
  // Try bold format: **调研日期**：YYYY-MM-DD
  const boldMatch = content.match(/\*\*调研日期\*\*：\s*(\d{4}-\d{2}-\d{2})/);
  if (boldMatch) {
    return boldMatch[1];
  }
  return null;
}

/**
 * 从内容中提取调研主题
 * 格式：> 调研主题：XXX  或  **调研主题**：XXX
 */
function extractResearchTopic(content: string): string | null {
  // Try blockquote format first: > 调研主题：XXX
  const blockquoteMatch = content.match(/>\s*调研主题：(.+)/);
  if (blockquoteMatch) {
    return blockquoteMatch[1].trim();
  }
  // Try bold format: **调研主题**：XXX
  const boldMatch = content.match(/\*\*调研主题\*\*：\s*(.+)/);
  if (boldMatch) {
    return boldMatch[1].trim();
  }
  return null;
}

/**
 * 将 Markdown 转换为 HTML，支持 LaTeX 数学公式和表格
 * 带缓存机制，避免重复转换
 */
async function markdownToHtml(markdown: string, filePath: string): Promise<string> {
  // 检查缓存
  const cached = htmlCache.get(filePath);
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.html;
  }

  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm)
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    // @ts-ignore - rehype-katex options
    .use(rehypeKatex, {
      throwOnError: false,
      strict: false,
      output: 'htmlAndMathml',
    })
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(markdown);

  const html = result.toString();

  // 存入缓存
  htmlCache.set(filePath, { html, timestamp: Date.now() });

  return html;
}

export interface PostData {
  id: string;
  topic: string;
  title: string;
  scope?: string; // 研究范围
  date: string;
  researchDate?: string; // 调研日期
  description?: string;
  tags?: string[];
  domain?: string; // 领域分类（从 topic 派生）
}

export interface CachedPostData extends PostData {
  contentHtml: string;
}

/**
 * 获取所有文章数据
 */
export function getSortedPostsData(): PostData[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const dirEntries = fs.readdirSync(postsDirectory, { withFileTypes: true });

  const allPostsData: PostData[] = dirEntries
    .filter(dirent => dirent.isDirectory())
    .filter(dirent => dirent.name !== 'agent-framework') // 隐藏 agent-framework
    .flatMap(dirent => {
      const topic = dirent.name;
      const topicDir = path.join(postsDirectory, topic);

      if (!fs.existsSync(topicDir)) {
        return [];
      }

      const files = fs.readdirSync(topicDir);

      const researchFile = files.find(
        file => file === `${topic}-research.md` || file === 'research.md'
      );

      if (!researchFile) {
        return [];
      }

      const filePath = path.join(topicDir, researchFile);
      const fileContents = fs.readFileSync(filePath, 'utf8');
      const { data, content } = matter(fileContents);

      // 从内容中提取研究范围、调研日期和调研主题
      const scope = extractResearchScope(content);
      const researchDate = extractResearchDate(content);
      const researchTopic = extractResearchTopic(content);
      const mainTitle = extractMainTitle(content);

      // 优先使用主标题 > 研究范围 > frontmatter title > topic id
      const title = mainTitle || scope || data.title || topic;
      const date = researchDate || data.date || '';
      const id = topic;

      return {
        id,
        topic,
        title,
        scope,
        date,
        researchDate,
        description: data.description || '',
        ...(data.tags && { tags: Array.isArray(data.tags) ? data.tags : [data.tags] }),
        domain: researchTopic || topic,
      };
    })
    .filter(post => post !== null)
    .sort((a, b) => {
      if (a.date < b.date) {
        return 1;
      } else {
        return -1;
      }
    });

  return allPostsData;
}

/**
 * 获取所有文章的 ID 用于生成静态页面
 */
export function getAllPostIds(): { params: { id: string } }[] {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const dirEntries = fs.readdirSync(postsDirectory, { withFileTypes: true });

  return dirEntries
    .filter(dirent => dirent.isDirectory())
    .map(dirent => {
      const topic = dirent.name;
      const topicDir = path.join(postsDirectory, topic);

      if (!fs.existsSync(topicDir)) {
        return null;
      }

      const files = fs.readdirSync(topicDir);

      const researchFile = files.find(
        file => file === `${topic}-research.md` || file === 'research.md'
      );

      if (!researchFile) {
        return null;
      }

      return {
        params: {
          id: topic,
        },
      };
    })
    .filter(item => item !== null);
}

/**
 * 获取单篇文章的数据，支持 LaTeX 数学公式
 */
export async function getPostData(id: string): Promise<CachedPostData | null> {
  const topicDir = path.join(postsDirectory, id);
  const filePath = path.join(topicDir, `${id}-research.md`);
  const fallbackPath = path.join(topicDir, 'research.md');

  let actualFilePath = filePath;
  if (!fs.existsSync(actualFilePath)) {
    actualFilePath = fallbackPath;
  }

  if (!fs.existsSync(actualFilePath)) {
    return null;
  }

  const fileContents = fs.readFileSync(actualFilePath, 'utf8');
  const { data, content } = matter(fileContents);

  const scope = extractResearchScope(content);
  const researchDate = extractResearchDate(content);
  const researchTopic = extractResearchTopic(content);
  const mainTitle = extractMainTitle(content);

  const title = mainTitle || scope || data.title || id;
  const date = researchDate || data.date || '';

  const contentHtml = await markdownToHtml(content, actualFilePath);

  return {
    id,
    topic: id,
    contentHtml,
    title,
    scope,
    date,
    researchDate,
    description: data.description || '',
    ...(data.tags && { tags: data.tags }),
    domain: researchTopic || id,
  };
}

/**
 * 获取所有唯一的领域分类
 */
export function getAllDomains(posts: PostData[]): string[] {
  const domains: string[] = [];
  posts.forEach(post => {
    if (post.domain) {
      domains.push(post.domain);
    }
  });
  return Array.from(new Set(domains)).sort();
}

/**
 * 获取所有唯一的日期（用于筛选）
 */
export function getAllDates(posts: PostData[]): string[] {
  const dates: string[] = [];
  posts.forEach(post => {
    if (post.date) {
      dates.push(post.date);
    }
  });
  return Array.from(new Set(dates)).sort().reverse();
}
