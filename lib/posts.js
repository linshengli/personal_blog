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

/**
 * 将 Markdown 转换为 HTML，支持 LaTeX 数学公式和表格
 */
async function markdownToHtml(markdown) {
  const result = await unified()
    .use(remarkParse)
    .use(remarkGfm) // 支持表格等 GitHub Flavored Markdown
    .use(remarkMath)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(rehypeKatex, {
      throwOnError: false,
      strict: false,
      output: 'htmlAndMathml',
    })
    .use(rehypeStringify, { allowDangerousHtml: true })
    .process(markdown);

  return result.toString();
}

/**
 * 获取所有文章数据
 * 读取规则：
 * 1. research/{topic}/{topic}-research.md (优先)
 * 2. research/{topic}/research.md (兼容旧格式)
 */
export function getSortedPostsData() {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const dirEntries = fs.readdirSync(postsDirectory, { withFileTypes: true });

  const allPostsData = dirEntries
    .filter(dirent => dirent.isDirectory())
    .flatMap(dirent => {
      const topic = dirent.name;
      const topicDir = path.join(postsDirectory, topic);

      if (!fs.existsSync(topicDir)) {
        return [];
      }

      const files = fs.readdirSync(topicDir);

      // 优先查找 {topic}-research.md
      const researchFile = files.find(
        file => file === `${topic}-research.md` || file === 'research.md'
      );

      if (!researchFile) {
        return [];
      }

      const filePath = path.join(topicDir, researchFile);
      const fileContents = fs.readFileSync(filePath, 'utf8');
      const { data } = matter(fileContents);

      const id = topic;

      return {
        id,
        topic,
        title: data.title || topic,
        date: data.date || '',
        description: data.description || '',
        ...(data.tags && { tags: Array.isArray(data.tags) ? data.tags : [data.tags] }),
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
export function getAllPostIds() {
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

      // 查找 {topic}-research.md 或 research.md
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
export async function getPostData(id) {
  const topicDir = path.join(postsDirectory, id);
  const filePath = path.join(topicDir, `${id}-research.md`);
  const fallbackPath = path.join(topicDir, 'research.md');

  // 优先查找 {topic}-research.md，否则使用 research.md
  let actualFilePath = filePath;
  if (!fs.existsSync(actualFilePath)) {
    actualFilePath = fallbackPath;
  }

  if (!fs.existsSync(actualFilePath)) {
    return null;
  }

  const fileContents = fs.readFileSync(actualFilePath, 'utf8');
  const { data, content } = matter(fileContents);

  // 使用 unified 处理 Markdown 和 LaTeX
  const contentHtml = await markdownToHtml(content);

  return {
    id,
    topic: id,
    contentHtml,
    title: data.title || id,
    date: data.date || '',
    description: data.description || '',
    ...(data.tags && { tags: data.tags }),
  };
}
