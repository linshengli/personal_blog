import fs from 'fs';
import path from 'path';
import matter from 'gray-matter';
import { remark } from 'remark';
import html from 'remark-html';

const postsDirectory = path.join(process.cwd(), 'research');

export function getSortedPostsData() {
  // 获取 research 目录下的所有子目录
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const dirEntries = fs.readdirSync(postsDirectory, { withFileTypes: true });

  const allPostsData = dirEntries
    .filter(dirent => dirent.isDirectory())
    .map(dirent => {
      const folderName = dirent.name;
      const filePath = path.join(postsDirectory, folderName, 'research.md');

      if (!fs.existsSync(filePath)) {
        return null;
      }

      const fileContents = fs.readFileSync(filePath, 'utf8');
      const { data } = matter(fileContents);

      return {
        id: folderName,
        title: data.title || folderName,
        date: data.date || '',
        description: data.description || '',
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

export function getAllPostIds() {
  if (!fs.existsSync(postsDirectory)) {
    return [];
  }

  const dirEntries = fs.readdirSync(postsDirectory, { withFileTypes: true });

  return dirEntries
    .filter(dirent => dirent.isDirectory())
    .map(dirent => {
      const filePath = path.join(postsDirectory, dirent.name, 'research.md');
      if (fs.existsSync(filePath)) {
        return {
          params: {
            id: dirent.name,
          },
        };
      }
      return null;
    })
    .filter(item => item !== null);
}

export async function getPostData(id) {
  const filePath = path.join(postsDirectory, id, 'research.md');

  if (!fs.existsSync(filePath)) {
    return null;
  }

  const fileContents = fs.readFileSync(filePath, 'utf8');
  const { data, content } = matter(fileContents);

  const processedContent = await remark()
    .use(html, { sanitize: false })
    .process(content);
  const contentHtml = processedContent.toString();

  return {
    id,
    contentHtml,
    title: data.title || id,
    date: data.date || '',
    description: data.description || '',
    ...(data.tags && { tags: data.tags }),
  };
}
