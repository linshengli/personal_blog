---
title: "示例文章"
date: "2026-02-28"
description: "这是一篇示例博客文章"
---

# 欢迎使用个人博客系统

这是一篇示例文章，用于测试博客系统是否正常工作。

## 功能特点

- 支持 Markdown 格式
- 自动读取 research 目录下的文章
- 支持 GitHub Issues 评论
- 通过 Vercel 一键部署

## 如何添加新文章

1. 在 `research` 目录下创建新的文件夹
2. 在文件夹中创建 `research.md` 文件
3. 在文件顶部添加 YAML front matter：

```yaml
---
title: "文章标题"
date: "2026-02-28"
description: "文章描述"
---
```

4. 编写你的 Markdown 内容

## 部署

```bash
# 1. 初始化 git 仓库
git init
git add .
git commit -m "Initial commit"

# 2. 创建 GitHub 仓库并推送
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git push -u origin main

# 3. 在 Vercel 中导入 GitHub 仓库
# 访问 https://vercel.com/new 导入项目
```

## 本地开发

```bash
npm install
npm run dev
```

然后访问 http://localhost:3000
