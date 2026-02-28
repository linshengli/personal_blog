# 个人博客

一个简单的个人博客系统，基于 Next.js + Vercel 构建。

## 功能特点

- 自动读取 `research` 目录下的 `research.md` 文件
- 支持 Markdown 格式解析
- 支持 GitHub Issues 评论（通过 Giscus）
- 响应式设计，支持深色模式
- 通过 Vercel 一键部署

## 项目结构

```
personal_blogs/
├── app/
│   ├── blog/[id]/
│   │   └── page.js      # 文章详情页
│   ├── globals.css      # 全局样式
│   ├── layout.js        # 根布局
│   └── page.js          # 首页
├── components/
│   └── GitHubComments.js # GitHub 评论组件
├── lib/
│   └── posts.js         # 文章数据读取工具
├── research/            # 博客文章目录
│   └── example-post/
│       └── research.md
├── package.json
└── vercel.json
```

## 快速开始

### 本地开发

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

访问 http://localhost:3000

### 添加新文章

1. 在 `research` 目录下创建新的文件夹
2. 在文件夹中创建 `research.md` 文件
3. 添加 YAML front matter：

```yaml
---
title: "文章标题"
date: "2026-02-28"
description: "文章描述"
---
```

4. 编写 Markdown 内容

## 部署

### 1. 初始化 Git 仓库

```bash
git init
git add .
git commit -m "Initial commit"
```

### 2. 创建 GitHub 仓库并推送

```bash
# 在 GitHub 上创建新仓库后
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 3. 在 Vercel 部署

1. 访问 [Vercel](https://vercel.com/new)
2. 点击 "Import Git Repository"
3. 选择你的博客仓库
4. 点击 "Deploy"

### 4. 配置 GitHub 评论（可选）

1. 访问 [Giscus](https://giscus.app/zh-CN)
2. 按照说明配置你的 GitHub 仓库
3. 获取 `repo` 和 `repoId` 等信息
4. 更新 `components/GitHubComments.js` 中的配置：

```javascript
data-repo="YOUR_GITHUB_USERNAME/YOUR_REPO_NAME"
data-repo-id="YOUR_REPO_ID"
data-category="General"
data-category-id="YOUR_CATEGORY_ID"
```

## 参考设计

本博客设计参考了 [概念解剖](https://concept.x.fish/) 的极简风格。
