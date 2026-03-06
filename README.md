# 个人博客

一个极简风格的个人博客系统，基于 Next.js + Vercel 构建。设计参考了 [概念解剖](https://concept.x.fish/)。

## 在线演示

> 部署后在此处添加你的博客链接

## 功能特点

- **自动读取文章**：自动扫描 `research` 目录下所有子目录中的 `research.md` 文件
- **每日自动调研**：GitHub Actions 定时任务自动选择领域进行调研并生成文章
- **Topic 去重管理**：智能检测已生成的 topic，避免重复生成
- **Markdown 支持**：完整支持 Markdown 格式，包括代码高亮、表格、引用等
- **搜索功能**：首页支持实时搜索文章标题和描述
- **GitHub 评论**：集成 Giscus，支持 GitHub Issues 评论系统
- **深色模式**：自动跟随系统深色/浅色模式切换
- **响应式设计**：完美适配移动端和桌面端
- **SEO 友好**：支持自定义文章标题、描述等元数据
- **静态页面生成**：使用 SSG 预构建静态页面，访问速度更快

## 项目结构

```
personal_blogs/
├── app/
│   ├── blog/[id]/
│   │   └── page.js          # 文章详情页路由
│   ├── globals.css          # 全局样式（CSS 变量 + 深色模式）
│   ├── layout.js            # 根布局组件
│   ├── page.js              # 首页（服务端组件）
│   └── ClientHome.js        # 首页客户端组件（含搜索）
├── components/
│   └── GitHubComments.js    # GitHub Issues 评论组件
├── lib/
│   └── posts.js             # 文章数据读取和解析工具
├── scripts/
│   ├── daily.ts             # 每日定时任务脚本
│   └── lib/
│       ├── topic-manager.ts # Topic 去重管理模块
│       └── research-agent.ts # 研究内容生成模块
├── research/                # 博客文章目录
│   ├── <article-folder>/
│   │   └── research.md      # 每篇文章的 markdown 文件
│   │   └── daily-log-*.md   # 每日生成日志
│   └── topics-history.json  # Topic 历史记录
├── .github/workflows/
│   ├── daily-task.yml       # 每日任务工作流
│   └── build.yml            # 静态构建工作流
├── .env.example             # 环境变量模板
├── next.config.js           # Next.js 配置
├── package.json             # 项目依赖
├── vercel.json              # Vercel 部署配置
└── README.md                # 项目说明
```

## 快速开始

### 安装依赖

```bash
npm install
```

### 本地开发

```bash
npm run dev
```

访问 http://localhost:3000 查看博客。

### 添加新文章

1. 在 `research` 目录下创建新的文件夹，例如 `research/my-article/`
2. 在文件夹中创建 `research.md` 文件
3. 在文件顶部添加 YAML front matter：

```yaml
---
title: "文章标题"
date: "2026-02-28"
description: "文章描述（可选，会显示在首页列表）"
tags: ["标签 1", "标签 2"]  # 可选
---
```

4. 编写 Markdown 正文内容

## 每日自动调研

博客系统支持每日自动选择技术领域进行调研并生成文章。

### 配置环境变量

在 GitHub Secrets 中添加以下环境变量：

| 变量名 | 说明 | 获取方式 |
|--------|------|---------|
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 | [Anthropic Console](https://console.anthropic.com/) |

**可选：使用云服务**

| 云服务 | 环境变量 | 说明 |
|--------|---------|------|
| Amazon Bedrock | `CLAUDE_CODE_USE_BEDROCK=1` | 需要配置 AWS 凭证 |
| Google Vertex AI | `CLAUDE_CODE_USE_VERTEX=1` | 需要配置 Google Cloud 凭证 |
| Azure AI Foundry | `CLAUDE_CODE_USE_FOUNDRY=1` | 需要配置 Azure 凭证 |

**注意**：Claude Agent SDK 不支持阿里云百炼 API，请使用 Anthropic API 或其他云服务。

### 手动触发每日任务

1. 访问 GitHub 仓库的 Actions 页面
2. 选择 "Daily Research Task" 工作流
3. 点击 "Run workflow" 按钮
4. 等待任务执行完成

### 查看 Topic 统计

在本地运行以下命令查看 Topic 完成情况：

```bash
npx tsx -e "import { getTopicStats } from './scripts/lib/topic-manager'; console.log(getTopicStats());"
```

### 预定义的研究领域

系统预定义了 15 个研究领域：

- AI Agent
- MCP (Model Context Protocol)
- RAG (Retrieval-Augmented Generation)
- LLM Fine-tuning
- Prompt Engineering
- Vector Database
- AI Workflow
- Code Generation
- Multi-Agent System
- AI Memory
- Function Calling
- Semantic Search
- Knowledge Graph
- AI Evaluation
- Edge AI

当所有领域都完成后，系统会自动循环复用。

## 部署

### 1. 推送到 GitHub

```bash
# 关联远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

### 2. 在 Vercel 部署

1. 访问 [Vercel New Project](https://vercel.com/new)
2. 点击 "Import Git Repository"
3. 选择你的博客仓库
4. 保持默认设置，点击 "Deploy"

部署完成后，Vercel 会自动分配一个 `*.vercel.app` 域名，你也可以绑定自定义域名。

### 3. 配置 GitHub 评论（可选）

1. 访问 [Giscus](https://giscus.app/zh-CN)
2. 按照说明配置你的 GitHub 仓库：
   - 安装 Giscus GitHub App
   - 选择仓库和 Announcements 分类
3. 获取配置信息：
   - `repo`：例如 `tbxsx/personal_blogs`
   - `repoId`：仓库 ID
   - `category`：分类名称（如 "Announcements"）
   - `categoryId`：分类 ID
4. 更新 `components/GitHubComments.js` 中的配置：

```javascript
<script
  src="https://giscus.app/client.js"
  data-repo="YOUR_GITHUB_USERNAME/YOUR_REPO_NAME"
  data-repo-id="YOUR_REPO_ID"
  data-category="Announcements"
  data-category-id="YOUR_CATEGORY_ID"
  // ...其他配置
/>
```

5. 提交并推送更改，Vercel 会自动重新部署

## 文章 Front Matter 格式

```yaml
---
title: "文章标题"           # 必填
date: "2026-02-28"        # 必填，格式 YYYY-MM-DD
description: "文章描述"    # 可选，显示在首页列表
tags: ["标签 1", "标签 2"] # 可选，显示为标签
---
```

## 样式定制

在 `app/globals.css` 中修改 CSS 变量来定制主题：

```css
:root {
  --bg: #fcfcfa;              /* 背景色 */
  --fg: #1a1a1a;              /* 文字颜色 */
  --accent: #2d6a4f;          /* 强调色 */
  --border: #e8e5df;          /* 边框颜色 */
  /* ...更多变量 */
}

@media (prefers-color-scheme: dark) {
  :root {
    /* 深色模式配色 */
  }
}
```

## 技术栈

- [Next.js 16](https://nextjs.org/) - React 框架
- [React 19](https://react.dev/) - UI 库
- [Tailwind CSS v4](https://tailwindcss.com/) - CSS 框架
- [remark](https://github.com/remarkjs/remark) - Markdown 解析
- [gray-matter](https://github.com/jonschlinkert/gray-matter) - Front Matter 解析
- [Giscus](https://giscus.app/) - GitHub Issues 评论

## License

MIT
