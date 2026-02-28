'use client';

import { useEffect } from 'react';

export default function GitHubComments({ issueTerm }) {
  useEffect(() => {
    // Giscus 脚本会在页面加载后自动执行
    // 这里不需要额外处理，因为 script 标签会在 HTML 中渲染
  }, []);

  return (
    <section className="comments-section">
      <h2 className="comments-title">评论</h2>
      <div className="giscus-container" />
      <script
        src="https://giscus.app/client.js"
        data-repo="YOUR_GITHUB_USERNAME/YOUR_REPO_NAME"
        data-repo-id="YOUR_REPO_ID"
        data-category="Announcements"
        data-category-id="YOUR_CATEGORY_ID"
        data-mapping="term"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="preferred_color_scheme"
        data-lang="zh-CN"
        data-issue-term={issueTerm}
        crossOrigin="anonymous"
        async
      />
    </section>
  );
}
