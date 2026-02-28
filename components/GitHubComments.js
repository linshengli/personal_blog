'use client';

import { useEffect, useState } from 'react';

export default function GitHubComments({ issueTerm }) {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    return null;
  }

  return (
    <section className="comments-section">
      <h2 className="comments-title">评论</h2>
      <div
        className="giscus-container"
        data-issue-term={issueTerm}
        data-label="Comments"
        data-reaction-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="preferred_color_scheme"
        data-lang="zh-CN"
        crossOrigin="anonymous"
      />
      <script
        src="https://giscus.app/client.js"
        data-repo="YOUR_GITHUB_USERNAME/YOUR_REPO_NAME"
        data-repo-id="YOUR_REPO_ID"
        data-category="General"
        data-category-id="YOUR_CATEGORY_ID"
        data-mapping="term"
        data-strict="0"
        data-reactions-enabled="1"
        data-emit-metadata="0"
        data-input-position="top"
        data-theme="preferred_color_scheme"
        data-lang="zh-CN"
        crossOrigin="anonymous"
        async
      />
      <style jsx>{`
        .comments-section {
          max-width: var(--max-width);
          margin: 3rem auto;
          padding-top: 2rem;
          border-top: 1px solid var(--border);
        }

        .comments-title {
          font-size: 1.3rem;
          font-weight: 600;
          margin-bottom: 1.5rem;
          color: var(--fg);
        }

        .giscus-container {
          min-height: 200px;
        }
      `}</style>
    </section>
  );
}
