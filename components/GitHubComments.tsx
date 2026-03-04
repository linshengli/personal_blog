'use client';

import { useEffect, useState } from 'react';

// 配置：请在使用前更新这些值
const GISCUS_CONFIG = {
  repo: 'linshengli/personal_blog',
  repoId: 'R_kgDORbKyMA',
  category: 'General',
  categoryId: 'DIC_kwDORbKyMM4C3aHJ',
};

interface GitHubCommentsProps {
  issueTerm: string;
}

interface RenderMathInElementOptions {
  delimiters: {
    left: string;
    right: string;
    display: boolean;
  }[];
  throwOnError: boolean;
}

declare global {
  interface Window {
    renderMathInElement: (element: HTMLElement, options: RenderMathInElementOptions) => void;
  }
}

export default function GitHubComments({ issueTerm }: GitHubCommentsProps) {
  const [isLoaded, setIsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // 检查配置是否已设置
    if (!GISCUS_CONFIG.repo || !GISCUS_CONFIG.repoId) {
      console.log('Giscus 评论系统未配置，请在 components/GitHubComments.js 中设置');
      return;
    }

    console.log('加载 Giscus 评论系统...', {
      repo: GISCUS_CONFIG.repo,
      repoId: GISCUS_CONFIG.repoId,
      category: GISCUS_CONFIG.category,
      categoryId: GISCUS_CONFIG.categoryId,
      issueTerm: issueTerm,
    });

    // 检查是否已加载过
    const existingScript = document.querySelector('script[src="https://giscus.app/client.js"]');
    if (existingScript) {
      console.log('Giscus 脚本已存在，跳过加载');
      setIsLoaded(true);
      return;
    }

    // 动态加载 Giscus 脚本
    const script = document.createElement('script');
    script.src = 'https://giscus.app/client.js';
    script.async = true;
    script.crossOrigin = 'anonymous';
    script.setAttribute('data-repo', GISCUS_CONFIG.repo);
    script.setAttribute('data-repo-id', GISCUS_CONFIG.repoId);
    script.setAttribute('data-category', GISCUS_CONFIG.category);
    script.setAttribute('data-category-id', GISCUS_CONFIG.categoryId);
    script.setAttribute('data-mapping', 'term');
    script.setAttribute('data-strict', '0');
    script.setAttribute('data-reactions-enabled', '1');
    script.setAttribute('data-emit-metadata', '0');
    script.setAttribute('data-input-position', 'top');
    script.setAttribute('data-theme', 'preferred_color_scheme');
    script.setAttribute('data-lang', 'zh-CN');
    script.setAttribute('data-issue-term', issueTerm);

    script.onload = () => {
      console.log('Giscus 脚本加载成功');
      setIsLoaded(true);
    };

    script.onerror = (err) => {
      console.error('Giscus 脚本加载失败:', err);
      setError('加载失败，请刷新重试');
    };

    const container = document.querySelector('.giscus-container');
    if (container) {
      container.appendChild(script);
    } else {
      console.error('找不到 .giscus-container 元素');
    }

    return () => {
      // 不清除脚本，避免重复加载
    };
  }, [issueTerm]);

  // 如果未配置，不显示评论区
  if (!GISCUS_CONFIG.repo || !GISCUS_CONFIG.repoId) {
    return null;
  }

  return (
    <section className="comments-section">
      <h2 className="comments-title">评论</h2>
      <div className="giscus-container" />
      {!isLoaded && !error && (
        <p style={{ color: 'var(--muted)', textAlign: 'center', padding: '2rem 0' }}>
          评论加载中...
        </p>
      )}
      {error && (
        <p style={{ color: 'var(--muted)', textAlign: 'center', padding: '2rem 0' }}>
          {error}
        </p>
      )}
    </section>
  );
}
