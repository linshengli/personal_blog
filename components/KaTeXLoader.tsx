'use client';

import { useEffect } from 'react';

export default function KaTeXLoader() {
  useEffect(() => {
    // 等待 KaTeX 脚本加载完成后自动渲染
    const initializeKatex = () => {
      if (typeof window !== 'undefined' && window.renderMathInElement) {
        window.renderMathInElement(document.body, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
        });
      }
    };

    // 页面加载完成后渲染
    if (document.readyState === 'complete') {
      // 延迟一点确保 KaTeX 脚本已加载
      setTimeout(initializeKatex, 100);
    } else {
      window.addEventListener('load', () => {
        setTimeout(initializeKatex, 100);
      });
    }

    // 防抖：避免频繁触发 KaTeX 渲染
    let debounceTimer: NodeJS.Timeout;
    const observer = new MutationObserver(() => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        initializeKatex();
      }, 500);
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    return () => {
      clearTimeout(debounceTimer);
      observer.disconnect();
    };
  }, []);

  return null;
}
