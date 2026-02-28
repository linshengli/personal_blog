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

    // 检查脚本是否已加载
    const script = document.querySelector('script[src*="katex/auto-render"]');
    if (script && script.readyState === 'complete') {
      initializeKatex();
    } else if (script) {
      script.addEventListener('load', initializeKatex);
    }

    // 页面加载完成后也尝试渲染（防止脚本已加载但事件已错过）
    if (document.readyState === 'complete') {
      initializeKatex();
    } else {
      window.addEventListener('load', initializeKatex);
    }

    return () => {
      window.removeEventListener('load', initializeKatex);
    };
  }, []);

  return null;
}
