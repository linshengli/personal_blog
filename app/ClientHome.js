'use client';

import { useState } from 'react';
import Link from 'next/link';

export default function ClientHome({ allPostsData }) {
  const [searchTerm, setSearchTerm] = useState('');

  const filteredPosts = allPostsData.filter(post => {
    const term = searchTerm.toLowerCase();
    return (
      post.title.toLowerCase().includes(term) ||
      (post.description && post.description.toLowerCase().includes(term))
    );
  });

  return (
    <div className="container">
      <header className="site-header">
        <h1>个人博客</h1>
        <p className="subtitle">记录技术与思考</p>
      </header>

      <div className="search-box">
        <input
          type="text"
          placeholder="搜索文章..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </div>

      <main>
        {filteredPosts.length > 0 ? (
          <ul className="item-list">
            {filteredPosts.map(({ id, title, date, description, tags }) => (
              <li key={id}>
                <Link href={`/blog/${id}`}>
                  <h2 className="item-title">{title}</h2>
                  <div className="item-meta">
                    <span>{date}</span>
                  </div>
                  {description && (
                    <p className="item-description">{description}</p>
                  )}
                  {tags && tags.length > 0 && (
                    <div className="item-tags">
                      {tags.map((tag, index) => (
                        <span key={index} className="tag">
                          {tag}
                        </span>
                      ))}
                    </div>
                  )}
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <div className="no-results">没有找到相关文章</div>
        )}
      </main>

      <footer className="site-footer">
        <p>Powered by Next.js & Vercel</p>
      </footer>
    </div>
  );
}
