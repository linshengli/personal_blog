'use client';

import { useState, useMemo } from 'react';
import Link from 'next/link';

interface PostData {
  id: string;
  topic: string;
  title: string;
  scope?: string;
  date: string;
  researchDate?: string;
  description?: string;
  tags?: string[];
  domain?: string;
}

interface ClientHomeProps {
  allPostsData: PostData[];
}

// 为不同领域生成不同的颜色
const domainColorCache = new Map<string, { bg: string; text: string }>();

function getDomainColor(domain: string): string {
  if (domainColorCache.has(domain)) {
    return domainColorCache.get(domain)!.bg;
  }

  const colors = [
    { bg: '#fef3c7', text: '#92400e' },
    { bg: '#dbeafe', text: '#1e40af' },
    { bg: '#dcfce7', text: '#166534' },
    { bg: '#fce7f3', text: '#9d174d' },
    { bg: '#ede9fe', text: '#5b21b6' },
    { bg: '#ffedd5', text: '#c2410c' },
    { bg: '#e0f2fe', text: '#075985' },
    { bg: '#f1f5f9', text: '#334155' },
  ];

  const index = domain.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
  const color = colors[index];
  domainColorCache.set(domain, color);
  return color.bg;
}

function getDomainTextColor(domain: string): string {
  if (domainColorCache.has(domain)) {
    return domainColorCache.get(domain)!.text;
  }

  const colors = [
    { bg: '#fef3c7', text: '#92400e' },
    { bg: '#dbeafe', text: '#1e40af' },
    { bg: '#dcfce7', text: '#166534' },
    { bg: '#fce7f3', text: '#9d174d' },
    { bg: '#ede9fe', text: '#5b21b6' },
    { bg: '#ffedd5', text: '#c2410c' },
    { bg: '#e0f2fe', text: '#075985' },
    { bg: '#f1f5f9', text: '#334155' },
  ];

  const index = domain.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % colors.length;
  const color = colors[index];
  domainColorCache.set(domain, color);
  return color.text;
}

export default function ClientHome({ allPostsData }: ClientHomeProps) {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDomain, setSelectedDomain] = useState<string>('all');
  const [selectedDate, setSelectedDate] = useState<string>('all');

  // 获取所有唯一的领域和时间
  const domains = useMemo(() => {
    const domainSet = new Set<string>();
    allPostsData.forEach(post => {
      if (post.domain) {
        domainSet.add(post.domain);
      }
    });
    return Array.from(domainSet).sort();
  }, [allPostsData]);

  const dates = useMemo(() => {
    const dateSet = new Set(allPostsData.map(post => post.date).filter(Boolean));
    return Array.from(dateSet).sort().reverse();
  }, [allPostsData]);

  // 筛选逻辑
  const filteredPosts = useMemo(() => {
    const now = new Date();
    const sevenDaysAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    return allPostsData.filter(post => {
      const term = searchTerm.toLowerCase();
      const matchSearch =
        post.title.toLowerCase().includes(term) ||
        (post.description && post.description.toLowerCase().includes(term)) ||
        (post.domain && post.domain.toLowerCase().includes(term));

      const matchDomain = selectedDomain === 'all' || post.domain === selectedDomain;

      let matchDate = true;
      if (selectedDate !== 'all') {
        if (selectedDate === '7days') {
          matchDate = !!(post.date && new Date(post.date) >= sevenDaysAgo);
        } else if (selectedDate === '30days') {
          matchDate = !!(post.date && new Date(post.date) >= thirtyDaysAgo);
        } else {
          matchDate = post.date === selectedDate;
        }
      }

      return matchSearch && matchDomain && matchDate;
    });
  }, [allPostsData, searchTerm, selectedDomain, selectedDate]);

  const clearFilters = () => {
    setSearchTerm('');
    setSelectedDomain('all');
    setSelectedDate('all');
  };

  const hasActiveFilters = searchTerm || selectedDomain !== 'all' || selectedDate !== 'all';

  return (
    <div className="container">
      <header className="site-header">
        <h1>linsheng 的博客</h1>
        <p className="subtitle">记录技术与思考</p>
      </header>

      <div className="search-box">
        <input
          type="text"
          placeholder="搜索文章..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
        {hasActiveFilters && (
          <button className="clear-filters-btn" onClick={clearFilters}>
            清除筛选
          </button>
        )}
      </div>

      <div className="filter-section">
        <div className="filter-group filter-group-full">
          <label className="filter-label">领域：</label>
          <div className="domain-tags">
            <button
              className={`domain-tag ${selectedDomain === 'all' ? 'active' : ''}`}
              onClick={() => setSelectedDomain('all')}
            >
              全部
            </button>
            {domains.map(domain => (
              <button
                key={domain}
                className={`domain-tag ${selectedDomain === domain ? 'active' : ''}`}
                onClick={() => setSelectedDomain(domain)}
                style={{
                  backgroundColor: selectedDomain === domain ? getDomainTextColor(domain) : getDomainColor(domain),
                  color: selectedDomain === domain ? '#fff' : getDomainTextColor(domain),
                }}
              >
                {domain}
              </button>
            ))}
          </div>
        </div>

        <div className="filter-group filter-group-full">
          <label className="filter-label">时间：</label>
          <div className="domain-tags">
            <button
              className={`domain-tag time-tag ${selectedDate === 'all' ? 'active' : ''}`}
              onClick={() => setSelectedDate('all')}
            >
              全部
            </button>
            {dates.map(date => (
              <button
                key={date}
                className={`domain-tag time-tag ${selectedDate === date ? 'active' : ''}`}
                onClick={() => setSelectedDate(date)}
              >
                {date}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="results-count">
        共 {allPostsData.length} 篇，当前显示 {filteredPosts.length} 篇
      </div>

      <main>
        {filteredPosts.length > 0 ? (
          <ul className="item-list">
            {filteredPosts.map(({ id, title, date, domain }) => (
              <li key={id}>
                <Link href={`/blog/${id}`}>
                  <div className="item-header">
                    <h2 className="item-title">{title}</h2>
                    {domain && (
                      <span
                        className="item-domain-tag"
                        style={{
                          backgroundColor: getDomainColor(domain),
                          color: getDomainTextColor(domain),
                        }}
                      >
                        {domain}
                      </span>
                    )}
                  </div>
                  <div className="item-meta">
                    <span className="date">{date}</span>
                  </div>
                </Link>
              </li>
            ))}
          </ul>
        ) : (
          <div className="no-results">
            没有找到相关文章
            {hasActiveFilters && (
              <button className="clear-filters-btn" onClick={clearFilters}>
                清除筛选条件
              </button>
            )}
          </div>
        )}
      </main>

      <footer className="site-footer">
        <p>Powered by Next.js & Vercel</p>
      </footer>
    </div>
  );
}
