import { getAllPostIds, getPostData } from '@/lib/posts';
import Link from 'next/link';
import GitHubComments from '@/components/GitHubComments';

export default async function BlogPost({ params }) {
  const { id } = await params;
  const postData = await getPostData(id);

  if (!postData) {
    return (
      <div className="container">
        <div className="no-results">文章不存在</div>
      </div>
    );
  }

  return (
    <div className="container">
      <Link href="/" className="back-link">
        ← 返回首页
      </Link>

      <header className="article-header">
        <h1>{postData.title}</h1>
        <div className="meta">
          <span>{postData.date}</span>
        </div>
        {postData.tags && postData.tags.length > 0 && (
          <div className="tags">
            {postData.tags.map((tag, index) => (
              <span key={index} className="tag">
                {tag}
              </span>
            ))}
          </div>
        )}
      </header>

      <article
        className="markdown"
        dangerouslySetInnerHTML={{ __html: postData.contentHtml }}
      />

      <GitHubComments issueTerm={id} />

      <footer className="post-footer">
        <Link href="/" className="back-link">
          ← 返回首页
        </Link>
      </footer>
    </div>
  );
}

export async function generateStaticParams() {
  const allPostIds = getAllPostIds();
  return allPostIds.map(({ params }) => ({
    id: params.id,
  }));
}
