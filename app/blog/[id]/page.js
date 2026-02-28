import { getAllPostIds, getPostData } from '@/lib/posts';
import Head from 'next/head';
import Link from 'next/link';
import GitHubComments from '@/components/GitHubComments';

export default async function BlogPost({ params }) {
  const { id } = await params;
  const postData = await getPostData(id);

  if (!postData) {
    return <div>Article not found</div>;
  }

  return (
    <div className="container">
      <Head>
        <title>{postData.title} - 个人博客</title>
        <meta name="description" content={postData.description || ''} />
      </Head>

      <header className="post-header">
        <Link href="/" className="back-link">← 返回首页</Link>
        <h1 className="post-title">{postData.title}</h1>
        <div className="post-meta">
          <span>{postData.date}</span>
        </div>
      </header>

      <article
        className="markdown"
        dangerouslySetInnerHTML={{ __html: postData.contentHtml }}
      />

      <GitHubComments issueTerm={id} />

      <footer className="post-footer">
        <Link href="/" className="back-link">← 返回首页</Link>
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
