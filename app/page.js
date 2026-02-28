import Link from "next/link";
import { getSortedPostsData } from "@/lib/posts";

export default async function Home() {
  const allPostsData = getSortedPostsData();

  return (
    <div className="container">
      <header className="site-header">
        <h1>个人博客</h1>
        <p className="subtitle">记录技术与思考</p>
      </header>

      <main className="item-list">
        {allPostsData.map(({ id, title, date, description }) => (
          <article key={id}>
            <Link href={`/blog/${id}`}>
              <h2 className="item-title">{title}</h2>
              <div className="item-meta">
                <span>{date}</span>
              </div>
              {description && <p className="item-description">{description}</p>}
            </Link>
          </article>
        ))}
      </main>

      <footer className="site-footer">
        <p>Powered by Next.js & Vercel</p>
      </footer>
    </div>
  );
}
