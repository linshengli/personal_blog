import { getSortedPostsData } from '@/lib/posts';
import ClientHome from './ClientHome';

export default async function Home() {
  const allPostsData = getSortedPostsData();
  return <ClientHome allPostsData={allPostsData} />;
}
