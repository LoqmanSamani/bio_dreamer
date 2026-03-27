import Link from "next/link";
import { Calendar, User, ArrowLeft, ExternalLink } from "lucide-react";
import { getAllSlugs, getPostBySlug } from "@/lib/blog";
import { MDXRemote } from "next-mdx-remote/rsc";
import remarkGfm from "remark-gfm";
import rehypeSlug from "rehype-slug";

export function generateStaticParams() {
  return getAllSlugs().map((slug) => ({ slug }));
}

export default function BlogPostPage({
  params,
}: {
  params: { slug: string };
}) {
  const post = getPostBySlug(params.slug);

  if (!post) {
    return (
      <div className="section-container py-16 text-center">
        <h1 className="text-3xl font-bold text-white mb-4">Post Not Found</h1>
        <Link href="/blog" className="btn-outline">
          <ArrowLeft size={16} /> Back to Blog
        </Link>
      </div>
    );
  }

  return (
    <div className="section-container py-16">
      <div className="max-w-3xl mx-auto">
        {/* Back link */}
        <Link
          href="/blog"
          className="inline-flex items-center gap-1.5 text-sm text-slate-500 hover:text-protein-400 transition-colors mb-8"
        >
          <ArrowLeft size={14} /> All Posts
        </Link>

        {/* Header */}
        <header className="mb-10">
          <h1 className="text-3xl sm:text-4xl font-bold text-white mb-4 leading-tight">
            {post.title}
          </h1>
          <div className="flex flex-wrap items-center gap-4 text-sm text-slate-500 mb-4">
            <span className="inline-flex items-center gap-1.5">
              <Calendar size={14} /> {post.date}
            </span>
            <span className="inline-flex items-center gap-1.5">
              <User size={14} /> {post.author}
            </span>
          </div>
          <div className="flex flex-wrap gap-2 mb-4">
            {post.tags.map((tag) => (
              <span key={tag} className="tag">
                {tag}
              </span>
            ))}
          </div>
          <a
            href={`https://github.com/LoqmanSamani/bio_dreamer/blob/systembiology/blog/posts/${params.slug}.md`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-slate-600 hover:text-protein-400 transition-colors inline-flex items-center gap-1"
          >
            <ExternalLink size={12} /> Edit on GitHub
          </a>
        </header>

        {/* Content */}
        <article className="prose-bio">
          <MDXRemote
            source={post.content}
            options={{
              mdxOptions: {
                remarkPlugins: [remarkGfm],
                rehypePlugins: [rehypeSlug as any],
              },
            }}
          />
        </article>
      </div>
    </div>
  );
}
