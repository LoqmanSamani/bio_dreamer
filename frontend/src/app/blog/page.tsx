import Link from "next/link";
import { Calendar, User, ArrowRight } from "lucide-react";
import { getAllPosts } from "@/lib/blog";

export default function BlogPage() {
  const posts = getAllPosts();

  return (
    <div className="section-container py-16">
      <h1 className="text-4xl font-bold text-white mb-2">Blog</h1>
      <p className="text-lg text-slate-400 mb-12">
        Research updates, technical deep-dives, and project milestones.
      </p>

      {posts.length === 0 ? (
        <p className="text-slate-500">No posts yet. Check back soon!</p>
      ) : (
        <div className="max-w-3xl space-y-6">
          {posts.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="card block group hover:border-protein-500/30"
            >
              <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 mb-3">
                <span className="inline-flex items-center gap-1">
                  <Calendar size={12} /> {post.date}
                </span>
                <span className="inline-flex items-center gap-1">
                  <User size={12} /> {post.author}
                </span>
              </div>

              <h2 className="text-xl font-semibold text-white mb-2 group-hover:text-protein-400 transition-colors">
                {post.title}
              </h2>

              <p className="text-sm text-slate-400 mb-4 leading-relaxed">
                {post.summary}
              </p>

              <div className="flex items-center justify-between">
                <div className="flex flex-wrap gap-2">
                  {post.tags.map((tag) => (
                    <span key={tag} className="tag">
                      {tag}
                    </span>
                  ))}
                </div>
                <span className="text-protein-400 text-sm font-medium inline-flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  Read more <ArrowRight size={14} />
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
