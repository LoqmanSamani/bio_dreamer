"use client";

import { useState } from "react";
import {
  Github,
  Mail,
  BookOpen,
  User,
  Linkedin,
  Twitter,
  ExternalLink,
} from "lucide-react";

const LINKS = [
  {
    icon: Github,
    label: "Project GitHub",
    href: "https://github.com/LoqmanSamani/bio_dreamer",
    description: "Source code, issues, and contributions",
  },
  {
    icon: Mail,
    label: "Email",
    href: "mailto:samaniloqman91@gmail.com",
    description: "samaniloqman91@gmail.com",
  },
  {
    icon: BookOpen,
    label: "ReadTheDocs",
    href: "#",
    description: "Coming soon. Full API documentation",
  },
  {
    icon: User,
    label: "Personal GitHub",
    href: "https://github.com/LoqmanSamani",
    description: "More projects and research",
  },
  {
    icon: Linkedin,
    label: "LinkedIn",
    href: "https://www.linkedin.com/in/loghman-samani-8a5208199/",
    description: "Professional profile and updates",
  },
  {
    icon: Twitter,
    label: "X (Twitter)",
    href: "https://x.com/Loqman_Samani",
    description: "Latest thoughts and announcements",
  },
];

export default function ContactPage() {
  return (
    <div className="section-container py-16">
      <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4 text-center">
        Contact
      </h1>
      <p className="text-lg text-slate-500 dark:text-slate-400 text-center max-w-2xl mx-auto mb-14">
        Get in touch, follow developments, or contribute to BioDreamer.
      </p>

      <div className="max-w-3xl mx-auto grid sm:grid-cols-2 gap-4">
        {LINKS.map((link) => (
          <a
            key={link.label}
            href={link.href}
            target={link.href.startsWith("mailto") ? undefined : "_blank"}
            rel={
              link.href.startsWith("mailto")
                ? undefined
                : "noopener noreferrer"
            }
            className="card group flex items-start gap-4 hover:border-protein-500/30"
          >
            <div className="w-10 h-10 rounded-xl bg-protein-500/10 border border-protein-500/20 flex items-center justify-center shrink-0 group-hover:scale-110 transition-transform">
              <link.icon size={18} className="text-protein-400" />
            </div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="font-medium text-slate-900 dark:text-white text-sm">
                  {link.label}
                </span>
                {link.href !== "#" && !link.href.startsWith("mailto") && (
                  <ExternalLink
                    size={12}
                    className="text-slate-400 dark:text-slate-600 group-hover:text-protein-400 transition-colors"
                  />
                )}
              </div>
              <p className="text-xs text-slate-400 dark:text-slate-500 mt-0.5 truncate">
                {link.description}
              </p>
            </div>
          </a>
        ))}
      </div>

      {/* Contact form */}
      <ContactForm />
    </div>
  );
}

function ContactForm() {
  const [status, setStatus] = useState<"idle" | "sending" | "success" | "error">("idle");

  async function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setStatus("sending");
    const form = e.currentTarget;
    try {
      const res = await fetch("https://formspree.io/f/meeperpk", {
        method: "POST",
        body: new FormData(form),
        headers: { Accept: "application/json" },
      });
      if (res.ok) {
        setStatus("success");
        form.reset();
      } else {
        setStatus("error");
      }
    } catch {
      setStatus("error");
    }
  }

  const inputClass =
    "w-full px-4 py-2.5 bg-slate-100 dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-xl text-slate-900 dark:text-white text-sm placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-protein-500 transition-colors";

  return (
    <div className="max-w-xl mx-auto mt-14">
      <div className="card">
        <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
          Send a Message
        </h2>
        {status === "success" ? (
          <p className="text-center text-green-500 py-8 text-sm font-medium">
            Thanks! Your message has been sent.
          </p>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="name"
                className="block text-sm font-medium text-slate-500 dark:text-slate-400 mb-1"
              >
                Name
              </label>
              <input
                type="text"
                id="name"
                name="name"
                required
                className={inputClass}
                placeholder="Your name"
              />
            </div>
            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-slate-500 dark:text-slate-400 mb-1"
              >
                Email
              </label>
              <input
                type="email"
                id="email"
                name="email"
                required
                className={inputClass}
                placeholder="you@example.com"
              />
            </div>
            <div>
              <label
                htmlFor="message"
                className="block text-sm font-medium text-slate-500 dark:text-slate-400 mb-1"
              >
                Message
              </label>
              <textarea
                id="message"
                name="message"
                rows={4}
                required
                className={`${inputClass} resize-none`}
                placeholder="Your message..."
              />
            </div>
            {status === "error" && (
              <p className="text-red-500 text-xs text-center">
                Something went wrong. Please try again.
              </p>
            )}
            <button
              type="submit"
              disabled={status === "sending"}
              className="btn-primary w-full justify-center disabled:opacity-50"
            >
              <Mail size={16} />
              {status === "sending" ? "Sending…" : "Send Message"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
