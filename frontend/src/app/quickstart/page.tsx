import Link from "next/link";
import { ArrowRight, BookOpen, Terminal, Copy } from "lucide-react";

export default function QuickStartPage() {
  return (
    <div className="section-container py-16">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4">Quick Start</h1>
        <p className="text-lg text-slate-500 dark:text-slate-400 mb-10">
          Get up and running with BioDreamer in minutes.
        </p>

        {/* Install */}
        <div className="mb-10">
          <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
            <Terminal size={22} className="text-protein-400" />
            Installation
          </h2>
          <div className="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 font-mono text-sm">
            <div className="flex items-center justify-between mb-3">
              <span className="text-slate-400 dark:text-slate-500 text-xs uppercase tracking-wide">
                pip
              </span>
            </div>
            <code className="text-mol-400">pip install bio-dreamer</code>
          </div>
          <p className="text-sm text-slate-400 dark:text-slate-500 mt-3">
            Requires Python ≥ 3.9 and PyTorch ≥ 2.1
          </p>
        </div>

        {/* Basic Usage */}
        <div className="mb-10">
          <h2 className="text-2xl font-semibold text-slate-900 dark:text-white mb-4 flex items-center gap-2">
            <Copy size={22} className="text-protein-400" />
            Basic Usage
          </h2>
          <div className="bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-xl p-5 font-mono text-sm overflow-x-auto">
            <pre className="text-slate-700 dark:text-slate-300">
              <span className="text-protein-400">from</span>{" "}
              <span className="text-mol-400">biodreamer</span>{" "}
              <span className="text-protein-400">import</span>{" "}
              ProteinDreamer{"\n\n"}
              <span className="text-slate-400 dark:text-slate-500">
                # Load a pre-trained world model
              </span>
              {"\n"}
              dreamer = ProteinDreamer.from_pretrained(
              <span className="text-cell-400">
                &quot;biodreamer/protein-dreamer-v1&quot;
              </span>
              ){"\n\n"}
              <span className="text-slate-400 dark:text-slate-500">
                # Define wild-type sequence and objective
              </span>
              {"\n"}
              result = dreamer.design({"\n"}
              {"    "}sequence=
              <span className="text-cell-400">
                &quot;MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH...&quot;
              </span>
              ,{"\n"}
              {"    "}objective=
              <span className="text-cell-400">
                &quot;stability&quot;
              </span>
              ,{"\n"}
              {"    "}n_steps=
              <span className="text-cell-400">10</span>,{"\n"}
              {"    "}n_candidates=
              <span className="text-cell-400">50</span>,{"\n"}){"\n\n"}
              <span className="text-slate-400 dark:text-slate-500">
                # Ranked mutation paths dreamed in the world model
              </span>
              {"\n"}
              <span className="text-protein-400">for</span> candidate{" "}
              <span className="text-protein-400">in</span>{" "}
              result.top_candidates(
              <span className="text-cell-400">5</span>):{"\n"}
              {"    "}
              <span className="text-protein-400">print</span>(
              candidate.mutations, candidate.predicted_fitness)
            </pre>
          </div>
        </div>

        {/* What's Next */}
        <div className="card bg-slate-100/40 dark:bg-slate-900/40">
          <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-3 flex items-center gap-2">
            <BookOpen size={20} className="text-protein-400" />
            What&apos;s Next
          </h2>
          <p className="text-slate-500 dark:text-slate-400 mb-4">
            This section will expand as the project matures. Full documentation
            will cover advanced configuration, custom training, active learning
            loops, and the API server.
          </p>
          <div className="flex flex-wrap gap-3">
            <Link href="/features" className="btn-outline text-sm">
              Explore Features <ArrowRight size={14} />
            </Link>
            <Link href="/how-it-works" className="btn-outline text-sm">
              How It Works <ArrowRight size={14} />
            </Link>
            <a
              href="https://github.com/LoqmanSamani/bio_dreamer"
              target="_blank"
              rel="noopener noreferrer"
              className="btn-outline text-sm"
            >
              Full Documentation (coming soon) <ArrowRight size={14} />
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
