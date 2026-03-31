import Link from "next/link";
import { ArrowRight, BookOpen, Github } from "lucide-react";
import HeroAnimation from "@/components/HeroAnimation";

export default function HomePage() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden min-h-[85vh] flex items-center">
        <HeroAnimation />

        {/* Gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-b from-slate-50/60 via-slate-50/80 to-slate-50 dark:from-slate-950/60 dark:via-slate-950/80 dark:to-slate-950 pointer-events-none" />

        <div className="section-container relative z-10 py-20">
          <div className="max-w-3xl">
            <div className="inline-flex items-center gap-2 mb-6 px-4 py-1.5 rounded-full border border-slate-300 dark:border-slate-700 bg-white/60 dark:bg-slate-900/60 text-sm text-slate-600 dark:text-slate-300">
              <span className="w-2 h-2 rounded-full bg-mol-500 animate-pulse-slow" />
              Open-source research framework
            </div>

            <h1 className="text-5xl sm:text-6xl lg:text-7xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-[1.1] mb-6">
              World Models for{" "}
              <span className="gradient-text">Biological Design</span>
            </h1>

            <p className="text-lg sm:text-xl text-slate-500 dark:text-slate-400 leading-relaxed mb-8 max-w-2xl">
              BioDreamer applies JEPA-based world models and Active Inference to
              biological design. Instead of querying expensive real environments,
              the agent learns a latent simulator and plans optimal mutations,
              force-field changes, and gene perturbations in imagination.
            </p>

            <div className="flex flex-wrap gap-4">
              <a
                href="https://github.com/LoqmanSamani/bio_dreamer"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary"
              >
                <Github size={18} />
                View on GitHub
              </a>
              <Link href="/quickstart" className="btn-outline">
                <BookOpen size={18} />
                Quick Start
                <ArrowRight size={16} />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Module Cards */}
      <section className="section-container py-20">
        <h2 className="text-3xl font-bold text-slate-900 dark:text-white text-center mb-4">
          Three Modules, One Architecture
        </h2>
        <p className="text-slate-500 dark:text-slate-400 text-center max-w-2xl mx-auto mb-12">
          BioDreamer operates across three biological scales (atomic, protein,
          and cellular), all sharing a common JEPA world-model backbone.
        </p>

        <div className="grid md:grid-cols-3 gap-6">
          {/* MolDreamer */}
          <div className="card group hover:border-mol-500/40">
            <div className="w-12 h-12 rounded-xl bg-mol-500/10 border border-mol-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <span className="text-mol-400 text-xl">⚛</span>
            </div>
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              MolDreamer
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
              Learns molecular dynamics in latent space. Predicts binding free
              energy and stability via SE(3)-equivariant GNN encoding and latent
              diffusion dynamics.
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="tag">Molecular Dynamics</span>
              <span className="tag">Binding ΔG</span>
              <span className="tag">Stability</span>
            </div>
          </div>

          {/* ProteinDreamer */}
          <div className="card group hover:border-protein-500/40 ring-1 ring-protein-500/10">
            <div className="w-12 h-12 rounded-xl bg-protein-500/10 border border-protein-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <span className="text-protein-400 text-xl">🧬</span>
            </div>
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              ProteinDreamer
              <span className="ml-2 text-xs font-medium text-protein-400 bg-protein-500/10 px-2 py-0.5 rounded-full">
                Core
              </span>
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
              Navigates protein fitness landscapes via dreaming. Plans
              multi-step mutation strategies using JEPA world models and Active
              Inference for principled exploration.
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="tag">Protein Design</span>
              <span className="tag">ΔΔG</span>
              <span className="tag">Active Inference</span>
            </div>
          </div>

          {/* CellDreamer */}
          <div className="card group hover:border-cell-500/40">
            <div className="w-12 h-12 rounded-xl bg-cell-500/10 border border-cell-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
              <span className="text-cell-400 text-xl">🔬</span>
            </div>
            <h3 className="text-xl font-semibold text-slate-900 dark:text-white mb-2">
              CellDreamer
            </h3>
            <p className="text-sm text-slate-500 dark:text-slate-400 mb-3">
              Plans cell reprogramming and perturbation strategies. Uses neural
              ODE/SDE dynamics on scRNA-seq data to simulate multi-step gene
              interventions.
            </p>
            <div className="flex flex-wrap gap-2">
              <span className="tag">scRNA-seq</span>
              <span className="tag">Perturbation</span>
              <span className="tag">Cell Fate</span>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="section-container pb-20">
        <div className="card text-center py-12 bg-gradient-to-br from-slate-100 to-slate-100/40 dark:from-slate-900 dark:to-slate-900/40 border-slate-300/50 dark:border-slate-700/50">
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-3">
            Ready to explore?
          </h2>
          <p className="text-slate-500 dark:text-slate-400 mb-6 max-w-lg mx-auto">
            Dive into the documentation, check out the codebase, or read about
            the science behind BioDreamer.
          </p>
          <div className="flex flex-wrap justify-center gap-4">
            <Link href="/quickstart" className="btn-primary">
              Get Started <ArrowRight size={16} />
            </Link>
            <Link href="/how-it-works" className="btn-outline">
              How It Works
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
