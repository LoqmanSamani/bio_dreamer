import { Clock, ArrowRight } from "lucide-react";
import Link from "next/link";

export default function ExamplesPage() {
  return (
    <div className="section-container py-16">
      <h1 className="text-4xl font-bold text-white mb-4 text-center">
        Examples
      </h1>
      <p className="text-lg text-slate-400 text-center max-w-2xl mx-auto mb-14">
        Interactive demos and use cases showing BioDreamer in action.
      </p>

      {/* Coming Soon */}
      <div className="max-w-2xl mx-auto">
        <div className="card text-center py-16 border-dashed border-slate-700">
          <div className="w-16 h-16 rounded-2xl bg-protein-500/10 border border-protein-500/20 flex items-center justify-center mx-auto mb-6">
            <Clock size={28} className="text-protein-400" />
          </div>
          <h2 className="text-2xl font-semibold text-white mb-3">
            Coming Soon
          </h2>
          <p className="text-slate-400 max-w-lg mx-auto mb-8 leading-relaxed">
            We&apos;re working on interactive examples and demo notebooks. When
            ready, this section will include:
          </p>

          <div className="grid sm:grid-cols-2 gap-4 text-left max-w-lg mx-auto mb-8">
            {[
              {
                icon: "🧬",
                label: "Protein stability optimisation",
                detail: "Walk through a ProteinDreamer mutation trajectory",
              },
              {
                icon: "⚛",
                label: "Molecular dynamics dreaming",
                detail: "MolDreamer latent-space MD rollouts",
              },
              {
                icon: "🔬",
                label: "Cell reprogramming planning",
                detail: "CellDreamer perturbation strategies on scRNA-seq",
              },
              {
                icon: "📓",
                label: "Jupyter notebooks",
                detail: "End-to-end training and inference tutorials",
              },
            ].map((item) => (
              <div
                key={item.label}
                className="flex gap-3 p-3 rounded-xl bg-slate-800/40"
              >
                <span className="text-xl shrink-0">{item.icon}</span>
                <div>
                  <p className="text-sm font-medium text-white">
                    {item.label}
                  </p>
                  <p className="text-xs text-slate-500">{item.detail}</p>
                </div>
              </div>
            ))}
          </div>

          <Link href="/blog" className="btn-outline text-sm">
            Read the Blog in the meantime <ArrowRight size={14} />
          </Link>
        </div>
      </div>
    </div>
  );
}
