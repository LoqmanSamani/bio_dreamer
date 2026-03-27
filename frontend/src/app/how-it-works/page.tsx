import { ArrowDown, ArrowRight } from "lucide-react";
import Link from "next/link";

const STEPS = [
  {
    number: "01",
    title: "Observe",
    subtitle: "Encode biological state into latent space",
    description:
      "The domain-specific encoder (ESM-2 for proteins, SE(3)-GNN for molecules, scVI-VAE for cells) maps raw biological data — sequences, coordinates, gene expression — into a compact latent representation z_t.",
    color: "text-mol-400",
    border: "border-mol-500/30",
    bg: "bg-mol-500/5",
  },
  {
    number: "02",
    title: "Dream",
    subtitle: "Simulate outcomes in imagination",
    description:
      "The JEPA dynamics model predicts what happens after an intervention: given the current state z_t and a proposed action (mutation, force change, gene knockout), it generates the predicted next state ẑ_{t+1} — entirely in latent space, no decoder needed.",
    color: "text-protein-400",
    border: "border-protein-500/30",
    bg: "bg-protein-500/5",
  },
  {
    number: "03",
    title: "Evaluate",
    subtitle: "Score fitness and uncertainty",
    description:
      "The reward head predicts target properties (ΔΔG, binding affinity, cell state distance) from the dreamed state. The uncertainty module estimates model confidence — high uncertainty signals promising regions to explore.",
    color: "text-cell-400",
    border: "border-cell-500/30",
    bg: "bg-cell-500/5",
  },
  {
    number: "04",
    title: "Plan",
    subtitle: "Select optimal interventions via Active Inference",
    description:
      "The RL policy (PPO, SAC, or MCTS) rolls out multi-step trajectories inside the world model and selects actions that minimise expected free energy — balancing high fitness (exploitation) with reducing model uncertainty (exploration).",
    color: "text-protein-400",
    border: "border-protein-500/30",
    bg: "bg-protein-500/5",
  },
  {
    number: "05",
    title: "Act & Update",
    subtitle: "Validate and refine the model",
    description:
      "Top-ranked candidates are evaluated by a real oracle (MD simulation, ESMFold, or wet-lab assay). The results update the world model, improving its predictions for the next round — closing the active learning loop.",
    color: "text-mol-400",
    border: "border-mol-500/30",
    bg: "bg-mol-500/5",
  },
];

export default function HowItWorksPage() {
  return (
    <div className="section-container py-16">
      <h1 className="text-4xl font-bold text-white mb-4 text-center">
        How It Works
      </h1>
      <p className="text-lg text-slate-400 text-center max-w-2xl mx-auto mb-14">
        BioDreamer replaces brute-force experimentation with intelligent
        planning in a learned latent simulator.
      </p>

      {/* Pipeline diagram */}
      <div className="max-w-2xl mx-auto mb-16">
        <div className="card p-6 sm:p-8 text-center font-mono text-sm">
          <div className="flex flex-col items-center gap-1">
            <span className="text-slate-300">Observation</span>
            <ArrowDown size={16} className="text-slate-600" />
            <span className="text-mol-400 font-semibold">
              Encoder → Latent State (z_t)
            </span>
            <ArrowDown size={16} className="text-slate-600" />
            <span className="text-protein-400 font-semibold">
              Dynamics Model (z_t, action → z_&#123;t+1&#125;)
            </span>
            <ArrowDown size={16} className="text-slate-600" />
            <span className="text-cell-400 font-semibold">
              Reward Model (z_t → fitness)
            </span>
            <ArrowDown size={16} className="text-slate-600" />
            <span className="text-protein-400 font-semibold">
              Policy (RL agent plans in imagination)
            </span>
            <ArrowDown size={16} className="text-slate-600" />
            <span className="text-slate-300">Predicted Outcome</span>
          </div>
        </div>
      </div>

      {/* Steps */}
      <div className="max-w-3xl mx-auto space-y-6">
        {STEPS.map((step, i) => (
          <div
            key={step.number}
            className={`card ${step.bg} border ${step.border}`}
          >
            <div className="flex gap-5">
              <div
                className={`text-3xl font-extrabold ${step.color} opacity-60 shrink-0`}
              >
                {step.number}
              </div>
              <div>
                <h3 className="text-xl font-semibold text-white mb-1">
                  {step.title}
                </h3>
                <p className={`text-sm ${step.color} mb-2`}>
                  {step.subtitle}
                </p>
                <p className="text-sm text-slate-400 leading-relaxed">
                  {step.description}
                </p>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* CTA */}
      <div className="text-center mt-14">
        <Link href="/features" className="btn-primary">
          See All Features <ArrowRight size={16} />
        </Link>
      </div>
    </div>
  );
}
