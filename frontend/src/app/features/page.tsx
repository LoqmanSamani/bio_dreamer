import {
  Brain,
  Dna,
  FlaskConical,
  Microscope,
  Network,
  Layers,
  BarChart3,
  RefreshCcw,
  Upload,
  Shield,
  Zap,
  GitBranch,
} from "lucide-react";

const FEATURES = [
  {
    icon: Brain,
    title: "World Model Planning",
    description:
      "Dreams multi-step intervention strategies in latent space before committing to expensive real-world experiments.",
    color: "protein",
  },
  {
    icon: Dna,
    title: "JEPA Architecture",
    description:
      "Joint-Embedding Predictive Architecture encodes observations into latent states and predicts transitions without a decoder during planning.",
    color: "protein",
  },
  {
    icon: FlaskConical,
    title: "Active Inference",
    description:
      "Grounded in the Free Energy Principle — balances exploitation (fitness) and exploration (epistemic uncertainty) automatically.",
    color: "mol",
  },
  {
    icon: Microscope,
    title: "Multi-Scale Biology",
    description:
      "Operates across atomic (MolDreamer), protein (ProteinDreamer), and cellular (CellDreamer) scales with a unified codebase.",
    color: "cell",
  },
  {
    icon: Network,
    title: "Multiple Backends",
    description:
      "Choose from Latent Diffusion JEPA, Energy-Based JEPA, RSSM (DreamerV3), or discrete token (IRIS) world model backends.",
    color: "protein",
  },
  {
    icon: Layers,
    title: "Modular Design",
    description:
      "Swap any component — encoder, dynamics model, reward head, policy — independently. Built with PyTorch and clean abstractions.",
    color: "mol",
  },
  {
    icon: BarChart3,
    title: "Multi-Objective Optimisation",
    description:
      "Optimise for stability, binding affinity, catalytic activity, and expression simultaneously with Pareto-based ranking.",
    color: "cell",
  },
  {
    icon: RefreshCcw,
    title: "Active Learning Loop",
    description:
      "Dream → Propose → Evaluate → Update. Iteratively refines the world model with each round of experimental feedback.",
    color: "mol",
  },
  {
    icon: Upload,
    title: "Hugging Face Hub",
    description:
      "Push and pull pre-trained models via Hugging Face Hub with safetensors serialisation and auto-generated model cards.",
    color: "protein",
  },
  {
    icon: Shield,
    title: "Uncertainty Quantification",
    description:
      "Ensemble, evidential, and MC-dropout methods provide calibrated uncertainty estimates to guide exploration.",
    color: "cell",
  },
  {
    icon: Zap,
    title: "FastAPI Backend",
    description:
      "Production-ready REST API with Redis job queue, GPU worker pool, and real-time progress streaming.",
    color: "mol",
  },
  {
    icon: GitBranch,
    title: "Reproducible Science",
    description:
      "YAML configs, W&B tracking, deterministic seeding, and versioned model checkpoints for fully reproducible experiments.",
    color: "protein",
  },
];

const COLOR_MAP: Record<string, { bg: string; border: string; text: string }> =
  {
    protein: {
      bg: "bg-protein-500/10",
      border: "border-protein-500/20",
      text: "text-protein-400",
    },
    mol: {
      bg: "bg-mol-500/10",
      border: "border-mol-500/20",
      text: "text-mol-400",
    },
    cell: {
      bg: "bg-cell-500/10",
      border: "border-cell-500/20",
      text: "text-cell-400",
    },
  };

export default function FeaturesPage() {
  return (
    <div className="section-container py-16">
      <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4 text-center">
        Features
      </h1>
      <p className="text-lg text-slate-500 dark:text-slate-400 text-center max-w-2xl mx-auto mb-14">
        Everything you need for intelligent, model-based biological design — from
        world model training to candidate ranking.
      </p>

      <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
        {FEATURES.map((f) => {
          const c = COLOR_MAP[f.color];
          return (
            <div key={f.title} className="card group">
              <div
                className={`w-11 h-11 rounded-xl ${c.bg} border ${c.border} flex items-center justify-center mb-4 group-hover:scale-110 transition-transform`}
              >
                <f.icon size={20} className={c.text} />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-2">
                {f.title}
              </h3>
              <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
                {f.description}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
