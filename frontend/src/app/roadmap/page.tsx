import { CheckCircle2, Circle, Clock } from "lucide-react";

type Status = "done" | "active" | "planned";

interface Milestone {
  title: string;
  status: Status;
  items: string[];
  period: string;
}

const ROADMAP: Milestone[] = [
  {
    title: "Project Foundation",
    status: "done",
    period: "Q1 2026",
    items: [
      "Repository structure and modular architecture",
      "Core base classes (encoder, dynamics, decoder, reward, policy)",
      "YAML configuration system",
      "Project website and blog",
    ],
  },
  {
    title: "ProteinDreamer v0.1 — Energy-Based JEPA",
    status: "active",
    period: "Q1–Q2 2026",
    items: [
      "ESM-2 + GVP-GNN dual encoder with FusionMLP",
      "Energy-Based JEPA predictor with SIGReg",
      "Fitness reward head (ΔΔG, Kd)",
      "PPO policy in latent space",
      "Benchmark on ProteinGym single-mutant landscapes",
    ],
  },
  {
    title: "ProteinDreamer v0.2 — Latent Diffusion JEPA",
    status: "planned",
    period: "Q3 2026",
    items: [
      "Conditional latent diffusion predictor",
      "Active Inference with EFE-based exploration",
      "Multi-step mutation planning on evolutionary paths",
      "Structure-aware encoding via ESMFold",
    ],
  },
  {
    title: "Multi-Objective & Active Learning",
    status: "planned",
    period: "Q4 2026",
    items: [
      "Multi-objective Pareto ranking (stability + activity + expression)",
      "Active learning loop: Dream → Propose → Evaluate → Update",
      "Uncertainty calibration (ensemble, evidential, MC-dropout)",
      "Hugging Face Hub integration for model sharing",
    ],
  },
  {
    title: "MolDreamer & CellDreamer",
    status: "planned",
    period: "2027",
    items: [
      "MolDreamer: SE(3)-equivariant encoder + latent diffusion dynamics",
      "CellDreamer: scVI-VAE encoder + neural ODE/SDE dynamics",
      "FastAPI backend + interactive web interface",
      "Comprehensive benchmarks and documentation",
    ],
  },
];

const STATUS_CONFIG: Record<Status, { icon: typeof CheckCircle2; color: string; bg: string; border: string; label: string }> = {
  done: {
    icon: CheckCircle2,
    color: "text-mol-400",
    bg: "bg-mol-500/10",
    border: "border-mol-500/30",
    label: "Completed",
  },
  active: {
    icon: Clock,
    color: "text-protein-400",
    bg: "bg-protein-500/10",
    border: "border-protein-500/30",
    label: "In Progress",
  },
  planned: {
    icon: Circle,
    color: "text-slate-400 dark:text-slate-500",
    bg: "bg-slate-200/50 dark:bg-slate-800/50",
    border: "border-slate-300 dark:border-slate-700",
    label: "Planned",
  },
};

export default function RoadmapPage() {
  return (
    <div className="section-container py-16">
      <h1 className="text-4xl font-bold text-slate-900 dark:text-white mb-4 text-center">
        Roadmap
      </h1>
      <p className="text-lg text-slate-500 dark:text-slate-400 text-center max-w-2xl mx-auto mb-14">
        Where we are and where we&apos;re headed. BioDreamer is an active
        research project — milestones are updated regularly.
      </p>

      <div className="max-w-3xl mx-auto">
        {/* Timeline */}
        <div className="relative">
          {/* Vertical line */}
          <div className="absolute left-[19px] top-0 bottom-0 w-px bg-slate-200 dark:bg-slate-800" />

          <div className="space-y-8">
            {ROADMAP.map((milestone) => {
              const cfg = STATUS_CONFIG[milestone.status];
              const Icon = cfg.icon;
              return (
                <div key={milestone.title} className="relative pl-14">
                  {/* Icon on timeline */}
                  <div
                    className={`absolute left-0 top-1 w-10 h-10 rounded-xl ${cfg.bg} border ${cfg.border} flex items-center justify-center`}
                  >
                    <Icon size={18} className={cfg.color} />
                  </div>

                  <div className={`card ${cfg.bg} border ${cfg.border}`}>
                    <div className="flex flex-wrap items-center gap-3 mb-3">
                      <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                        {milestone.title}
                      </h3>
                      <span
                        className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${cfg.bg} ${cfg.color} border ${cfg.border}`}
                      >
                        {cfg.label}
                      </span>
                      <span className="text-xs text-slate-500 ml-auto">
                        {milestone.period}
                      </span>
                    </div>
                    <ul className="space-y-1.5">
                      {milestone.items.map((item) => (
                        <li
                          key={item}
                          className="text-sm text-slate-500 dark:text-slate-400 flex items-start gap-2"
                        >
                          <span
                            className={`w-1.5 h-1.5 rounded-full ${
                              milestone.status === "done"
                                ? "bg-mol-500"
                                : milestone.status === "active"
                                ? "bg-protein-500"
                                : "bg-slate-400 dark:bg-slate-600"
                            } mt-1.5 shrink-0`}
                          />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
