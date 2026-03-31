import { Github, Mail, BookOpen } from "lucide-react";
import Logo from "@/components/Logo";

export default function Footer() {
  return (
    <footer className="border-t border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-950">
      <div className="section-container py-10">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Logo size={28} />
              <span className="font-bold text-slate-900 dark:text-white">BioDreamer</span>
            </div>
            <p className="text-sm text-slate-500 dark:text-slate-400 leading-relaxed">
              JEPA-based world models and Active Inference for molecular
              dynamics, protein engineering, and cell reprogramming.
            </p>
          </div>

          {/* Links */}
          <div>
            <h4 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Links</h4>
            <ul className="space-y-2 text-sm text-slate-500 dark:text-slate-400">
              <li>
                <a
                  href="https://github.com/LoqmanSamani/bio_dreamer"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="hover:text-slate-900 dark:hover:text-white transition-colors inline-flex items-center gap-1.5"
                >
                  <Github size={14} /> GitHub
                </a>
              </li>
              <li>
                <a
                  href="mailto:samaniloqman91@gmail.com"
                  className="hover:text-slate-900 dark:hover:text-white transition-colors inline-flex items-center gap-1.5"
                >
                  <Mail size={14} /> Email
                </a>
              </li>
              <li>
                <a
                  href="#"
                  className="hover:text-slate-900 dark:hover:text-white transition-colors inline-flex items-center gap-1.5"
                >
                  <BookOpen size={14} /> Docs (coming soon)
                </a>
              </li>
            </ul>
          </div>

          {/* Modules */}
          <div>
            <h4 className="text-sm font-semibold text-slate-900 dark:text-white mb-3">Modules</h4>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-mol-500" />
                <span className="text-slate-500 dark:text-slate-400">MolDreamer</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-protein-500" />
                <span className="text-slate-500 dark:text-slate-400">ProteinDreamer</span>
              </li>
              <li className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cell-500" />
                <span className="text-slate-500 dark:text-slate-400">CellDreamer</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-slate-200 dark:border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-400 dark:text-slate-500">
            &copy; {new Date().getFullYear()} Loqman Samani. MIT License.
          </p>
          <p className="text-xs text-slate-400 dark:text-slate-500">
            Built with Next.js, Tailwind CSS &amp; TypeScript
          </p>
        </div>
      </div>
    </footer>
  );
}
