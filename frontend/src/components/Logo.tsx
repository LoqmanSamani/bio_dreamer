export default function Logo({ size = 32 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-label="BioDreamer logo"
    >
      <defs>
        <linearGradient id="s1" x1="14" y1="28" x2="50" y2="28" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#34d399" />
        </linearGradient>
        <linearGradient id="s2" x1="14" y1="28" x2="50" y2="28" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#818cf8" />
        </linearGradient>
      </defs>

      {/* Thought bubble */}
      <circle cx="32" cy="28" r="22" stroke="#6366f1" strokeWidth="2.8" strokeOpacity="0.35" fill="#6366f1" fillOpacity="0.04" />

      {/* Thought trail dots */}
      <circle cx="16" cy="56" r="3.5" fill="#6366f1" fillOpacity="0.3" stroke="#6366f1" strokeWidth="1.5" strokeOpacity="0.25" />
      <circle cx="22" cy="49" r="5.5" fill="#6366f1" fillOpacity="0.2" stroke="#6366f1" strokeWidth="1.8" strokeOpacity="0.25" />

      {/* Strand 1 (emerald) */}
      <path d="M 14 28 C 20 14, 26 14, 32 28 C 38 42, 44 42, 50 28" stroke="url(#s1)" strokeWidth="3.5" strokeLinecap="round" fill="none" />

      {/* Strand 2 (indigo) — anti-phase */}
      <path d="M 14 28 C 20 42, 26 42, 32 28 C 38 14, 44 14, 50 28" stroke="url(#s2)" strokeWidth="3.5" strokeLinecap="round" fill="none" />
    </svg>
  );
}
