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
      {/* Definitions */}
      <defs>
        <linearGradient id="globe-grad" x1="0" y1="0" x2="64" y2="64" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#6366f1" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#10b981" stopOpacity="0.08" />
        </linearGradient>
        <linearGradient id="strand1-grad" x1="10" y1="8" x2="54" y2="56" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#10b981" />
          <stop offset="100%" stopColor="#34d399" />
        </linearGradient>
        <linearGradient id="strand2-grad" x1="10" y1="8" x2="54" y2="56" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#6366f1" />
          <stop offset="100%" stopColor="#818cf8" />
        </linearGradient>
        <clipPath id="globe-clip">
          <circle cx="32" cy="32" r="28" />
        </clipPath>
      </defs>

      {/* Globe fill */}
      <circle cx="32" cy="32" r="28" fill="url(#globe-grad)" />

      {/* Globe ring (equator ellipse) */}
      <ellipse cx="32" cy="32" rx="28" ry="10" stroke="#6366f1" strokeWidth="1" strokeOpacity="0.25" fill="none" />

      {/* Globe meridian */}
      <ellipse cx="32" cy="32" rx="10" ry="28" stroke="#6366f1" strokeWidth="1" strokeOpacity="0.2" fill="none" />

      {/* Globe outer ring */}
      <circle cx="32" cy="32" r="28" stroke="#6366f1" strokeWidth="1.5" strokeOpacity="0.4" fill="none" />

      {/* DNA double helix crossing the globe (clipped) */}
      <g clipPath="url(#globe-clip)">
        {/* Base pair rungs */}
        <line x1="26" y1="14" x2="38" y2="14" stroke="#f59e0b" strokeWidth="1.5" strokeOpacity="0.35" strokeLinecap="round" />
        <line x1="21" y1="22" x2="43" y2="22" stroke="#f59e0b" strokeWidth="1.5" strokeOpacity="0.45" strokeLinecap="round" />
        <line x1="20" y1="30" x2="44" y2="30" stroke="#f59e0b" strokeWidth="1.5" strokeOpacity="0.5" strokeLinecap="round" />
        <line x1="22" y1="38" x2="42" y2="38" stroke="#f59e0b" strokeWidth="1.5" strokeOpacity="0.45" strokeLinecap="round" />
        <line x1="26" y1="46" x2="38" y2="46" stroke="#f59e0b" strokeWidth="1.5" strokeOpacity="0.35" strokeLinecap="round" />

        {/* Strand 1 (emerald) — sinusoidal path */}
        <path
          d="M 30 4 C 18 10, 44 18, 20 26 C 44 34, 18 42, 44 50 C 18 58, 34 62, 34 62"
          stroke="url(#strand1-grad)"
          strokeWidth="2.5"
          strokeLinecap="round"
          fill="none"
        />

        {/* Strand 2 (indigo) — anti-phase sinusoidal path */}
        <path
          d="M 34 4 C 46 10, 20 18, 44 26 C 20 34, 46 42, 20 50 C 46 58, 30 62, 30 62"
          stroke="url(#strand2-grad)"
          strokeWidth="2.5"
          strokeLinecap="round"
          fill="none"
        />

        {/* Small nucleotide dots at rung endpoints */}
        <circle cx="26" cy="14" r="2" fill="#10b981" fillOpacity="0.7" />
        <circle cx="38" cy="14" r="2" fill="#6366f1" fillOpacity="0.7" />
        <circle cx="21" cy="22" r="2" fill="#10b981" fillOpacity="0.8" />
        <circle cx="43" cy="22" r="2" fill="#6366f1" fillOpacity="0.8" />
        <circle cx="20" cy="30" r="2.5" fill="#10b981" fillOpacity="0.9" />
        <circle cx="44" cy="30" r="2.5" fill="#6366f1" fillOpacity="0.9" />
        <circle cx="22" cy="38" r="2" fill="#10b981" fillOpacity="0.8" />
        <circle cx="42" cy="38" r="2" fill="#6366f1" fillOpacity="0.8" />
        <circle cx="26" cy="46" r="2" fill="#10b981" fillOpacity="0.7" />
        <circle cx="38" cy="46" r="2" fill="#6366f1" fillOpacity="0.7" />
      </g>

      {/* Subtle outer glow */}
      <circle cx="32" cy="32" r="28" stroke="#10b981" strokeWidth="0.5" strokeOpacity="0.2" fill="none" />
    </svg>
  );
}
