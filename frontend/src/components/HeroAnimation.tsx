"use client";

import { useEffect, useRef } from "react";

export default function HeroAnimation() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animationId: number;
    let time = 0;

    function resize() {
      if (!canvas) return;
      canvas.width = canvas.offsetWidth * window.devicePixelRatio;
      canvas.height = canvas.offsetHeight * window.devicePixelRatio;
      ctx!.scale(window.devicePixelRatio, window.devicePixelRatio);
    }
    resize();
    window.addEventListener("resize", resize);

    const w = () => canvas!.offsetWidth;
    const h = () => canvas!.offsetHeight;

    function draw() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, w(), h());

      const cx = w() / 2;
      const cy = h() / 2;

      // Double helix parameters
      const helixPoints = 200;
      const helixHeight = h() * 0.85;
      const helixRadius = Math.min(w() * 0.35, 220);
      const startY = (h() - helixHeight) / 2;
      const turns = 5;

      // Store strand coordinates for layered rendering (back→rungs→front)
      const strand1: { x: number; y: number; depth: number }[] = [];
      const strand2: { x: number; y: number; depth: number }[] = [];

      for (let i = 0; i <= helixPoints; i++) {
        const t = i / helixPoints;
        const angle = t * Math.PI * 2 * turns + time * 0.6;
        const y = startY + t * helixHeight;
        const d1 = Math.cos(angle);
        const d2 = Math.cos(angle + Math.PI);
        strand1.push({ x: cx + Math.sin(angle) * helixRadius, y, depth: d1 });
        strand2.push({ x: cx + Math.sin(angle + Math.PI) * helixRadius, y, depth: d2 });
      }

      // Draw back portions of strands (depth < 0)
      function drawStrandSegment(
        strand: typeof strand1,
        color: string,
        backOnly: boolean
      ) {
        ctx!.lineWidth = 9;
        ctx!.lineCap = "round";
        let inSegment = false;
        for (let i = 0; i < strand.length; i++) {
          const isFront = strand[i].depth >= 0;
          const shouldDraw = backOnly ? !isFront : isFront;
          const alpha = backOnly
            ? 0.25 + (1 + strand[i].depth) * 0.2
            : 0.7 + strand[i].depth * 0.3;

          if (shouldDraw) {
            if (!inSegment) {
              ctx!.beginPath();
              ctx!.moveTo(strand[i].x, strand[i].y);
              inSegment = true;
            } else {
              ctx!.lineTo(strand[i].x, strand[i].y);
            }
            ctx!.strokeStyle = color;
            ctx!.globalAlpha = alpha;
          } else if (inSegment) {
            ctx!.stroke();
            inSegment = false;
          }
        }
        if (inSegment) ctx!.stroke();
        ctx!.globalAlpha = 1;
      }

      // Back strands
      drawStrandSegment(strand1, "#10b981", true);
      drawStrandSegment(strand2, "#6366f1", true);

      // Base-pair rungs (every ~10 points)
      const rungSpacing = Math.floor(helixPoints / (turns * 5));
      for (let i = 0; i < helixPoints; i += rungSpacing) {
        const p1 = strand1[i];
        const p2 = strand2[i];
        // Depth-based alpha: visible when both ends face front
        const avgDepth = (p1.depth + p2.depth) / 2;
        const rungAlpha = 0.08 + Math.max(0, avgDepth) * 0.35;

        // Two-color rung (split in the middle like real base pairs)
        const mx = (p1.x + p2.x) / 2;
        const my = (p1.y + p2.y) / 2;

        ctx.lineWidth = 3;
        ctx.lineCap = "round";

        // Left half
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(mx, my);
        ctx.strokeStyle = "#10b981";
        ctx.globalAlpha = rungAlpha;
        ctx.stroke();

        // Right half
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(p2.x, p2.y);
        ctx.strokeStyle = "#6366f1";
        ctx.globalAlpha = rungAlpha;
        ctx.stroke();

        // Small dot at each rung endpoint for nucleotide look
        for (const p of [p1, p2]) {
          ctx.beginPath();
          ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
          ctx.fillStyle = p === p1 ? "#10b981" : "#6366f1";
          ctx.globalAlpha = rungAlpha * 1.4;
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // Front strands (depth >= 0) — drawn on top with glow
      ctx.shadowBlur = 18;
      ctx.shadowColor = "#10b981";
      drawStrandSegment(strand1, "#34d399", false);
      ctx.shadowColor = "#6366f1";
      drawStrandSegment(strand2, "#818cf8", false);
      ctx.shadowBlur = 0;

      // Floating particles
      for (let i = 0; i < 30; i++) {
        const px =
          cx +
          Math.sin(time * 0.3 + i * 2.1) * (helixRadius * 2.5) +
          Math.cos(time * 0.2 + i * 1.3) * 30;
        const py =
          cy +
          Math.cos(time * 0.25 + i * 1.7) * (helixHeight * 0.45) +
          Math.sin(time * 0.15 + i * 2.5) * 20;
        const size = 1.5 + Math.sin(time + i) * 0.8;
        const colors = ["#10b981", "#6366f1", "#f59e0b"];
        ctx.beginPath();
        ctx.arc(px, py, size, 0, Math.PI * 2);
        ctx.fillStyle = colors[i % 3];
        ctx.globalAlpha = 0.25 + Math.sin(time * 0.5 + i) * 0.15;
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      time += 0.015;
      animationId = requestAnimationFrame(draw);
    }

    draw();

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener("resize", resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none"
      aria-hidden="true"
    />
  );
}
