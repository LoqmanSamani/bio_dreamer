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

      // Draw double helix
      const helixPoints = 60;
      const helixHeight = h() * 0.7;
      const helixRadius = Math.min(w() * 0.15, 80);
      const startY = (h() - helixHeight) / 2;

      for (let strand = 0; strand < 2; strand++) {
        const offset = strand * Math.PI;
        ctx.beginPath();
        for (let i = 0; i <= helixPoints; i++) {
          const t = i / helixPoints;
          const y = startY + t * helixHeight;
          const x =
            cx + Math.sin(t * Math.PI * 4 + time * 0.8 + offset) * helixRadius;
          const depth = Math.cos(t * Math.PI * 4 + time * 0.8 + offset);
          const alpha = 0.15 + (depth + 1) * 0.2;

          if (i === 0) {
            ctx.moveTo(x, y);
          } else {
            ctx.lineTo(x, y);
          }

          // Draw base pair connections every few points
          if (i % 5 === 0 && strand === 0) {
            const x2 =
              cx +
              Math.sin(t * Math.PI * 4 + time * 0.8 + Math.PI) * helixRadius;
            ctx.save();
            ctx.globalAlpha = alpha * 0.3;
            ctx.strokeStyle = "#6366f1";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, y);
            ctx.lineTo(x2, y);
            ctx.stroke();
            ctx.restore();
          }
        }
        const colors = ["#10b981", "#6366f1"];
        ctx.strokeStyle = colors[strand];
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.4;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }

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
        const size = 1 + Math.sin(time + i) * 0.8;
        const colors = ["#10b981", "#6366f1", "#f59e0b"];
        ctx.beginPath();
        ctx.arc(px, py, size, 0, Math.PI * 2);
        ctx.fillStyle = colors[i % 3];
        ctx.globalAlpha = 0.2 + Math.sin(time * 0.5 + i) * 0.15;
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
