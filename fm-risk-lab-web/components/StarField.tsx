"use client";
import { useEffect, useRef } from "react";

interface Star {
  x: number;
  y: number;
  ox: number; // origin x (for spring-back)
  oy: number; // origin y
  r: number;
  baseOpacity: number;
  opacity: number;
  twinkleSpeed: number;
  twinklePhase: number;
  vx: number; // slow drift
  vy: number;
  color: string;
}

const STAR_COLORS = [
  "200,220,255", // cool white-blue (most common)
  "200,220,255",
  "200,220,255",
  "34,211,238",  // cyan (accent)
  "34,211,238",
  "245,158,11",  // amber (occasional warm star)
];

export default function StarField() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: -9999, y: -9999, active: false });
  const glowRef  = useRef({ x: -9999, y: -9999 }); // lerped glow position
  const rafRef   = useRef<number>(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let W = window.innerWidth;
    let H = window.innerHeight;
    canvas.width  = W;
    canvas.height = H;

    // Build star field
    const NUM = 200;
    const stars: Star[] = Array.from({ length: NUM }, () => {
      const x     = Math.random() * W;
      const y     = Math.random() * H;
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 0.06 + 0.01;
      return {
        x, y, ox: x, oy: y,
        r:            Math.random() * 1.3 + 0.2,
        baseOpacity:  Math.random() * 0.45 + 0.08,
        opacity:      0,
        twinkleSpeed: Math.random() * 0.018 + 0.004,
        twinklePhase: Math.random() * Math.PI * 2,
        vx:           Math.cos(angle) * speed,
        vy:           Math.sin(angle) * speed,
        color:        STAR_COLORS[Math.floor(Math.random() * STAR_COLORS.length)],
      };
    });

    let frame = 0;

    function draw() {
      if (!ctx || !canvas) return;
      ctx.clearRect(0, 0, W, H);
      frame++;

      // Smooth-lerp the glow toward actual mouse position
      const mx = mouseRef.current.x;
      const my = mouseRef.current.y;
      glowRef.current.x += (mx - glowRef.current.x) * 0.055;
      glowRef.current.y += (my - glowRef.current.y) * 0.055;
      const gx = glowRef.current.x;
      const gy = glowRef.current.y;

      // Cursor nebula glow
      if (mouseRef.current.active) {
        const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, 220);
        grad.addColorStop(0,   "rgba(34,211,238,0.08)");
        grad.addColorStop(0.4, "rgba(34,211,238,0.04)");
        grad.addColorStop(1,   "rgba(34,211,238,0)");
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);
      }

      // Stars
      for (const s of stars) {
        // Twinkle
        s.opacity = s.baseOpacity
          + Math.sin(frame * s.twinkleSpeed + s.twinklePhase) * s.baseOpacity * 0.55;

        // Mouse push
        if (mouseRef.current.active) {
          const dx   = s.x - mx;
          const dy   = s.y - my;
          const dist = Math.sqrt(dx * dx + dy * dy);
          const PUSH = 130;
          if (dist < PUSH && dist > 0) {
            const force = (1 - dist / PUSH) * 1.8;
            s.x += (dx / dist) * force;
            s.y += (dy / dist) * force;
          }
        }

        // Spring back toward origin + base drift
        s.x += (s.ox - s.x) * 0.004 + s.vx;
        s.y += (s.oy - s.y) * 0.004 + s.vy;

        // Edge wrap — reset origin so spring doesn't pull back across screen
        if (s.x < -2)  { s.x = W + 2; s.ox = W + 2; }
        if (s.x > W+2) { s.x = -2;    s.ox = -2; }
        if (s.y < -2)  { s.y = H + 2; s.oy = H + 2; }
        if (s.y > H+2) { s.y = -2;    s.oy = -2; }

        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${s.color},${Math.max(0, Math.min(1, s.opacity))})`;
        ctx.fill();
      }

      rafRef.current = requestAnimationFrame(draw);
    }

    draw();

    const onMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY, active: true };
    };
    const onMouseLeave = () => {
      mouseRef.current.active = false;
    };
    const onResize = () => {
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width  = W;
      canvas.height = H;
    };

    window.addEventListener("mousemove",  onMouseMove);
    window.addEventListener("mouseleave", onMouseLeave);
    window.addEventListener("resize",     onResize);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("mousemove",  onMouseMove);
      window.removeEventListener("mouseleave", onMouseLeave);
      window.removeEventListener("resize",     onResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position:      "fixed",
        inset:         0,
        zIndex:        0,
        pointerEvents: "none",
      }}
    />
  );
}
