"use client";
import { motion } from "framer-motion";

interface RiskGaugeProps {
  label:  string;
  value:  number;  // 0 to 1
  format?: (v: number) => string;
  color?: string;
}

export default function RiskGauge({ label, value, format, color = "#22d3ee" }: RiskGaugeProps) {
  const clipped   = Math.min(Math.max(value, 0), 1);
  const pct       = clipped * 100;
  const displayFn = format ?? ((v: number) => `${(v * 100).toFixed(2)}%`);

  // SVG arc parameters
  const r  = 48;
  const cx = 60;
  const cy = 60;
  const strokeWidth = 8;
  const circumference = Math.PI * r; // half-circle circumference
  const offset = circumference * (1 - clipped);

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width="120" height="72" viewBox="0 0 120 72">
        {/* Background arc */}
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke="#1e3a5f"
          strokeWidth={strokeWidth}
          strokeLinecap="round"
        />
        {/* Filled arc */}
        <motion.path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          fill="none"
          stroke={color}
          strokeWidth={strokeWidth}
          strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        />
        {/* Value text */}
        <text x={cx} y={cy - 4} textAnchor="middle" fill="white" fontSize="13" fontWeight="bold">
          {displayFn(clipped)}
        </text>
        {/* Tick marks at 25% / 50% / 75% */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const angle = Math.PI * (1 - t);
          const x1 = cx + r * Math.cos(angle);
          const y1 = cy - r * Math.sin(angle);
          const x2 = cx + (r - 6) * Math.cos(angle);
          const y2 = cy - (r - 6) * Math.sin(angle);
          return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#334155" strokeWidth={1.5} />;
        })}
      </svg>
      <p className="text-xs text-gray-500 uppercase tracking-widest">{label}</p>
    </div>
  );
}
