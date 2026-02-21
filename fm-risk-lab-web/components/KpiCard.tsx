"use client";
import { motion } from "framer-motion";

interface KpiCardProps {
  label:    string;
  value:    string;
  sub?:     string;
  accent?:  "cyan" | "amber" | "red" | "green";
  animate?: boolean;
}

const accentClasses = {
  cyan:  "text-[#22d3ee] border-[#22d3ee]/20",
  amber: "text-[#f59e0b] border-[#f59e0b]/20",
  red:   "text-[#ef4444] border-[#ef4444]/20",
  green: "text-emerald-400 border-emerald-400/20",
};

export default function KpiCard({
  label, value, sub, accent = "cyan", animate = true,
}: KpiCardProps) {
  const cls = accentClasses[accent];
  return (
    <motion.div
      className={`card border ${cls} glow-${accent}`}
      initial={animate ? { opacity: 0, y: 16 } : false}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <p className="text-xs text-gray-500 uppercase tracking-widest mb-2">{label}</p>
      <p className={`text-3xl font-bold ${cls.split(" ")[0]}`}>{value}</p>
      {sub && <p className="text-xs text-gray-400 mt-1">{sub}</p>}
    </motion.div>
  );
}
