/** Format numbers as compact currency strings */
export function fmtCurrency(n: number): string {
  if (Math.abs(n) >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(2)}M`;
  if (Math.abs(n) >= 1e3) return `$${(n / 1e3).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

/** Format as percentage */
export function fmtPct(n: number, decimals = 2): string {
  return `${(n * 100).toFixed(decimals)}%`;
}

/** Format as basis points */
export function fmtBps(n: number): string {
  return `${(n * 10000).toFixed(0)} bps`;
}

/** Map risk level to Tailwind colour class */
export function riskColor(level: string): string {
  const map: Record<string, string> = {
    LOW:       "text-emerald-400",
    MEDIUM:    "text-amber-400",
    HIGH:      "text-orange-500",
    "VERY HIGH": "text-red-500",
  };
  return map[level] ?? "text-gray-400";
}

/** Map IFRS 9 stage number to display label */
export function stageLabel(stage: number): string {
  return { 1: "Stage 1", 2: "Stage 2", 3: "Stage 3" }[stage] ?? "Unknown";
}

export function stageColor(stage: number): string {
  return { 1: "#22d3ee", 2: "#f59e0b", 3: "#ef4444" }[stage] ?? "#888";
}

/** Debounce – returns a debounced version of fn */
export function debounce<T extends (...args: unknown[]) => void>(fn: T, ms: number): T {
  let timer: ReturnType<typeof setTimeout>;
  return ((...args: unknown[]) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  }) as T;
}
