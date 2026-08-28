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

export type ScenarioDeltaTone = "baseline" | "improved" | "mild" | "severe";

export interface ScenarioDeltaMessage {
  tone: ScenarioDeltaTone;
  text: string;
}

/**
 * Explain a scenario's ECL change vs baseline in plain language.
 *
 * The risk-lab page's HPI slider runs from -40 to +10, and unemployment/rate
 * shocks bottom out at 0, so baseline (all shocks at 0) is not the minimum
 * reachable ECL: moving the HPI slider to any negative value alone (a
 * completely ordinary interaction -- it is the primary control on this page)
 * produces a negative ecl_delta_pct. The three-branch version of this used to
 * only handle ==0, (0, 30), and >=30, so a negative delta matched none of
 * them and the explanatory card rendered nothing at all instead of telling
 * the user their scenario is more favorable than baseline.
 */
export function scenarioDeltaMessage(deltaPct: number): ScenarioDeltaMessage {
  if (deltaPct === 0) {
    return {
      tone: "baseline",
      text: "This is the baseline scenario with no macro shocks applied.",
    };
  }
  if (deltaPct < 0) {
    return {
      tone: "improved",
      text: `Under this scenario, expected losses fall by ${Math.abs(deltaPct).toFixed(1)}% compared to baseline, reflecting more favorable macro conditions than the baseline assumptions.`,
    };
  }
  if (deltaPct < 30) {
    return {
      tone: "mild",
      text: `Under this scenario, expected losses rise by ${deltaPct.toFixed(1)}% compared to baseline, consistent with a mild stress environment.`,
    };
  }
  return {
    tone: "severe",
    text: `Under this scenario, expected losses surge by ${deltaPct.toFixed(1)}% vs baseline, this represents severe stress requiring significant capital buffer.`,
  };
}
