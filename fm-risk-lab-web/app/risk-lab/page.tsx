"use client";
import { useState, useCallback, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import ScenarioSlider from "@/components/ScenarioSlider";
import KpiCard from "@/components/KpiCard";
import { api, ScenarioResult } from "@/lib/api";
import { fmtCurrency, fmtPct } from "@/lib/utils";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, Cell,
} from "recharts";

const PRESETS = [
  { label: "Baseline",    unemployment_shock: 0,   hpi_shock:   0,  rate_shock: 0 },
  { label: "Mild",        unemployment_shock: 2,   hpi_shock:  -5,  rate_shock: 1 },
  { label: "Severe",      unemployment_shock: 5,   hpi_shock: -20,  rate_shock: 2.5 },
  { label: "GFC-Like",   unemployment_shock: 8,   hpi_shock: -33,  rate_shock: 0.5 },
];

export default function RiskLabPage() {
  const [unemploymentShock, setUnemploymentShock] = useState(0);
  const [hpiShock,          setHpiShock]          = useState(0);
  const [rateShock,         setRateShock]          = useState(0);
  const [result,            setResult]             = useState<ScenarioResult | null>(null);
  const [loading,           setLoading]            = useState(false);
  const [error,             setError]              = useState<string | null>(null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const runScenario = useCallback(
    (u: number, h: number, r: number) => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(async () => {
        setLoading(true);
        setError(null);
        try {
          const res = await api.runScenario({
            unemployment_shock: u,
            hpi_shock: h,
            rate_shock: r,
            scenario_name: "custom",
          });
          setResult(res);
        } catch (e: unknown) {
          setError((e as Error).message);
        } finally {
          setLoading(false);
        }
      }, 500);
    },
    []
  );

  useEffect(() => { runScenario(unemploymentShock, hpiShock, rateShock); }, []);

  const handleSlider = (setter: (v: number) => void) => (v: number) => {
    setter(v);
    runScenario(
      setter === setUnemploymentShock ? v : unemploymentShock,
      setter === setHpiShock          ? v : hpiShock,
      setter === setRateShock         ? v : rateShock,
    );
  };

  const delta_pct = result?.ecl_delta_pct ?? 0;
  const accentColor = delta_pct === 0 ? "#22d3ee" : delta_pct < 20 ? "#f59e0b" : "#ef4444";

  // Radar data for risk dimensions
  const radarData = result ? [
    { metric: "PD",    value: Math.min(result.mean_pd    * 2000, 100) },
    { metric: "LGD",   value: Math.min(result.mean_lgd   * 100,  100) },
    { metric: "ECL%",  value: Math.min(result.ecl_rate   * 5000, 100) },
    { metric: "Unemp", value: Math.min((4.87 + unemploymentShock) * 6, 100) },
    { metric: "HPI",   value: Math.max(0, 50 + hpiShock * 2) },
  ] : [];

  const ficoBands = result?.by_fico_band?.slice(0, 6) ?? [];

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-[#f59e0b]">Risk Lab — Stress Testing</h1>
        <p className="text-sm text-gray-500 mt-1">
          Adjust macro shocks below and watch the portfolio ECL react in real time.
        </p>
      </div>

      <div className="grid lg:grid-cols-[380px_1fr] gap-8">
        {/* ── Left: Controls ────────────────────────────────────────── */}
        <div className="space-y-6">
          {/* Presets */}
          <div className="card">
            <p className="text-xs text-gray-500 mb-3 uppercase tracking-widest">Preset Scenarios</p>
            <div className="grid grid-cols-2 gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p.label}
                  onClick={() => {
                    setUnemploymentShock(p.unemployment_shock);
                    setHpiShock(p.hpi_shock);
                    setRateShock(p.rate_shock);
                    runScenario(p.unemployment_shock, p.hpi_shock, p.rate_shock);
                  }}
                  className="text-xs px-3 py-2 rounded-md border border-[#1e3a5f] text-gray-300 hover:border-[#22d3ee]/40 hover:text-[#22d3ee] transition-all"
                >
                  {p.label}
                </button>
              ))}
            </div>
          </div>

          {/* Sliders */}
          <div className="card space-y-6">
            <p className="text-xs text-gray-500 uppercase tracking-widest">Macro Shocks</p>
            <ScenarioSlider
              label="Unemployment Rate"
              unit="pp"
              min={0} max={12} step={0.5}
              value={unemploymentShock}
              onChange={handleSlider(setUnemploymentShock)}
              color="#ef4444"
            />
            <ScenarioSlider
              label="House Price Index (YoY)"
              unit="%"
              min={-40} max={10} step={1}
              value={hpiShock}
              onChange={handleSlider(setHpiShock)}
              color="#f59e0b"
            />
            <ScenarioSlider
              label="Mortgage Rate"
              unit="pp"
              min={0} max={5} step={0.25}
              value={rateShock}
              onChange={handleSlider(setRateShock)}
              color="#a78bfa"
            />
          </div>

          {/* Radar */}
          {radarData.length > 0 && (
            <div className="card">
              <p className="text-xs text-gray-500 mb-3 uppercase tracking-widest">Risk Dimensions</p>
              <ResponsiveContainer width="100%" height={200}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="#1e3a5f" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: "#94a3b8", fontSize: 11 }} />
                  <Radar dataKey="value" stroke={accentColor} fill={accentColor} fillOpacity={0.15} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>

        {/* ── Right: Results ─────────────────────────────────────────── */}
        <div className="space-y-6">
          {/* Loading / Error */}
          {loading && (
            <div className="flex items-center gap-3 text-sm text-gray-400">
              <span className="w-4 h-4 border border-[#22d3ee] border-t-transparent rounded-full animate-spin" />
              Computing ECL ...
            </div>
          )}
          {error && <p className="text-red-400 text-sm">{error}</p>}

          <AnimatePresence mode="wait">
            {result && !loading && (
              <motion.div
                key={`${unemploymentShock}-${hpiShock}-${rateShock}`}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.3 }}
                className="space-y-6"
              >
                {/* KPIs */}
                <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
                  <KpiCard label="Total ECL"    value={fmtCurrency(result.total_ecl)}    accent="amber" animate={false} />
                  <KpiCard label="ECL Rate"     value={`${(result.ecl_rate * 10000).toFixed(0)} bps`} accent="amber" animate={false} />
                  <KpiCard label="Mean PD"      value={fmtPct(result.mean_pd)}            accent="red"   animate={false} />
                  <KpiCard label="Mean LGD"     value={fmtPct(result.mean_lgd)}           accent="amber" animate={false} />
                  <KpiCard
                    label="ECL vs Baseline"
                    value={`${delta_pct >= 0 ? "+" : ""}${delta_pct.toFixed(1)}%`}
                    accent={delta_pct < 0 ? "green" : delta_pct < 30 ? "amber" : "red"}
                    animate={false}
                  />
                  <KpiCard
                    label="ECL Delta"
                    value={fmtCurrency(result.ecl_delta ?? 0)}
                    accent={delta_pct < 20 ? "amber" : "red"}
                    animate={false}
                  />
                </div>

                {/* Micro-copy */}
                <div className="card text-xs text-gray-400 leading-relaxed">
                  {delta_pct === 0 && (
                    <span>This is the <strong className="text-[#22d3ee]">baseline</strong> scenario with no macro shocks applied.</span>
                  )}
                  {delta_pct > 0 && delta_pct < 30 && (
                    <span>Under this scenario, expected losses rise by <strong className="text-[#f59e0b]">{delta_pct.toFixed(1)}%</strong> compared to baseline—consistent with a <strong className="text-gray-200">mild stress</strong> environment.</span>
                  )}
                  {delta_pct >= 30 && (
                    <span>Under this scenario, expected losses surge by <strong className="text-[#ef4444]">{delta_pct.toFixed(1)}%</strong> vs baseline—this represents <strong className="text-gray-200">severe stress</strong> requiring significant capital buffer.</span>
                  )}
                </div>

                {/* ECL by FICO band */}
                {ficoBands.length > 0 && (
                  <div className="card">
                    <p className="text-xs text-gray-500 mb-4 uppercase tracking-widest">ECL by FICO Band</p>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={ficoBands} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
                        <XAxis dataKey="FICO_BAND" stroke="#475569" tick={{ fontSize: 10 }} />
                        <YAxis tickFormatter={(v) => `${(v * 10000).toFixed(0)}`} stroke="#475569" tick={{ fontSize: 10 }} width={40} />
                        <Tooltip
                          contentStyle={{ background: "#0f172a", border: "1px solid #1e3a5f", fontSize: 11 }}
                          formatter={(v: number | undefined) => [`${((v ?? 0) * 10000).toFixed(0)} bps`, "ECL Rate"]}
                        />
                        <Bar dataKey="ecl_rate" radius={[3, 3, 0, 0]}>
                          {ficoBands.map((_, i) => (
                            <Cell key={i} fill={accentColor} opacity={0.5 + 0.5 * (1 - i / ficoBands.length)} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}
