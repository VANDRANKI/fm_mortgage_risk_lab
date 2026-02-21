"use client";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import KpiCard from "@/components/KpiCard";
import VintageChart from "@/components/VintageChart";
import { api, PortfolioOverview, VintageSeries, StateECL, ScenarioSummary } from "@/lib/api";
import { fmtCurrency, fmtPct, stageColor, stageLabel } from "@/lib/utils";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from "recharts";

export default function PortfolioPage() {
  const [overview,  setOverview]  = useState<PortfolioOverview | null>(null);
  const [vintages,  setVintages]  = useState<VintageSeries[]>([]);
  const [stateData, setStateData] = useState<StateECL[]>([]);
  const [scenarios, setScenarios] = useState<ScenarioSummary[]>([]);
  const [loading,   setLoading]   = useState(true);
  const [error,     setError]     = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      api.getOverview(),
      api.getVintages(),
      api.getStateEcl(),
      api.getAllScenarios(),
    ])
      .then(([ov, vt, st, sc]) => {
        setOverview(ov);
        setVintages(vt);
        setStateData((st as StateECL[]).slice(0, 20));
        setScenarios(sc);
      })
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <LoadingScreen />;
  if (error)   return <ErrorScreen msg={error} />;
  if (!overview) return null;

  const ecl_bps = overview.ecl_rate * 10000;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-[#22d3ee]">Portfolio Overview</h1>
        <p className="text-sm text-gray-500 mt-1">
          Freddie Mac Single-Family · Vintages 2010–2016 · Baseline Macro
        </p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <KpiCard label="Total EAD"    value={fmtCurrency(overview.total_ead)}           accent="cyan"  />
        <KpiCard label="Total ECL"    value={fmtCurrency(overview.total_ecl)}           accent="amber" />
        <KpiCard label="ECL Rate"     value={`${ecl_bps.toFixed(0)} bps`}              accent="amber" />
        <KpiCard label="Active Loans" value={overview.loan_count.toLocaleString()}      accent="cyan"  />
        <KpiCard label="Mean PD"      value={fmtPct(overview.mean_pd)}                 accent="red"   />
        <KpiCard label="Mean LGD"     value={fmtPct(overview.mean_lgd)}                accent="amber" />
      </div>

      {/* IFRS 9 Stages */}
      {overview.stage_summary.length > 0 && (
        <motion.div className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <h2 className="text-xs font-semibold text-gray-400 mb-4 uppercase tracking-widest">
            IFRS 9 Stage Distribution
          </h2>
          <div className="grid grid-cols-3 gap-4">
            {overview.stage_summary.map((s) => (
              <div key={s.ifrs9_stage}
                className="text-center p-4 rounded-lg border"
                style={{
                  borderColor: `${stageColor(s.ifrs9_stage)}40`,
                  background:  `${stageColor(s.ifrs9_stage)}08`,
                }}
              >
                <p className="text-xs text-gray-400 mb-1">{stageLabel(s.ifrs9_stage)}</p>
                <p className="text-2xl font-bold" style={{ color: stageColor(s.ifrs9_stage) }}>
                  {(s.loan_count ?? 0).toLocaleString()}
                </p>
                <p className="text-xs text-gray-500 mt-1">{fmtCurrency(s.ecl ?? 0)} ECL</p>
              </div>
            ))}
          </div>
        </motion.div>
      )}

      {/* Vintage curves */}
      {vintages.length > 0 && (
        <motion.div className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
          <h2 className="text-xs font-semibold text-gray-400 mb-4 uppercase tracking-widest">
            Vintage Default Curves — Cumulative Default Rate %
          </h2>
          <VintageChart data={vintages} />
        </motion.div>
      )}

      {/* ECL by State */}
      {stateData.length > 0 && (
        <motion.div className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
          <h2 className="text-xs font-semibold text-gray-400 mb-4 uppercase tracking-widest">
            ECL Rate by State (Top 20) — Basis Points
          </h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={stateData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f" />
              <XAxis dataKey="PROPERTY_STATE" stroke="#475569" tick={{ fontSize: 10 }} />
              <YAxis
                tickFormatter={(v) => `${(v * 10000).toFixed(0)}`}
                stroke="#475569"
                tick={{ fontSize: 10 }}
                width={40}
                label={{ value: "bps", angle: -90, position: "insideLeft", fill: "#475569", fontSize: 10 }}
              />
              <Tooltip
                contentStyle={{ background: "#0f172a", border: "1px solid #1e3a5f", fontSize: 11 }}
                formatter={(v: number | undefined) => [`${((v ?? 0) * 10000).toFixed(0)} bps`, "ECL Rate"]}
              />
              <Bar dataKey="ecl_rate" radius={[3, 3, 0, 0]}>
                {stateData.map((_, i) => (
                  <Cell key={i} fill="#22d3ee" opacity={0.5 + 0.5 * (1 - i / stateData.length)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>
      )}

      {/* Scenario comparison table */}
      {scenarios.length > 0 && (
        <motion.div className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
          <h2 className="text-xs font-semibold text-gray-400 mb-4 uppercase tracking-widest">
            Pre-Defined Scenario Comparison
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="text-gray-600 border-b border-gray-800">
                  <th className="text-left py-2 pr-4">Scenario</th>
                  <th className="text-right pr-4">Total ECL</th>
                  <th className="text-right pr-4">ECL Rate (bps)</th>
                  <th className="text-right pr-4">Mean PD</th>
                  <th className="text-right">Mean LGD</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((s) => (
                  <tr key={s.scenario_name} className="border-b border-gray-800/40 hover:bg-white/2">
                    <td className="py-2 pr-4 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full" style={{ background: s.color }} />
                      {s.scenario_label}
                    </td>
                    <td className="text-right pr-4 tabular-nums">{fmtCurrency(s.total_ecl)}</td>
                    <td className="text-right pr-4 tabular-nums">{(s.ecl_rate * 10000).toFixed(0)}</td>
                    <td className="text-right pr-4 tabular-nums">{fmtPct(s.mean_pd)}</td>
                    <td className="text-right tabular-nums">{fmtPct(s.mean_lgd)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </div>
  );
}

function LoadingScreen() {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-4">
      <div className="w-8 h-8 border-2 border-[#22d3ee] border-t-transparent rounded-full animate-spin" />
      <p className="text-sm text-gray-500">Loading portfolio data ...</p>
    </div>
  );
}

function ErrorScreen({ msg }: { msg: string }) {
  return (
    <div className="flex flex-col items-center justify-center h-64 gap-4 text-center">
      <p className="text-red-400 text-sm">API Error: {msg}</p>
      <p className="text-xs text-gray-600 max-w-md">
        Make sure the backend is running:
        <br />
        <code className="text-[#22d3ee]">py -3 -m uvicorn src.api.main:app --reload</code>
        <br />
        and the pipeline has been executed:
        <br />
        <code className="text-[#22d3ee]">py -3 run_pipeline.py</code>
      </p>
    </div>
  );
}
