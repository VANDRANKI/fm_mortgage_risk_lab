"use client";
import { useState } from "react";
import { motion } from "framer-motion";
import RiskGauge from "@/components/RiskGauge";
import { api, LoanPredictRequest, LoanPredictResponse } from "@/lib/api";
import { fmtCurrency, fmtPct, riskColor, stageLabel, stageColor } from "@/lib/utils";

const US_STATES = [
  "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA",
  "HI","ID","IL","IN","IA","KS","KY","LA","ME","MD",
  "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
  "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC",
  "SD","TN","TX","UT","VT","VA","WA","WV","WI","WY",
];

const DEFAULT_FORM: LoanPredictRequest = {
  credit_score:           720,
  original_ltv:           80,
  original_cltv:          80,
  original_dti:           35,
  original_upb:           350000,
  original_interest_rate: 4.5,
  original_loan_term:     360,
  occ_status:             "P",
  loan_purpose:           "C",
  property_type:          "SF",
  property_state:         "CA",
  product_type:           "FRM",
  channel:                "R",
  first_time_homebuyer:   "N",
  mi_pct:                 0,
  unemployment_shock:     0,
  hpi_shock:              0,
  rate_shock:             0,
};

export default function LoanExplorerPage() {
  const [form,    setForm]    = useState<LoanPredictRequest>(DEFAULT_FORM);
  const [result,  setResult]  = useState<LoanPredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const res = await api.predictLoan(form);
      setResult(res);
    } catch (err: unknown) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const field = (
    label: string,
    key: keyof LoanPredictRequest,
    type: "number" | "select" = "number",
    opts?: string[]
  ) => (
    <div className="space-y-1">
      <label className="text-xs text-gray-500 uppercase tracking-widest block">{label}</label>
      {type === "select" ? (
        <select
          value={String(form[key])}
          onChange={(e) => setForm({ ...form, [key]: e.target.value })}
          className="w-full bg-[#0f172a] border border-[#1e3a5f] rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-[#22d3ee] transition-colors"
        >
          {opts?.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : (
        <input
          type="number"
          value={Number(form[key])}
          onChange={(e) =>
            setForm({ ...form, [key]: parseFloat(e.target.value) || 0 })
          }
          className="w-full bg-[#0f172a] border border-[#1e3a5f] rounded px-2 py-1.5 text-sm text-gray-200 focus:outline-none focus:border-[#22d3ee] transition-colors tabular-nums"
        />
      )}
    </div>
  );

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-[#a78bfa]">Loan Explorer</h1>
        <p className="text-sm text-gray-500 mt-1">
          Enter loan characteristics and get instant PD, LGD, and ECL predictions.
        </p>
      </div>

      <div className="grid lg:grid-cols-[400px_1fr] gap-8">
        {/* ── Input Form ──────────────────────────────────── */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="card">
            <p className="text-xs text-gray-500 mb-4 uppercase tracking-widest">Loan Characteristics</p>
            <div className="grid grid-cols-2 gap-4">
              {field("FICO Score",          "credit_score")}
              {field("LTV (%)",             "original_ltv")}
              {field("CLTV (%)",            "original_cltv")}
              {field("DTI (%)",             "original_dti")}
              {field("UPB ($)",             "original_upb")}
              {field("Interest Rate (%)",   "original_interest_rate")}
              {field("Loan Term (months)",  "original_loan_term")}
              {field("MI Pct (%)",          "mi_pct")}
            </div>
          </div>

          <div className="card">
            <p className="text-xs text-gray-500 mb-4 uppercase tracking-widest">Loan Profile</p>
            <div className="grid grid-cols-2 gap-4">
              {field("Occupancy",    "occ_status",          "select", ["P","I","S"])}
              {field("Purpose",      "loan_purpose",        "select", ["C","N","R","U"])}
              {field("Property",     "property_type",       "select", ["SF","CO","PU","MH"])}
              {field("State",        "property_state",      "select", US_STATES)}
              {field("Product",      "product_type",        "select", ["FRM","ARM"])}
              {field("Channel",      "channel",             "select", ["R","B","C","T"])}
              {field("1st Time HB",  "first_time_homebuyer","select", ["Y","N"])}
            </div>
          </div>

          <div className="card">
            <p className="text-xs text-gray-500 mb-4 uppercase tracking-widest">Macro Scenario (optional)</p>
            <div className="grid grid-cols-3 gap-4">
              {field("Unemp Shock (pp)", "unemployment_shock")}
              {field("HPI Shock (%)",    "hpi_shock")}
              {field("Rate Shock (pp)",  "rate_shock")}
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 rounded-lg bg-[#22d3ee]/10 border border-[#22d3ee]/40 text-[#22d3ee] text-sm font-semibold hover:bg-[#22d3ee]/20 transition-all disabled:opacity-50"
          >
            {loading ? "Computing ..." : "Predict Risk"}
          </button>

          {error && <p className="text-red-400 text-xs">{error}</p>}
        </form>

        {/* ── Results ─────────────────────────────────────── */}
        {result && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-6"
          >
            {/* Risk level badge */}
            <div className="card flex items-center justify-between">
              <div>
                <p className="text-xs text-gray-500 uppercase tracking-widest mb-1">Risk Assessment</p>
                <p className={`text-3xl font-bold ${riskColor(result.risk_level)}`}>
                  {result.risk_level}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {stageLabel(result.ifrs9_stage)} — IFRS 9
                </p>
              </div>
              <div
                className="w-3 h-16 rounded-full"
                style={{ background: stageColor(result.ifrs9_stage) }}
              />
            </div>

            {/* Gauges */}
            <div className="card">
              <p className="text-xs text-gray-500 mb-4 uppercase tracking-widest">Risk Metrics</p>
              <div className="flex justify-around flex-wrap gap-6">
                <RiskGauge
                  label="PD (12-Month)"
                  value={result.pd_12m * 10}  // scale 0-10% → 0-1
                  format={(v) => fmtPct(v / 10)}
                  color="#ef4444"
                />
                <RiskGauge
                  label="LGD"
                  value={result.lgd}
                  format={(v) => fmtPct(v)}
                  color="#f59e0b"
                />
                <RiskGauge
                  label="ECL Rate"
                  value={result.ecl_rate * 100}  // scale 0-1% → 0-1
                  format={(v) => `${(v * 100).toFixed(0)} bps`}
                  color="#a78bfa"
                />
              </div>
            </div>

            {/* Numeric summary */}
            <div className="card">
              <p className="text-xs text-gray-500 mb-4 uppercase tracking-widest">ECL Breakdown</p>
              <div className="space-y-3">
                {[
                  { label: "PD (12-Month)",  value: fmtPct(result.pd_12m) },
                  { label: "LGD",            value: fmtPct(result.lgd) },
                  { label: "EAD",            value: fmtCurrency(result.ead) },
                  { label: "ECL",            value: fmtCurrency(result.ecl) },
                  { label: "ECL Rate",       value: `${(result.ecl_rate * 10000).toFixed(0)} bps` },
                  { label: "IFRS 9 Stage",   value: stageLabel(result.ifrs9_stage) },
                ].map(({ label, value }) => (
                  <div key={label} className="flex justify-between items-center text-sm border-b border-gray-800/40 pb-2">
                    <span className="text-gray-500 text-xs uppercase tracking-wider">{label}</span>
                    <span className="text-gray-200 tabular-nums font-semibold">{value}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ECL formula */}
            <div className="card text-xs text-gray-500 leading-relaxed">
              <p className="text-gray-400 font-semibold mb-2">ECL = PD × LGD × EAD</p>
              <p>
                {fmtPct(result.pd_12m)} × {fmtPct(result.lgd)} × {fmtCurrency(result.ead)}{" "}
                = <strong className="text-[#22d3ee]">{fmtCurrency(result.ecl)}</strong>
              </p>
              <p className="mt-2">
                IFRS 9 Stage {result.ifrs9_stage} loans use a{" "}
                {result.ifrs9_stage === 1 ? "12-month" : "lifetime"} ECL horizon.
              </p>
            </div>
          </motion.div>
        )}

        {/* Placeholder before submit */}
        {!result && !loading && (
          <div className="flex items-center justify-center h-full text-gray-600 text-sm">
            Fill in loan details and click <strong className="text-[#22d3ee] mx-1">Predict Risk</strong>
          </div>
        )}
      </div>
    </div>
  );
}
