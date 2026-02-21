/**
 * API client for the Mortgage Credit Risk backend.
 * All calls are centralised here and use NEXT_PUBLIC_API_BASE_URL.
 */

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API ${path} failed (${res.status}): ${err}`);
  }
  return res.json() as Promise<T>;
}

// ── Types ──────────────────────────────────────────────────────────────────────
export interface PortfolioOverview {
  total_ead: number;
  total_ecl: number;
  ecl_rate: number;
  loan_count: number;
  mean_pd: number;
  mean_lgd: number;
  stage_summary: Array<{ ifrs9_stage: number; loan_count: number; ead: number; ecl: number }>;
}

export interface VintagePoint {
  obs_year: number;
  default_rate: number;
  total_loans: number;
}

export interface VintageSeries {
  vintage_year: number;
  data: VintagePoint[];
}

export interface StateECL {
  PROPERTY_STATE: string;
  ead: number;
  ecl: number;
  ecl_rate: number;
  loan_count: number;
  mean_pd: number;
  mean_lgd: number;
}

export interface ScenarioResult {
  scenario_name: string;
  total_ead: number;
  total_ecl: number;
  ecl_rate: number;
  loan_count: number;
  mean_pd: number;
  mean_lgd: number;
  baseline_ecl?: number;
  ecl_delta?: number;
  ecl_delta_pct?: number;
  by_state?: StateECL[];
  by_fico_band?: Array<{ FICO_BAND: string; ecl: number; ecl_rate: number; loan_count: number }>;
}

export interface ScenarioRequest {
  unemployment_shock: number;
  hpi_shock: number;
  rate_shock: number;
  scenario_name?: string;
}

export interface LoanPredictRequest {
  credit_score: number;
  original_ltv: number;
  original_cltv: number;
  original_dti: number;
  original_upb: number;
  original_interest_rate: number;
  original_loan_term: number;
  occ_status: string;
  loan_purpose: string;
  property_type: string;
  property_state: string;
  product_type: string;
  channel: string;
  first_time_homebuyer: string;
  mi_pct: number;
  unemployment_shock: number;
  hpi_shock: number;
  rate_shock: number;
}

export interface LoanPredictResponse {
  pd_12m: number;
  lgd: number;
  ead: number;
  ecl: number;
  ecl_rate: number;
  ifrs9_stage: number;
  risk_level: string;
}

export interface ScenarioDef {
  name: string;
  label: string;
  color: string;
  unemployment_shock: number;
  hpi_shock: number;
  rate_shock: number;
}

export interface ScenarioSummary {
  scenario_name: string;
  scenario_label: string;
  color: string;
  total_ecl: number;
  ecl_rate: number;
  mean_pd: number;
  mean_lgd: number;
}

// ── API functions ─────────────────────────────────────────────────────────────
export const api = {
  getOverview:          ()  => apiFetch<PortfolioOverview>("/portfolio/overview"),
  getVintages:          ()  => apiFetch<VintageSeries[]>("/portfolio/vintages"),
  getStateEcl:          ()  => apiFetch<StateECL[]>("/portfolio/state_ecl"),
  getFicoBands:         ()  => apiFetch<unknown[]>("/portfolio/fico_bands"),
  getAllScenarios:       ()  => apiFetch<ScenarioSummary[]>("/portfolio/scenarios/summary"),
  listScenarioDefs:     ()  => apiFetch<ScenarioDef[]>("/scenario/list"),
  runScenario:          (req: ScenarioRequest) =>
    apiFetch<ScenarioResult>("/scenario/run", { method: "POST", body: JSON.stringify(req) }),
  predictLoan:          (req: LoanPredictRequest) =>
    apiFetch<LoanPredictResponse>("/loan/predict", { method: "POST", body: JSON.stringify(req) }),
};
