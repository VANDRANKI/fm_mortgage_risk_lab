"""
Scenario stress-testing endpoints.

POST /scenario/run   – custom macro shock → returns ECL metrics + comparisons
GET  /scenario/list  – return all pre-defined scenario definitions
"""
import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import ScenarioRequest
from src.config.settings import DATA_PROCESSED, SCENARIOS
from src.models.ecl_engine import ECLEngine

router = APIRouter(prefix="/scenario", tags=["Scenario"])
log    = logging.getLogger(__name__)

# Singleton engine (loaded once at startup)
_ENGINE: ECLEngine | None = None
_PORTFOLIO: pd.DataFrame | None = None


def _get_engine() -> ECLEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ECLEngine()
    return _ENGINE


def _get_portfolio() -> pd.DataFrame:
    """Load the active loan portfolio snapshot (cached after first call)."""
    global _PORTFOLIO
    if _PORTFOLIO is not None:
        return _PORTFOLIO

    panel_path = DATA_PROCESSED / "loan_monthly_panel.parquet"
    if not panel_path.exists():
        raise HTTPException(
            status_code=503,
            detail="Loan panel not found. Run the ingestion + feature pipeline first.",
        )

    panel = pd.read_parquet(panel_path)
    panel["REPORTING_DATE"] = pd.to_datetime(panel["REPORTING_DATE"])

    # Latest snapshot per loan
    latest = (
        panel.sort_values("REPORTING_DATE")
        .groupby("LOAN_SEQUENCE_NUMBER")
        .last()
        .reset_index()
    )
    _PORTFOLIO = latest[~latest["IS_ZERO_BALANCE"]].copy()
    log.info("Portfolio loaded: %d active loans", len(_PORTFOLIO))
    return _PORTFOLIO


@router.get("/list")
def list_scenarios():
    """Return all pre-defined scenario definitions."""
    return [
        {
            "name":  name,
            "label": params["label"],
            "color": params["color"],
            "unemployment_shock": params["unemployment_shock"],
            "hpi_shock":          params["hpi_shock"],
            "rate_shock":         params["rate_shock"],
        }
        for name, params in SCENARIOS.items()
    ]


@router.post("/run")
def run_scenario(req: ScenarioRequest):
    """
    Run a custom or pre-defined scenario and return portfolio ECL metrics.
    Also returns a comparison to baseline.
    """
    engine    = _get_engine()
    portfolio = _get_portfolio()

    macro_shock = {
        "unemployment_shock": req.unemployment_shock,
        "hpi_shock":          req.hpi_shock,
        "rate_shock":         req.rate_shock,
    }

    result = engine.compute_portfolio_ecl(portfolio, macro_scenario=macro_shock)
    result["scenario_name"] = req.scenario_name

    # Compute baseline for comparison delta
    baseline_result = engine.compute_portfolio_ecl(portfolio, macro_scenario={})
    baseline_ecl    = baseline_result["total_ecl"]

    result["baseline_ecl"]  = baseline_ecl
    result["ecl_delta"]     = result["total_ecl"] - baseline_ecl
    result["ecl_delta_pct"] = (
        (result["total_ecl"] - baseline_ecl) / max(baseline_ecl, 1) * 100
    )

    return result
