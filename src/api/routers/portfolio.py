"""
Portfolio-level endpoints.

GET /portfolio/overview    – headline KPIs for baseline scenario
GET /portfolio/vintages    – default-rate curves by vintage year
GET /portfolio/state_ecl   – ECL breakdown by US state
GET /portfolio/fico_bands  – ECL breakdown by FICO band
GET /portfolio/ltv_bands   – ECL breakdown by LTV band
"""
import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.config.settings import PORTFOLIO_BASELINE

router = APIRouter(prefix="/portfolio", tags=["Portfolio"])
log    = logging.getLogger(__name__)

_SCENARIOS_CACHE: dict | None = None
_VINTAGES_CACHE:  pd.DataFrame | None = None


def _load_scenarios_cache() -> dict:
    global _SCENARIOS_CACHE
    if _SCENARIOS_CACHE is not None:
        return _SCENARIOS_CACHE
    p = PORTFOLIO_BASELINE / "all_scenarios.json"
    if not p.exists():
        raise HTTPException(
            status_code=503,
            detail="Portfolio cache not found. Run recompute_all_scenarios.py first.",
        )
    with open(p) as f:
        _SCENARIOS_CACHE = json.load(f)
    return _SCENARIOS_CACHE


def _load_vintages() -> pd.DataFrame:
    global _VINTAGES_CACHE
    if _VINTAGES_CACHE is not None:
        return _VINTAGES_CACHE
    p = PORTFOLIO_BASELINE / "vintage_curves.parquet"
    if not p.exists():
        raise HTTPException(status_code=503, detail="Vintage curve data not found.")
    _VINTAGES_CACHE = pd.read_parquet(p)
    return _VINTAGES_CACHE


@router.get("/overview")
def get_overview():
    """Return baseline ECL KPIs."""
    cache = _load_scenarios_cache()
    base  = cache.get("baseline", {})
    return {
        "total_ead":         base.get("total_ead", 0),
        "total_ecl":         base.get("total_ecl", 0),
        "ecl_rate":          base.get("ecl_rate", 0),
        "loan_count":        base.get("loan_count", 0),
        "mean_pd":           base.get("mean_pd", 0),
        "mean_lgd":          base.get("mean_lgd", 0),
        "stage_summary":     base.get("by_ifrs9_stage", []),
    }


@router.get("/vintages")
def get_vintage_curves():
    """Return default-rate time series by vintage year."""
    df = _load_vintages()
    # Return list of series, one per vintage
    result = []
    for vintage, grp in df.groupby("vintage_year"):
        grp_sorted = grp.sort_values("obs_year")
        result.append({
            "vintage_year": int(vintage),
            "data": [
                {
                    "obs_year":     int(row["obs_year"]),
                    "default_rate": float(row["default_rate"]),
                    "total_loans":  int(row["total_loans"]),
                }
                for _, row in grp_sorted.iterrows()
            ],
        })
    return result


@router.get("/state_ecl")
def get_state_ecl():
    """Return ECL breakdown by US state under baseline scenario."""
    cache = _load_scenarios_cache()
    state_data = cache.get("baseline", {}).get("by_state", [])
    # Add ISO codes for choropleth
    return state_data


@router.get("/fico_bands")
def get_fico_bands():
    cache = _load_scenarios_cache()
    return cache.get("baseline", {}).get("by_fico_band", [])


@router.get("/ltv_bands")
def get_ltv_bands():
    cache = _load_scenarios_cache()
    return cache.get("baseline", {}).get("by_ltv_band", [])


@router.get("/scenarios/summary")
def get_all_scenario_summary():
    """Return key metrics for all pre-defined scenarios (for comparison table)."""
    cache = _load_scenarios_cache()
    summary = []
    for name, data in cache.items():
        summary.append({
            "scenario_name":  name,
            "scenario_label": data.get("scenario_label", name),
            "color":          data.get("color", "#888"),
            "total_ecl":      data.get("total_ecl", 0),
            "ecl_rate":       data.get("ecl_rate", 0),
            "mean_pd":        data.get("mean_pd", 0),
            "mean_lgd":       data.get("mean_lgd", 0),
        })
    return summary
