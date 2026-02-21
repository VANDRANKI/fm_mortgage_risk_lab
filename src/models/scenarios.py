"""
Scenario definitions and runner for stress testing.

Pre-defined scenarios:
  - baseline
  - mild
  - severe
  - gfc (Global Financial Crisis analog)

Also supports fully custom scenarios via the API.
"""
import json
import logging
from pathlib import Path

import pandas as pd

from src.config.settings import DATA_PROCESSED, MODELS_DIR, SCENARIOS, PORTFOLIO_BASELINE
from src.models.ecl_engine import ECLEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def run_scenario(
    df_portfolio: pd.DataFrame,
    scenario_params: dict,
    engine: ECLEngine | None = None,
) -> dict:
    """
    Run a single stress scenario on the portfolio.

    Parameters
    ----------
    df_portfolio   : DataFrame of loans (must have required feature columns)
    scenario_params: dict with keys: unemployment_shock, hpi_shock, rate_shock
    engine         : pre-loaded ECLEngine (will create one if None)

    Returns
    -------
    dict with aggregated ECL metrics
    """
    engine = engine or ECLEngine()
    return engine.compute_portfolio_ecl(df_portfolio, macro_scenario=scenario_params)


def run_all_scenarios(
    df_portfolio: pd.DataFrame,
    engine: ECLEngine | None = None,
) -> dict[str, dict]:
    """Run all pre-defined scenarios and return results keyed by scenario name."""
    engine = engine or ECLEngine()
    results = {}
    for name, params in SCENARIOS.items():
        log.info("Running scenario: %s", name)
        result = run_scenario(df_portfolio, params, engine)
        result["scenario_name"]  = name
        result["scenario_label"] = params["label"]
        result["color"]          = params["color"]
        results[name] = result
    return results


def precompute_and_cache_scenarios(portfolio_path: Path | None = None) -> None:
    """
    Precompute all scenarios for the standard portfolio and cache to disk.
    This is run once so the API can return results instantly.
    """
    PORTFOLIO_BASELINE.mkdir(parents=True, exist_ok=True)

    if portfolio_path is None:
        portfolio_path = DATA_PROCESSED / "loan_monthly_panel.parquet"

    log.info("Loading portfolio from %s ...", portfolio_path)

    # Use the most recent snapshot per loan (last month observed)
    panel = pd.read_parquet(portfolio_path)
    panel["REPORTING_DATE"] = pd.to_datetime(panel["REPORTING_DATE"])
    latest = (
        panel.sort_values("REPORTING_DATE")
        .groupby("LOAN_SEQUENCE_NUMBER")
        .last()
        .reset_index()
    )
    # Filter to active loans only (not yet zero-balanced)
    active = latest[~latest["IS_ZERO_BALANCE"]].copy()
    log.info("Active loans in portfolio snapshot: %d", len(active))

    engine = ECLEngine()
    results = run_all_scenarios(active, engine)

    # Save full results
    all_path = PORTFOLIO_BASELINE / "all_scenarios.json"
    with open(all_path, "w") as f:
        # Convert numpy types for JSON serialization
        import numpy as np

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        json.dump(results, f, cls=NpEncoder, indent=2)

    log.info("All scenario results saved to %s", all_path)

    # Save vintage curve data (defaulted % by vintage x observation year)
    _build_vintage_curves(panel)


def _build_vintage_curves(panel: pd.DataFrame) -> None:
    """Build vintage default curves for visualisation."""
    panel["obs_year"]     = pd.to_datetime(panel["REPORTING_DATE"]).dt.year
    panel["vintage_year"] = panel["VINTAGE_YEAR"]

    curves = (
        panel.groupby(["vintage_year", "obs_year"])
        .agg(
            total_loans=("LOAN_SEQUENCE_NUMBER", "nunique"),
            defaulted=("IS_DEFAULTED", "sum"),
        )
        .reset_index()
    )
    curves["default_rate"] = curves["defaulted"] / curves["total_loans"]

    out = PORTFOLIO_BASELINE / "vintage_curves.parquet"
    curves.to_parquet(out, index=False)
    log.info("Vintage curves saved -> %s", out)


if __name__ == "__main__":
    precompute_and_cache_scenarios()
