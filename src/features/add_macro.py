"""
Join macro economic data into the loan monthly panel.

Inputs:
  data/processed/loan_monthly_panel.parquet
  data/macro/fred_macro.parquet

Output:
  data/processed/loan_monthly_panel_with_macro.parquet
"""
import logging

import pandas as pd

from src.config.settings import DATA_PROCESSED, DATA_MACRO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MACRO_FEATURES = [
    "date", "unemployment_rate", "hpi_us", "hpi_yoy_chg",
    "mortgage_rate_30y", "financial_stress_idx", "unrate_chg_mom",
]


def add_macro_to_panel() -> None:
    log.info("Merging macro data into monthly panel ...")

    panel = pd.read_parquet(DATA_PROCESSED / "loan_monthly_panel.parquet")
    macro = pd.read_parquet(DATA_MACRO / "fred_macro.parquet",
                            columns=MACRO_FEATURES)

    # Normalise the date column to month-start
    macro["date"] = pd.to_datetime(macro["date"]).dt.to_period("M").dt.to_timestamp()
    panel["report_month"] = panel["REPORTING_DATE"].dt.to_period("M").dt.to_timestamp()

    panel = panel.merge(
        macro.rename(columns={"date": "report_month"}),
        on="report_month",
        how="left",
    )
    panel.drop(columns=["report_month"], inplace=True)

    out = DATA_PROCESSED / "loan_monthly_panel_with_macro.parquet"
    panel.to_parquet(out, index=False)
    log.info("Panel with macro saved -> %s  (%d rows)", out, len(panel))


def add_macro_to_outcomes() -> None:
    """
    Attach macro conditions as of origination month and default month
    to the per-loan outcomes table.
    """
    log.info("Merging macro into loan outcomes ...")

    outcomes = pd.read_parquet(DATA_PROCESSED / "loan_outcomes.parquet")
    macro    = pd.read_parquet(DATA_MACRO / "fred_macro.parquet", columns=MACRO_FEATURES)
    macro["date"] = pd.to_datetime(macro["date"]).dt.to_period("M").dt.to_timestamp()

    # --- Macro at origination ------------------------------------------------
    outcomes["orig_month"] = (
        pd.to_datetime(outcomes["ORIGINATION_DATE"])
        .dt.to_period("M").dt.to_timestamp()
    )
    outcomes = outcomes.merge(
        macro.rename(columns={
            "date":                "orig_month",
            "unemployment_rate":   "unrate_at_orig",
            "hpi_us":              "hpi_at_orig",
            "hpi_yoy_chg":         "hpi_yoy_at_orig",
            "mortgage_rate_30y":   "rate_at_orig",
            "financial_stress_idx":"fsi_at_orig",
        }).drop(columns=["unrate_chg_mom"], errors="ignore"),
        on="orig_month", how="left",
    )
    outcomes.drop(columns=["orig_month"], inplace=True)

    # Rate spread = original rate minus market rate at origination
    outcomes["RATE_SPREAD"] = (
        outcomes["ORIGINAL_INTEREST_RATE"] - outcomes["rate_at_orig"]
    )

    out = DATA_PROCESSED / "loan_outcomes_with_macro.parquet"
    outcomes.to_parquet(out, index=False)
    log.info("Loan outcomes with macro saved -> %s", out)


def build_all() -> None:
    add_macro_to_panel()
    add_macro_to_outcomes()


if __name__ == "__main__":
    build_all()
