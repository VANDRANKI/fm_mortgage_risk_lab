"""
Build the PD (Probability of Default) modeling dataset.

Strategy:
  Observation point = 12 months after origination (seasoned snapshot).
  Label             = loan defaulted within the NEXT 12 months?

We also build a 24-month PD dataset for term-structure analysis.

Output:
  data/processed/pd_12m_dataset.parquet
  data/processed/pd_24m_dataset.parquet
"""
import logging

import numpy as np
import pandas as pd

from src.config.settings import DATA_PROCESSED, DATA_MACRO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MACRO_COLS = [
    "date", "unemployment_rate", "hpi_yoy_chg", "mortgage_rate_30y",
    "financial_stress_idx", "unrate_chg_mom",
]


def _build_pd_dataset(observation_age: int, horizon: int) -> pd.DataFrame:
    """
    observation_age: months since origination at which we observe the loan.
    horizon:         months forward we look for a default event.
    """
    log.info(
        "Building PD dataset: obs_age=%d months, horizon=%d months",
        observation_age, horizon
    )

    panel = pd.read_parquet(DATA_PROCESSED / "loan_monthly_panel.parquet")
    macro = pd.read_parquet(DATA_MACRO / "fred_macro.parquet", columns=MACRO_COLS)
    macro["date"] = pd.to_datetime(macro["date"]).dt.to_period("M").dt.to_timestamp()

    # ── Step 1: get the snapshot at observation_age for each loan ─────────────
    obs_snap = panel[panel["LOAN_AGE_MONTHS"] == observation_age].copy()
    obs_snap = obs_snap.drop_duplicates(subset=["LOAN_SEQUENCE_NUMBER"])

    # Only keep loans that were still active (not yet zero-balanced) at obs point
    obs_snap = obs_snap[~obs_snap["IS_ZERO_BALANCE"]]

    log.info("  Loans at observation age %d: %d", observation_age, len(obs_snap))

    # ── Step 2: look-ahead default label ─────────────────────────────────────
    # For each loan, find max delinquency in months [obs_age, obs_age + horizon]
    future = panel[
        (panel["LOAN_AGE_MONTHS"] > observation_age)
        & (panel["LOAN_AGE_MONTHS"] <= observation_age + horizon)
    ].copy()

    future_default = (
        future.groupby("LOAN_SEQUENCE_NUMBER")["IS_DEFAULTED"]
        .any()
        .rename("PD_TARGET")
    )

    obs_snap = obs_snap.join(future_default, on="LOAN_SEQUENCE_NUMBER")
    # Loans not seen in future window (ended early) get PD_TARGET=False
    obs_snap["PD_TARGET"] = obs_snap["PD_TARGET"].fillna(False).astype(int)

    # ── Step 3: macro conditions at observation date ───────────────────────────
    obs_snap["obs_month"] = obs_snap["REPORTING_DATE"].dt.to_period("M").dt.to_timestamp()
    obs_snap = obs_snap.merge(
        macro.rename(columns={"date": "obs_month"}),
        on="obs_month", how="left",
    )
    obs_snap.drop(columns=["obs_month"], inplace=True)

    # ── Step 4: rate spread ────────────────────────────────────────────────────
    obs_snap["RATE_SPREAD"] = (
        obs_snap["ORIGINAL_INTEREST_RATE"] - obs_snap["mortgage_rate_30y"]
    )
    obs_snap["LOAN_AGE_AT_OBS"] = observation_age

    # ── Step 5: select final feature set ──────────────────────────────────────
    feature_cols = [
        "LOAN_SEQUENCE_NUMBER", "VINTAGE_YEAR", "PD_TARGET",
        # Loan-level
        "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
        "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
        "OCC_STATUS", "LOAN_PURPOSE", "PROPERTY_TYPE", "PROPERTY_STATE",
        "PRODUCT_TYPE", "CHANNEL",
        "FIRST_TIME_HOMEBUYER_FLAG", "MI_PCT",
        "RATE_SPREAD", "LOAN_AGE_AT_OBS",
        # Current state at obs
        "CURRENT_ACTUAL_UPB", "CURRENT_LOAN_DELINQUENCY_STATUS", "DLQ_STATUS_INT",
        # Macro
        "unemployment_rate", "hpi_yoy_chg", "mortgage_rate_30y",
        "financial_stress_idx", "unrate_chg_mom",
    ]
    available = [c for c in feature_cols if c in obs_snap.columns]
    result = obs_snap[available].copy()

    default_rate = result["PD_TARGET"].mean() * 100
    log.info(
        "  Final dataset: %d loans, default rate = %.2f%%",
        len(result), default_rate
    )
    return result


def build_pd_datasets() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # 12-month PD (observation at 12 months, horizon 12 months)
    pd_12 = _build_pd_dataset(observation_age=12, horizon=12)
    pd_12.to_parquet(DATA_PROCESSED / "pd_12m_dataset.parquet", index=False)
    log.info("Saved pd_12m_dataset.parquet (%d rows)", len(pd_12))

    # 24-month PD (observation at 12 months, horizon 24 months)
    pd_24 = _build_pd_dataset(observation_age=12, horizon=24)
    pd_24.to_parquet(DATA_PROCESSED / "pd_24m_dataset.parquet", index=False)
    log.info("Saved pd_24m_dataset.parquet (%d rows)", len(pd_24))

    # PD at origination (observation at 0 months, horizon 12 months)
    pd_orig = _build_pd_dataset(observation_age=0, horizon=12)
    pd_orig.to_parquet(DATA_PROCESSED / "pd_orig_dataset.parquet", index=False)
    log.info("Saved pd_orig_dataset.parquet (%d rows)", len(pd_orig))


if __name__ == "__main__":
    build_pd_datasets()
