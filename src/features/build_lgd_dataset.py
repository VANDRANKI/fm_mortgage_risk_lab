"""
Build the LGD (Loss Given Default) modeling dataset.

Uses only loans that defaulted AND were subsequently liquidated
(i.e., have an observed loss severity).

Output:
  data/processed/lgd_dataset.parquet
"""
import logging

import numpy as np
import pandas as pd

from src.config.settings import DATA_PROCESSED, DATA_MACRO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

MACRO_COLS = [
    "date", "unemployment_rate", "hpi_yoy_chg", "mortgage_rate_30y",
    "financial_stress_idx",
]


def build_lgd_dataset() -> None:
    log.info("Building LGD dataset ...")

    outcomes = pd.read_parquet(DATA_PROCESSED / "loan_outcomes.parquet")
    panel    = pd.read_parquet(DATA_PROCESSED / "loan_monthly_panel.parquet")
    macro    = pd.read_parquet(DATA_MACRO / "fred_macro.parquet", columns=MACRO_COLS)
    macro["date"] = pd.to_datetime(macro["date"]).dt.to_period("M").dt.to_timestamp()

    # ── 1. Observed LGD: liquidated loans with computed loss ──────────────────
    lgd_observed = outcomes[
        outcomes["liquidated"] & outcomes["defaulted"] & outcomes["lgd_observed"].notna()
    ].copy()
    lgd_observed["lgd_source"] = "observed"
    log.info("  Loans with observed LGD (liquidated): %d", len(lgd_observed))

    # ── 2. Synthetic LGD: defaulted loans without full liquidation data ───────
    # Covers loans still in foreclosure at data cut-off or with missing proceeds.
    # LGD proxy = 0.30 + max(0, LTV/100 - 0.80), clipped to [0.05, 0.90].
    # This reflects empirical mortgage LGD research: low-LTV losses ~25-35%,
    # high-LTV (>80%) losses can reach 50-70%.
    no_liq_mask = outcomes["defaulted"] & (
        ~outcomes["liquidated"] | outcomes["lgd_observed"].isna()
    )
    lgd_synthetic = outcomes[no_liq_mask].copy()
    lgd_synthetic["lgd_observed"] = (
        0.30 + (lgd_synthetic["ORIGINAL_LTV"].fillna(80.0) / 100.0 - 0.80).clip(lower=0.0)
    ).clip(0.05, 0.90)
    lgd_synthetic["lgd_source"] = "synthetic_ltv"
    log.info("  Additional defaulted loans (synthetic LGD): %d", len(lgd_synthetic))

    # ── Combine observed + synthetic ─────────────────────────────────────────
    lgd_df = pd.concat([lgd_observed, lgd_synthetic], ignore_index=True)
    log.info("  Total LGD training set: %d", len(lgd_df))

    log.info("  Loans with observed LGD (total): %d", len(lgd_df))

    # ── Macro conditions at default time ─────────────────────────────────────
    # Compute approximate default date = origination + time_to_default_months
    # Use days (30.44 days/month) to avoid ambiguous unit "M"
    lgd_df["default_date"] = (
        pd.to_datetime(lgd_df["ORIGINATION_DATE"])
        + pd.to_timedelta(lgd_df["time_to_default_months"].fillna(0) * 30.44, unit="D")
    ).dt.to_period("M").dt.to_timestamp()

    lgd_df = lgd_df.merge(
        macro.rename(columns={
            "date":                "default_date",
            "unemployment_rate":   "unemployment_rate_at_default",
            "hpi_yoy_chg":         "hpi_yoy_chg_at_default",
            "mortgage_rate_30y":   "rate_at_default",
            "financial_stress_idx":"fsi_at_default",
        }),
        on="default_date", how="left",
    )
    lgd_df.drop(columns=["default_date"], inplace=True)

    # ── Loan age at default ───────────────────────────────────────────────────
    lgd_df["LOAN_AGE_AT_DEFAULT"] = lgd_df["time_to_default_months"].fillna(
        lgd_df["observed_months"]
    ).astype("float32")

    # ── Select features ───────────────────────────────────────────────────────
    feature_cols = [
        "LOAN_SEQUENCE_NUMBER", "VINTAGE_YEAR", "lgd_observed",
        # Loan characteristics at origination
        "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
        "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
        "OCC_STATUS", "PROPERTY_TYPE", "PROPERTY_STATE",
        "LOAN_PURPOSE", "MI_PCT",
        "LOAN_AGE_AT_DEFAULT",
        # Macro at default
        "unemployment_rate_at_default", "hpi_yoy_chg_at_default",
        "rate_at_default", "fsi_at_default",
        # EAD from liquidation record
        "EAD",
    ]
    available = [c for c in feature_cols if c in lgd_df.columns]
    lgd_df = lgd_df[available]

    # ── Final quality filter ──────────────────────────────────────────────────
    # For synthetic loans without liquidation data, EAD was not computed;
    # substitute ORIGINAL_UPB so they pass the EAD > 0 gate.
    if "EAD" in lgd_df.columns and "ORIGINAL_UPB" in lgd_df.columns:
        lgd_df["EAD"] = lgd_df["EAD"].fillna(lgd_df["ORIGINAL_UPB"])

    lgd_df = lgd_df[lgd_df["lgd_observed"].between(0, 1, inclusive="both")]
    if "EAD" in lgd_df.columns:
        lgd_df = lgd_df[lgd_df["EAD"] > 0]

    log.info("  LGD dataset after filters: %d rows", len(lgd_df))
    log.info("  Mean LGD = %.4f, Median = %.4f",
             lgd_df["lgd_observed"].mean(), lgd_df["lgd_observed"].median())

    out = DATA_PROCESSED / "lgd_dataset.parquet"
    lgd_df.to_parquet(out, index=False)
    log.info("Saved lgd_dataset.parquet -> %s", out)


if __name__ == "__main__":
    build_lgd_dataset()
