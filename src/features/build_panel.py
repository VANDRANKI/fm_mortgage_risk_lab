"""
Build the monthly loan panel and per-loan outcome summary.

Inputs:
  data/interim/orig_all.parquet
  data/interim/svcg_all.parquet

Outputs:
  data/processed/loan_monthly_panel.parquet   (one row per loan per month)
  data/processed/loan_outcomes.parquet        (one row per loan, final outcomes)
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import (
    DATA_INTERIM, DATA_PROCESSED,
    SERIOUS_DELINQUENCY_CODES,
    LIQUIDATION_ZERO_BALANCE_CODES,
    PREPAYMENT_ZERO_BALANCE_CODE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# Origination columns we bring into the panel (static loan features)
ORIG_KEEP = [
    "LOAN_SEQUENCE_NUMBER", "VINTAGE_YEAR", "ORIGINATION_DATE",
    "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
    "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
    "OCC_STATUS", "LOAN_PURPOSE", "PROPERTY_TYPE", "PROPERTY_STATE",
    "PRODUCT_TYPE", "CHANNEL", "NUMBER_OF_UNITS",
    "FIRST_TIME_HOMEBUYER_FLAG", "MI_PCT", "MATURITY_DATE",
    "POSTAL_CODE", "MSA",
]

# Servicing columns we keep in the panel
SVCG_KEEP = [
    "LOAN_SEQUENCE_NUMBER", "MONTHLY_REPORTING_PERIOD", "REPORTING_DATE",
    "CURRENT_ACTUAL_UPB", "CURRENT_LOAN_DELINQUENCY_STATUS",
    "LOAN_AGE", "REMAINING_MONTHS_TO_MATURITY", "CURRENT_INTEREST_RATE",
    "ZERO_BALANCE_CODE", "ZERO_BALANCE_DATE",
    "IS_SERIOUSLY_DELINQUENT", "IS_ZERO_BALANCE", "IS_PREPAID",
    "IS_LIQUIDATED", "IS_DEFAULTED", "MODIFICATION_FLAG",
    "NET_SALES_PROCEEDS", "ACTUAL_LOSS_CALCULATION", "EXPENSES",
    "MI_RECOVERIES", "NON_MI_RECOVERIES",
]


def build_monthly_panel() -> Path:
    """Merge origination and servicing data into a monthly panel."""
    log.info("Building monthly loan panel ...")

    orig = pd.read_parquet(DATA_INTERIM / "orig_all.parquet", columns=ORIG_KEEP)

    # Read svcg, filtering to only the columns that exist in the file
    svcg_all = pd.read_parquet(DATA_INTERIM / "svcg_all.parquet")
    svcg_keep_actual = [c for c in SVCG_KEEP if c in svcg_all.columns]
    svcg = svcg_all[svcg_keep_actual]
    del svcg_all

    log.info("  orig rows: %d  |  svcg rows: %d", len(orig), len(svcg))

    # ── Merge on LOAN_SEQUENCE_NUMBER ─────────────────────────────────────────
    panel = svcg.merge(orig, on="LOAN_SEQUENCE_NUMBER", how="inner")
    log.info("  Panel rows after join: %d", len(panel))

    # ── Sort ──────────────────────────────────────────────────────────────────
    panel.sort_values(["LOAN_SEQUENCE_NUMBER", "REPORTING_DATE"], inplace=True)
    panel.reset_index(drop=True, inplace=True)

    # ── Derived panel features ────────────────────────────────────────────────
    # Loan age in months (use the reported value; fallback to computed)
    panel["LOAN_AGE_MONTHS"] = panel["LOAN_AGE"].fillna(
        ((panel["REPORTING_DATE"] - panel["ORIGINATION_DATE"])
         / pd.Timedelta(days=30.44)).round().astype("Int16")
    )

    # Delinquency as integer (XX -> 999, RA -> 998)
    panel["DLQ_STATUS_INT"] = (
        panel["CURRENT_LOAN_DELINQUENCY_STATUS"]
        .replace({"XX": "999", "RA": "998"})
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype("int16")
    )

    out_path = DATA_PROCESSED / "loan_monthly_panel.parquet"
    panel.to_parquet(out_path, index=False)
    log.info("Monthly panel saved -> %s  (%d rows)", out_path, len(panel))
    return out_path


def build_loan_outcomes() -> Path:
    """
    Compute per-loan outcome summary from the monthly panel.

    For each loan we compute:
      - defaulted              (bool)   : ever reached 90+ dpd or liquidation
      - prepaid                (bool)   : zero balance with prepayment code
      - liquidated             (bool)   : zero balance with loss code
      - time_to_default_months (int)    : months from origination to first serious delinquency
      - time_to_prepay_months  (int)    : months from origination to prepayment
      - lgd_observed           (float)  : realized LGD for liquidated loans
      - ever_modified          (bool)   : loan was modified at some point
    """
    log.info("Building per-loan outcome summary ...")

    panel = pd.read_parquet(DATA_PROCESSED / "loan_monthly_panel.parquet")

    # ── Per-loan groupby ──────────────────────────────────────────────────────
    grp = panel.groupby("LOAN_SEQUENCE_NUMBER")

    # --- Default event --------------------------------------------------------
    first_default = (
        panel[panel["IS_SERIOUSLY_DELINQUENT"]]
        .groupby("LOAN_SEQUENCE_NUMBER")["LOAN_AGE_MONTHS"]
        .min()
        .rename("time_to_default_months")
    )

    # --- Prepayment -----------------------------------------------------------
    first_prepay = (
        panel[panel["IS_PREPAID"]]
        .groupby("LOAN_SEQUENCE_NUMBER")["LOAN_AGE_MONTHS"]
        .min()
        .rename("time_to_prepay_months")
    )

    # --- Liquidation + LGD ---------------------------------------------------
    # IMPORTANT: When a loan is liquidated, CURRENT_ACTUAL_UPB on the final row
    # is 0 (the balance has already been zeroed). We must use the last
    # *non-zero* UPB across the loan's history as the proper EAD.
    last_nonzero_upb = (
        panel[panel["CURRENT_ACTUAL_UPB"].fillna(0) > 0]
        .groupby("LOAN_SEQUENCE_NUMBER")["CURRENT_ACTUAL_UPB"]
        .last()                  # chronologically last non-zero balance
        .rename("ead_upb")
    )

    liq_rows = panel[panel["IS_LIQUIDATED"]].copy()
    liq_rows = liq_rows.join(last_nonzero_upb, on="LOAN_SEQUENCE_NUMBER")
    liq_rows["EAD"] = liq_rows["ead_upb"]

    # Recoveries (null → 0 for arithmetic)
    liq_rows["net_proceeds_clean"] = (
        liq_rows["NET_SALES_PROCEEDS"].fillna(0)
        if "NET_SALES_PROCEEDS" in liq_rows.columns else 0
    )
    liq_rows["mi_rec_clean"]     = liq_rows["MI_RECOVERIES"].fillna(0)     if "MI_RECOVERIES"     in liq_rows.columns else 0
    liq_rows["non_mi_rec_clean"] = liq_rows["NON_MI_RECOVERIES"].fillna(0) if "NON_MI_RECOVERIES" in liq_rows.columns else 0
    liq_rows["expenses_clean"]   = liq_rows["EXPENSES"].fillna(0)           if "EXPENSES"          in liq_rows.columns else 0

    # Prefer ACTUAL_LOSS_CALCULATION (pre-computed by Freddie Mac) when available
    if "ACTUAL_LOSS_CALCULATION" in liq_rows.columns and liq_rows["ACTUAL_LOSS_CALCULATION"].notna().any():
        liq_rows["gross_loss"] = liq_rows["ACTUAL_LOSS_CALCULATION"].fillna(
            liq_rows["EAD"]
            - liq_rows["net_proceeds_clean"]
            - liq_rows["mi_rec_clean"]
            - liq_rows["non_mi_rec_clean"]
            + liq_rows["expenses_clean"]
        )
    else:
        liq_rows["gross_loss"] = (
            liq_rows["EAD"]
            - liq_rows["net_proceeds_clean"]
            - liq_rows["mi_rec_clean"]
            - liq_rows["non_mi_rec_clean"]
            + liq_rows["expenses_clean"]
        )

    liq_rows["lgd_raw"] = liq_rows["gross_loss"] / liq_rows["EAD"].replace(0, np.nan)

    # Winsorize LGD to [0, 1] for modeling
    liq_rows["lgd_observed"] = liq_rows["lgd_raw"].clip(0, 1)

    lgd_summary = (
        liq_rows.groupby("LOAN_SEQUENCE_NUMBER")[["lgd_observed", "EAD"]]
        .last()  # take the final liquidation row
    )

    # --- Modification -------------------------------------------------------
    mod_flag = (
        panel.groupby("LOAN_SEQUENCE_NUMBER")["MODIFICATION_FLAG"]
        .apply(lambda s: s.notna().any() and (s == "Y").any())
        .rename("ever_modified")
    )

    # --- Max delinquency ever seen -------------------------------------------
    max_dlq = (
        panel.groupby("LOAN_SEQUENCE_NUMBER")["DLQ_STATUS_INT"]
        .max()
        .rename("max_dlq_status")
    )

    # --- Final zero balance info ---------------------------------------------
    zb_summary = (
        panel[panel["IS_ZERO_BALANCE"]]
        .groupby("LOAN_SEQUENCE_NUMBER")
        .last()[["ZERO_BALANCE_CODE", "ZERO_BALANCE_DATE", "LOAN_AGE_MONTHS"]]
        .rename(columns={"LOAN_AGE_MONTHS": "zero_balance_age_months"})
    )

    # ── Base: one row per loan from origination ───────────────────────────────
    orig = pd.read_parquet(
        DATA_INTERIM / "orig_all.parquet",
        columns=["LOAN_SEQUENCE_NUMBER", "VINTAGE_YEAR", "ORIGINATION_DATE",
                 "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
                 "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
                 "OCC_STATUS", "LOAN_PURPOSE", "PROPERTY_TYPE", "PROPERTY_STATE",
                 "PRODUCT_TYPE", "CHANNEL", "FIRST_TIME_HOMEBUYER_FLAG", "MI_PCT"],
    )

    outcomes = orig.copy()

    # Boolean outcome flags
    defaulted_set = set(first_default.index)
    prepaid_set   = set(first_prepay.index)
    liq_set       = set(lgd_summary.index)

    outcomes["defaulted"]   = outcomes["LOAN_SEQUENCE_NUMBER"].isin(defaulted_set)
    outcomes["prepaid"]     = outcomes["LOAN_SEQUENCE_NUMBER"].isin(prepaid_set)
    outcomes["liquidated"]  = outcomes["LOAN_SEQUENCE_NUMBER"].isin(liq_set)

    # Merge timing
    outcomes = outcomes.join(first_default, on="LOAN_SEQUENCE_NUMBER")
    outcomes = outcomes.join(first_prepay,  on="LOAN_SEQUENCE_NUMBER")

    # Merge LGD
    outcomes = outcomes.join(lgd_summary[["lgd_observed", "EAD"]], on="LOAN_SEQUENCE_NUMBER")

    # Merge other summaries
    outcomes = outcomes.join(mod_flag,  on="LOAN_SEQUENCE_NUMBER")
    outcomes = outcomes.join(max_dlq,   on="LOAN_SEQUENCE_NUMBER")
    outcomes = outcomes.join(zb_summary, on="LOAN_SEQUENCE_NUMBER")

    # Observation period (latest month seen for this loan)
    latest_obs = (
        panel.groupby("LOAN_SEQUENCE_NUMBER")["LOAN_AGE_MONTHS"]
        .max()
        .rename("observed_months")
    )
    outcomes = outcomes.join(latest_obs, on="LOAN_SEQUENCE_NUMBER")

    # Fill NaNs for boolean
    outcomes["ever_modified"] = outcomes["ever_modified"].fillna(False)

    out_path = DATA_PROCESSED / "loan_outcomes.parquet"
    outcomes.to_parquet(out_path, index=False)
    log.info("Loan outcomes saved -> %s  (%d rows)", out_path, len(outcomes))
    return out_path


def build_all() -> None:
    build_monthly_panel()
    build_loan_outcomes()
    log.info("Feature panel build complete.")


if __name__ == "__main__":
    build_all()
