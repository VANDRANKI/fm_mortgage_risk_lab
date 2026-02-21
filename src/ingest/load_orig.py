"""
Load and parse Freddie Mac origination (acquisition) files.

Each year's file is pipe-delimited with no header.
Outputs per-year Parquet files to data/interim/.
"""
import logging
from pathlib import Path

import pandas as pd

from src.config.settings import (
    DATA_RAW, DATA_INTERIM, VINTAGE_YEARS,
    ORIG_COLS, ORIG_DTYPES,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Sentinel values that Freddie Mac uses to mean "missing / not applicable"
MISSING_SENTINELS = {
    "CREDIT_SCORE":       [9999],
    "ORIGINAL_DTI":       [999],
    "NUMBER_OF_BORROWERS":["99"],
    "MI_PCT":             [None],  # 000 means no MI – keep as 0
}

# Numeric cols that need explicit NaN replacement for sentinels
SENTINEL_MAP = {
    "CREDIT_SCORE":       9999,
    "ORIGINAL_DTI":       999,
}


def _parse_date_col(series: pd.Series, name: str) -> pd.Series:
    """Convert YYYYMM string to datetime (first day of month)."""
    cleaned = series.str.strip().replace("", None)
    return pd.to_datetime(cleaned, format="%Y%m", errors="coerce")


def load_orig_year(year: int) -> pd.DataFrame:
    """Load a single origination year, returning a cleaned DataFrame."""
    path = DATA_RAW / f"sample_orig_{year}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Origination file not found: {path}")

    log.info("Loading origination %d from %s", year, path)

    df = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=ORIG_COLS,
        dtype=str,          # read everything as string first; we cast manually
        na_values=["", " "],
        keep_default_na=True,
        low_memory=False,
    )

    log.info("  Raw rows: %d", len(df))

    # ── Cast numeric columns ──────────────────────────────────────────────────
    numeric_cols = [
        "CREDIT_SCORE", "MI_PCT", "NUMBER_OF_UNITS",
        "ORIGINAL_CLTV", "ORIGINAL_DTI", "ORIGINAL_UPB",
        "ORIGINAL_LTV", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Replace sentinel missing values ───────────────────────────────────────
    for col, sentinel in SENTINEL_MAP.items():
        df.loc[df[col] == sentinel, col] = pd.NA

    # ── Cast final dtypes ─────────────────────────────────────────────────────
    int_cols = {
        "CREDIT_SCORE":     "Int16",
        "NUMBER_OF_UNITS":  "Int8",
        "ORIGINAL_LOAN_TERM": "Int16",
    }
    float_cols = {
        "MI_PCT":                  "float32",
        "ORIGINAL_CLTV":           "float32",
        "ORIGINAL_DTI":            "float32",
        "ORIGINAL_LTV":            "float32",
        "ORIGINAL_INTEREST_RATE":  "float32",
    }
    for col, dtype in {**int_cols, **float_cols}.items():
        df[col] = df[col].astype(dtype)

    df["ORIGINAL_UPB"] = df["ORIGINAL_UPB"].astype("float64")

    # ── Parse date columns ────────────────────────────────────────────────────
    df["FIRST_PAYMENT_DATE"] = _parse_date_col(df["FIRST_PAYMENT_DATE"], "FIRST_PAYMENT_DATE")
    df["MATURITY_DATE"]      = _parse_date_col(df["MATURITY_DATE"],      "MATURITY_DATE")

    # ── Strip string columns ──────────────────────────────────────────────────
    str_cols = [
        "FIRST_TIME_HOMEBUYER_FLAG", "OCC_STATUS", "CHANNEL", "PPM_FLAG",
        "PRODUCT_TYPE", "PROPERTY_STATE", "PROPERTY_TYPE", "POSTAL_CODE",
        "LOAN_SEQUENCE_NUMBER", "LOAN_PURPOSE", "NUMBER_OF_BORROWERS",
        "SELLER_NAME", "SERVICER_NAME", "SUPER_CONFORMING_FLAG",
        "PRE_HARP_LOAN_SEQUENCE_NUMBER", "PROGRAM_INDICATOR", "HARP_INDICATOR",
        "PROPERTY_VALUATION_METHOD", "IO_INDICATOR", "MI_CANCELLATION_INDICATOR",
        "MSA",
    ]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().replace({"": pd.NA, "9": pd.NA, "99": pd.NA})

    # Keep loan purpose '9' as unknown (it's a real code in some years)
    # Restore if stripped
    df["LOAN_PURPOSE"] = df["LOAN_PURPOSE"].str.strip().replace("", pd.NA)

    # ── Add vintage year ──────────────────────────────────────────────────────
    df["VINTAGE_YEAR"] = year

    # ── Derived: approx origination date (month before first payment) ─────────
    df["ORIGINATION_DATE"] = df["FIRST_PAYMENT_DATE"] - pd.DateOffset(months=1)

    # ── Drop duplicate loan IDs (keep first occurrence) ───────────────────────
    dupes = df["LOAN_SEQUENCE_NUMBER"].duplicated().sum()
    if dupes:
        log.warning("  %d duplicate LOAN_SEQUENCE_NUMBER rows dropped", dupes)
        df = df.drop_duplicates(subset=["LOAN_SEQUENCE_NUMBER"], keep="first")

    log.info("  Clean rows: %d", len(df))
    return df


def load_and_save_orig_year(year: int, out_dir: Path | None = None) -> Path:
    """Load, parse, and save origination data for one year to Parquet."""
    out_dir = out_dir or DATA_INTERIM
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"orig_{year}.parquet"

    df = load_orig_year(year)
    df.to_parquet(out_path, index=False)
    log.info("  Saved -> %s  (%d rows)", out_path, len(df))
    return out_path


def load_all_orig_years(years: list[int] | None = None) -> None:
    """Load and save all origination years."""
    years = years or VINTAGE_YEARS
    for year in years:
        load_and_save_orig_year(year)
    log.info("All origination years saved to %s", DATA_INTERIM)


if __name__ == "__main__":
    load_all_orig_years()
