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
#
# ORIGINAL_LTV and ORIGINAL_CLTV use the same 999 "not available" sentinel as
# ORIGINAL_DTI per the Freddie Mac layout, but were missing from this map.
# A loan with a genuinely missing LTV/CLTV kept the literal value 999 (i.e.
# 999% LTV) instead of becoming NaN, so it was fed straight into the PD/LGD
# models as a wild numeric outlier -- and into ecl_engine's LTV banding,
# where pd.cut(bins=[...,100,200]) silently drops it as unbanded, since 999
# is outside every bin -- rather than being median-imputed like every other
# missing numeric feature.
SENTINEL_MAP = {
    "CREDIT_SCORE":       9999,
    "ORIGINAL_DTI":       999,
    "ORIGINAL_LTV":       999,
    "ORIGINAL_CLTV":      999,
}


def _parse_date_col(series: pd.Series, name: str) -> pd.Series:
    """Convert YYYYMM string to datetime (first day of month)."""
    cleaned = series.str.strip().replace("", None)
    return pd.to_datetime(cleaned, format="%Y%m", errors="coerce")


# LOAN_PURPOSE keeps "9" as a real value: it is a "not applicable / other" code
# in some vintage years, not a missing sentinel like it is on every other
# string column here.
_NO_SENTINEL_NINE_COLS = {"LOAN_PURPOSE"}


def _replace_numeric_sentinels(df: pd.DataFrame, sentinel_map: dict[str, int]) -> pd.DataFrame:
    """Replace Freddie Mac's "not available" sentinel values with NaN.

    Split out of load_orig_year so it can be unit tested without a synthetic
    raw file, the same way _clean_string_columns is. Columns must already be
    numeric (see load_orig_year's `pd.to_numeric` cast) before this runs.
    """
    for col, sentinel in sentinel_map.items():
        if col not in df.columns:
            continue
        df.loc[df[col] == sentinel, col] = pd.NA
    return df


def _clean_string_columns(df: pd.DataFrame, str_cols: list[str]) -> pd.DataFrame:
    """Strip whitespace and map blank/"9"/"99" sentinels to missing.

    Split out of load_orig_year so it can be unit tested without a synthetic
    raw file: a prior version applied the "9"/"99" sentinel rule to every
    column in str_cols including LOAN_PURPOSE, then tried to undo it for
    LOAN_PURPOSE with a second `.replace("", pd.NA)` call. That second call
    cannot recover a value that has already become <NA>, so the "9" was lost
    silently on every row regardless of the attempted fix.
    """
    for col in str_cols:
        if col not in df.columns:
            continue
        if col in _NO_SENTINEL_NINE_COLS:
            df[col] = df[col].str.strip().replace("", pd.NA)
        else:
            df[col] = df[col].str.strip().replace({"": pd.NA, "9": pd.NA, "99": pd.NA})
    return df


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
    df = _replace_numeric_sentinels(df, SENTINEL_MAP)

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
    df = _clean_string_columns(df, str_cols)

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
