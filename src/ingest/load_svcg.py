"""
Load and parse Freddie Mac servicing (performance) files.

Each year's file is pipe-delimited with no header.
Files are large (~270-360 MB each), so we process in chunks.
Outputs per-year Parquet files to data/interim/.
"""
import logging
from pathlib import Path

import pandas as pd

from src.config.settings import (
    DATA_RAW, DATA_INTERIM, VINTAGE_YEARS,
    SVCG_COLS, SVCG_DTYPES,
    SERIOUS_DELINQUENCY_CODES,
    LIQUIDATION_ZERO_BALANCE_CODES,
    PREPAYMENT_ZERO_BALANCE_CODE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Read in 500k-row chunks to keep memory manageable
CHUNK_SIZE = 500_000

# Numeric columns in the servicing file
SVCG_NUMERIC_COLS = [
    "CURRENT_ACTUAL_UPB", "LOAN_AGE", "REMAINING_MONTHS_TO_MATURITY",
    "CURRENT_INTEREST_RATE", "CURRENT_DEFERRED_UPB",
    "MI_RECOVERIES", "NET_SALES_PROCEEDS", "NON_MI_RECOVERIES",
    "EXPENSES", "LEGAL_COSTS", "MAINTENANCE_PRESERVATION_COSTS",
    "TAXES_INSURANCE", "MISCELLANEOUS_EXPENSES", "ACTUAL_LOSS_CALCULATION",
    "MODIFICATION_RELATED_NON_INTEREST_BEARING_UPB", "PRINCIPAL_FORGIVENESS_UPB",
    "ORIGINAL_LIST_PRICE", "CURRENT_LIST_PRICE",
    "VALUE_OF_DELINQUENT_ACCRUED_INTEREST",
    "CURRENT_NON_INTEREST_BEARING_UPB", "CURRENT_UPB_SCHEDULED",
]


def _parse_reporting_period(series: pd.Series) -> pd.Series:
    """Convert YYYYMM string to datetime (first day of month)."""
    return pd.to_datetime(series.str.strip(), format="%Y%m", errors="coerce")


def _process_chunk(chunk: pd.DataFrame, vintage_year: int) -> pd.DataFrame:
    """Clean and enrich a single chunk of servicing data."""
    # ── Cast numeric columns ──────────────────────────────────────────────────
    for col in SVCG_NUMERIC_COLS:
        if col in chunk.columns:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    chunk["LOAN_AGE"]                   = chunk["LOAN_AGE"].astype("Int16")
    chunk["REMAINING_MONTHS_TO_MATURITY"] = chunk["REMAINING_MONTHS_TO_MATURITY"].astype("Int16")

    # ── Parse dates ───────────────────────────────────────────────────────────
    chunk["REPORTING_DATE"] = _parse_reporting_period(chunk["MONTHLY_REPORTING_PERIOD"])
    chunk["ZERO_BALANCE_DATE"] = pd.to_datetime(
        chunk["ZERO_BALANCE_EFFECTIVE_DATE"].str.strip(), format="%Y%m", errors="coerce"
    )

    # ── Standardise delinquency status ───────────────────────────────────────
    chunk["CURRENT_LOAN_DELINQUENCY_STATUS"] = (
        chunk["CURRENT_LOAN_DELINQUENCY_STATUS"].str.strip().str.upper()
    )

    # ── Derived boolean flags ─────────────────────────────────────────────────
    # fillna("") is critical: na_values=[""] converts empty strings to NaN,
    # and NaN.ne("") == True, which would falsely mark all loans as zero-balanced.
    dlq_clean = chunk["CURRENT_LOAN_DELINQUENCY_STATUS"].fillna("").str.upper()
    zb_clean   = chunk["ZERO_BALANCE_CODE"].fillna("").str.strip()

    chunk["IS_SERIOUSLY_DELINQUENT"] = dlq_clean.isin(SERIOUS_DELINQUENCY_CODES)
    chunk["IS_ZERO_BALANCE"]  = zb_clean.ne("")
    chunk["IS_PREPAID"]       = zb_clean.eq(PREPAYMENT_ZERO_BALANCE_CODE)
    chunk["IS_LIQUIDATED"]    = zb_clean.isin(LIQUIDATION_ZERO_BALANCE_CODES)
    chunk["IS_DEFAULTED"]     = chunk["IS_SERIOUSLY_DELINQUENT"] | chunk["IS_LIQUIDATED"]

    # ── Vintage year tag ─────────────────────────────────────────────────────
    chunk["VINTAGE_YEAR"] = vintage_year

    return chunk


def load_svcg_year(year: int) -> pd.DataFrame:
    """
    Load a single servicing year in chunks, returning a cleaned DataFrame.
    Memory-efficient for large files.
    """
    path = DATA_RAW / f"sample_svcg_{year}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Servicing file not found: {path}")

    log.info("Loading servicing %d from %s (chunked)", year, path)

    chunks = []
    total_rows = 0

    reader = pd.read_csv(
        path,
        sep="|",
        header=None,
        names=SVCG_COLS,
        dtype=str,
        na_values=["", " "],
        keep_default_na=True,
        low_memory=False,
        chunksize=CHUNK_SIZE,
    )

    for i, chunk in enumerate(reader):
        chunk = _process_chunk(chunk, year)
        chunks.append(chunk)
        total_rows += len(chunk)
        if (i + 1) % 5 == 0:
            log.info("  Processed %d rows ...", total_rows)

    df = pd.concat(chunks, ignore_index=True)
    log.info("  Total rows for %d: %d", year, len(df))
    return df


def load_and_save_svcg_year(year: int, out_dir: Path | None = None) -> Path:
    """Load, parse, and save servicing data for one year to Parquet."""
    out_dir = out_dir or DATA_INTERIM
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"svcg_{year}.parquet"

    df = load_svcg_year(year)
    df.to_parquet(out_path, index=False)
    log.info("  Saved -> %s  (%d rows)", out_path, len(df))
    return out_path


def load_all_svcg_years(years: list[int] | None = None) -> None:
    """Load and save all servicing years."""
    years = years or VINTAGE_YEARS
    for year in years:
        load_and_save_svcg_year(year)
    log.info("All servicing years saved to %s", DATA_INTERIM)


if __name__ == "__main__":
    load_all_svcg_years()
