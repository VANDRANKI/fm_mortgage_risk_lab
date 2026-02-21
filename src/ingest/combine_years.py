"""
Combine per-year Parquet files into single multi-year tables.

Reads:  data/interim/orig_YYYY.parquet + data/interim/svcg_YYYY.parquet
Writes: data/interim/orig_all.parquet + data/interim/svcg_all.parquet
"""
import logging
from pathlib import Path

import pandas as pd

from src.config.settings import DATA_INTERIM, VINTAGE_YEARS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def combine_orig(years: list[int] | None = None, out_dir: Path | None = None) -> Path:
    """Concatenate all origination Parquet files into one."""
    years   = years   or VINTAGE_YEARS
    out_dir = out_dir or DATA_INTERIM
    out_path = out_dir / "orig_all.parquet"

    dfs = []
    for year in years:
        p = out_dir / f"orig_{year}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run load_orig.py first."
            )
        df = pd.read_parquet(p)
        dfs.append(df)
        log.info("  orig_%d: %d rows", year, len(df))

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_parquet(out_path, index=False)
    log.info("Combined orig -> %s  (%d rows)", out_path, len(combined))
    return out_path


def combine_svcg(years: list[int] | None = None, out_dir: Path | None = None) -> Path:
    """Concatenate all servicing Parquet files into one."""
    years   = years   or VINTAGE_YEARS
    out_dir = out_dir or DATA_INTERIM
    out_path = out_dir / "svcg_all.parquet"

    dfs = []
    for year in years:
        p = out_dir / f"svcg_{year}.parquet"
        if not p.exists():
            raise FileNotFoundError(
                f"Missing {p}. Run load_svcg.py first."
            )
        df = pd.read_parquet(p)
        dfs.append(df)
        log.info("  svcg_%d: %d rows", year, len(df))

    combined = pd.concat(dfs, ignore_index=True)
    # Sort by loan and time for downstream processing
    combined.sort_values(
        ["LOAN_SEQUENCE_NUMBER", "REPORTING_DATE"], inplace=True
    )
    combined.reset_index(drop=True, inplace=True)
    combined.to_parquet(out_path, index=False)
    log.info("Combined svcg -> %s  (%d rows)", out_path, len(combined))
    return out_path


def combine_all(years: list[int] | None = None) -> None:
    years = years or VINTAGE_YEARS
    combine_orig(years)
    combine_svcg(years)
    log.info("Combination complete.")


if __name__ == "__main__":
    combine_all()
