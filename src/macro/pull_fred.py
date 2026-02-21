"""
Pull macroeconomic time series from the FRED API.

Uses fredapi (https://github.com/mortada/fredapi).
Requires FRED_API_KEY in .env.

Outputs:
  data/macro/fred_macro.parquet  – monthly panel, 2010-01 to 2017-12
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import DATA_MACRO, FRED_SERIES, FRED_API_KEY

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Date range to cover all vintage years + performance window
START_DATE = "2010-01-01"
END_DATE   = "2023-12-31"


def pull_fred_series() -> pd.DataFrame:
    """Pull all configured FRED series and return a monthly panel DataFrame."""
    if not FRED_API_KEY:
        log.warning("FRED_API_KEY not set. Generating synthetic macro data for demo.")
        return _generate_synthetic_macro()

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
    except ImportError:
        log.warning("fredapi not installed. Generating synthetic macro data.")
        return _generate_synthetic_macro()

    # Build a monthly date index
    monthly_idx = pd.date_range(START_DATE, END_DATE, freq="MS")
    macro = pd.DataFrame(index=monthly_idx)

    for series_id, col_name in FRED_SERIES.items():
        try:
            log.info("  Pulling %s -> %s", series_id, col_name)
            raw = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            # Resample everything to monthly (forward-fill quarterly / weekly series)
            monthly = raw.resample("MS").last().reindex(monthly_idx).ffill()
            macro[col_name] = monthly.values
        except Exception as e:
            log.warning("  Failed to pull %s: %s – using NaN", series_id, e)
            macro[col_name] = np.nan

    macro.index.name = "date"
    macro.reset_index(inplace=True)

    # ── Derived macro features ────────────────────────────────────────────────
    # HPI year-over-year change (%)
    macro = macro.sort_values("date").reset_index(drop=True)
    macro["hpi_yoy_chg"] = macro["hpi_us"].pct_change(12) * 100

    # Unemployment rate change (MoM)
    macro["unrate_chg_mom"] = macro["unemployment_rate"].diff()

    # Mortgage rate level (already in %)
    # Financial stress index (already in std-dev units)

    return macro


def _generate_synthetic_macro() -> pd.DataFrame:
    """
    Generate realistic-looking synthetic macro data when FRED API is unavailable.
    Based on historical patterns 2010-2023.
    """
    dates = pd.date_range(START_DATE, END_DATE, freq="MS")
    n = len(dates)

    rng = np.random.default_rng(42)

    # Unemployment: starts at 9.8% in 2010, declines to ~3.5% by 2019, rises 2020, recovers
    t = np.arange(n)
    unemp_trend = 9.8 * np.exp(-0.015 * t) + 3.5 * (1 - np.exp(-0.015 * t))
    # COVID spike
    covid_spike = np.where((t >= 121) & (t <= 130), np.interp(t, [121, 125, 130], [0, 10, 0]), 0)
    unemp = (unemp_trend + covid_spike + rng.normal(0, 0.15, n)).clip(3.4, 15)

    # HPI: base 100 in 2010, grows ~4% / year with dip in COVID start
    hpi = 100.0
    hpi_series = [hpi]
    for i in range(1, n):
        growth = 0.003 + rng.normal(0, 0.005)
        if 121 <= i <= 122:
            growth = -0.01
        hpi = hpi * (1 + growth)
        hpi_series.append(hpi)
    hpi_arr = np.array(hpi_series)

    # 30-yr mortgage rate: ~4.7% in 2010, ranges 3.3-7.1 through 2023
    rate_base = np.array(
        [4.71, 4.87, 4.84, 4.84, 4.64, 4.51, 4.39, 4.35, 4.22, 4.07, 4.00, 3.96,  # 2010
         3.92, 3.92, 4.84, 4.84, 4.66, 4.51, 4.36, 4.27, 4.11, 4.07, 4.00, 3.96,  # 2011
         3.87, 3.87, 3.95, 3.91, 3.79, 3.67, 3.62, 3.60, 3.52, 3.47, 3.44, 3.35,  # 2012
         3.34, 3.53, 3.57, 3.45, 3.35, 3.98, 4.37, 4.46, 4.49, 4.28, 4.26, 4.46,  # 2013
         4.43, 4.30, 4.34, 4.34, 4.29, 4.16, 4.13, 4.12, 4.16, 3.92, 3.99, 3.86,  # 2014
         3.73, 3.71, 3.77, 3.67, 3.84, 4.02, 4.04, 3.91, 3.90, 3.76, 3.94, 3.97,  # 2015
         3.87, 3.65, 3.69, 3.61, 3.61, 3.57, 3.44, 3.43, 3.46, 3.42, 3.77, 4.20,  # 2016
         4.15, 4.17, 4.20, 4.03, 4.01, 3.89, 3.96, 3.88, 3.83, 3.90, 3.92, 3.99,  # 2017
         4.03, 4.33, 4.44, 4.47, 4.59, 4.57, 4.53, 4.51, 4.63, 4.83, 4.87, 4.64,  # 2018
         4.46, 4.37, 4.27, 4.14, 4.07, 3.82, 3.81, 3.60, 3.49, 3.69, 3.70, 3.74,  # 2019
         3.62, 3.47, 3.45, 3.31, 3.23, 3.16, 3.02, 2.96, 2.89, 2.81, 2.72, 2.67,  # 2020
         2.65, 2.81, 3.08, 3.06, 2.96, 2.98, 2.87, 2.84, 2.90, 3.07, 3.10, 3.11,  # 2021
         3.45, 3.76, 4.17, 4.98, 5.23, 5.52, 5.41, 5.22, 6.02, 6.90, 6.81, 6.36,  # 2022
         6.09, 6.26, 6.54, 6.39, 6.43, 6.71, 6.81, 7.09, 7.20, 7.62, 7.44, 6.82,  # 2023
        ]
    )
    if len(rate_base) < n:
        rate_base = np.pad(rate_base, (0, n - len(rate_base)), mode="edge")
    else:
        rate_base = rate_base[:n]

    # Financial stress index (std-devs, normal ≈ 0, crisis >> 0)
    fsi = rng.normal(-0.5, 0.3, n)
    fsi[121:127] += 3.0  # COVID spike

    macro = pd.DataFrame({
        "date":                dates,
        "unemployment_rate":   unemp,
        "hpi_us":              hpi_arr,
        "mortgage_rate_30y":   rate_base + rng.normal(0, 0.05, n),
        "financial_stress_idx": fsi,
        "real_gdp":            np.nan,  # quarterly – leave NaN for synthetic
        "cpi":                 np.nan,
    })

    macro["hpi_yoy_chg"]    = macro["hpi_us"].pct_change(12) * 100
    macro["unrate_chg_mom"] = macro["unemployment_rate"].diff()
    log.info("Synthetic macro generated for %d months", n)
    return macro


def save_macro() -> Path:
    DATA_MACRO.mkdir(parents=True, exist_ok=True)
    macro = pull_fred_series()
    out = DATA_MACRO / "fred_macro.parquet"
    macro.to_parquet(out, index=False)
    log.info("Macro data saved -> %s  (%d rows)", out, len(macro))
    return out


if __name__ == "__main__":
    save_macro()
