"""
Tests for src/macro/pull_fred.py.

The real FRED API is never called: a fake `fredapi` module is installed in
sys.modules, and the no-key path is exercised through the synthetic generator.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.macro import pull_fred  # noqa: E402

EXPECTED_MONTHS = len(pd.date_range(pull_fred.START_DATE, pull_fred.END_DATE, freq="MS"))
DERIVED = ["hpi_yoy_chg", "unrate_chg_mom"]


class TestSyntheticFallback:
    def test_no_api_key_returns_the_synthetic_panel(self, monkeypatch):
        monkeypatch.setattr(pull_fred, "FRED_API_KEY", "")
        macro = pull_fred.pull_fred_series()
        assert len(macro) == EXPECTED_MONTHS
        assert macro["date"].is_monotonic_increasing
        assert macro["date"].is_unique

    def test_synthetic_panel_has_every_column_the_pipeline_reads(self):
        macro = pull_fred._generate_synthetic_macro()
        needed = {
            "date", "unemployment_rate", "hpi_us", "hpi_yoy_chg",
            "mortgage_rate_30y", "financial_stress_idx", "unrate_chg_mom",
        }
        assert needed <= set(macro.columns)

    def test_synthetic_panel_is_deterministic(self):
        pd.testing.assert_frame_equal(
            pull_fred._generate_synthetic_macro(), pull_fred._generate_synthetic_macro()
        )

    def test_synthetic_values_are_in_plausible_ranges(self):
        macro = pull_fred._generate_synthetic_macro()
        assert macro["unemployment_rate"].between(3.4, 15).all()
        assert macro["mortgage_rate_30y"].between(2, 8).all()
        assert macro["hpi_us"].gt(0).all()

    def test_derived_columns_are_nan_only_where_undefined(self):
        macro = pull_fred._generate_synthetic_macro()
        assert macro["hpi_yoy_chg"].isna().sum() == 12  # needs 12 months of history
        assert macro["unrate_chg_mom"].isna().sum() == 1


class _FakeFred:
    """Stands in for fredapi.Fred: monthly data, except one series that fails."""

    failing_series = "GDPC1"

    def __init__(self, api_key):
        assert api_key == "test-key"

    def get_series(self, series_id, observation_start=None, observation_end=None):
        if series_id == self.failing_series:
            raise ValueError("Bad Request")
        idx = pd.date_range(observation_start, observation_end, freq="MS")
        return pd.Series(np.arange(len(idx), dtype=float) + 100.0, index=idx)


@pytest.fixture
def fake_fredapi(monkeypatch):
    module = types.ModuleType("fredapi")
    module.Fred = _FakeFred
    monkeypatch.setitem(sys.modules, "fredapi", module)
    monkeypatch.setattr(pull_fred, "FRED_API_KEY", "test-key")


class TestLivePath:
    def test_one_failing_series_becomes_nan_and_the_rest_survive(self, fake_fredapi):
        macro = pull_fred.pull_fred_series()
        assert len(macro) == EXPECTED_MONTHS
        assert macro["real_gdp"].isna().all()
        assert macro["unemployment_rate"].notna().all()

    def test_derived_features_are_computed_from_the_pulled_series(self, fake_fredapi):
        macro = pull_fred.pull_fred_series()
        # hpi_us counts 100, 101, ...; year-over-year change at month 12 is 12 / 100 percent.
        assert macro.loc[12, "hpi_yoy_chg"] == pytest.approx(12 / 100 * 100)
        assert macro["hpi_yoy_chg"].iloc[:12].isna().all()
        assert macro["unrate_chg_mom"].iloc[1:].eq(1.0).all()

    def test_columns_follow_the_configured_series_names(self, fake_fredapi):
        macro = pull_fred.pull_fred_series()
        for col in pull_fred.FRED_SERIES.values():
            assert col in macro.columns
        assert macro["date"].dtype.kind == "M"

    def test_missing_fredapi_package_falls_back_to_synthetic(self, monkeypatch):
        monkeypatch.setattr(pull_fred, "FRED_API_KEY", "test-key")
        monkeypatch.setitem(sys.modules, "fredapi", None)  # makes `import fredapi` raise ImportError
        macro = pull_fred.pull_fred_series()
        assert len(macro) == EXPECTED_MONTHS
        assert macro["unemployment_rate"].notna().all()


def test_save_macro_writes_a_parquet_file(monkeypatch, tmp_path):
    monkeypatch.setattr(pull_fred, "FRED_API_KEY", "")
    monkeypatch.setattr(pull_fred, "DATA_MACRO", tmp_path / "macro")
    out = pull_fred.save_macro()
    assert out == tmp_path / "macro" / "fred_macro.parquet"
    assert len(pd.read_parquet(out)) == EXPECTED_MONTHS
