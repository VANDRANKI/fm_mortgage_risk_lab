"""
Unit tests for src/models/ecl_engine.py.

These exercise the engine without any trained model artifacts on disk, so the
heuristic fallbacks and the defensive branches are the code under test. That is
deliberate: the fallbacks exist for portfolios that are missing columns, and
those were the paths that used to raise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ecl_engine import (  # noqa: E402
    ECLEngine,
    STAGE2_MIN_DLQ,
    STAGE2_MIN_PD,
    STAGE3_MIN_DLQ,
    STAGE3_MIN_PD,
)

BASELINE_MACRO = pd.Series(
    {
        "unemployment_rate": 4.87,
        "hpi_yoy_chg": 5.6,
        "mortgage_rate_30y": 3.65,
        "financial_stress_idx": -0.5,
        "unrate_chg_mom": 0.0,
    }
)


@pytest.fixture
def engine():
    """An engine with no model artifacts, so every prediction takes the heuristic path."""
    e = ECLEngine.__new__(ECLEngine)
    e.dataset_name = "pd_12m"
    e.pd_model = None
    e.lgd_model = None
    e.lgd_pre = None
    e._macro_baseline = BASELINE_MACRO.copy()
    return e


class TestHeuristicLGD:
    def test_missing_hpi_does_not_raise(self, engine):
        """Regression: `(1 + hpi/100).clip(...)` raised AttributeError when
        hpi_yoy_chg was absent, because the default 0.0 is a plain Python float."""
        macro = pd.Series({"unemployment_rate": 5.0})
        out = engine._heuristic_lgd(pd.DataFrame({"ORIGINAL_LTV": [80.0, 95.0]}), macro)
        assert np.isfinite(out).all()

    def test_lgd_within_bounds(self, engine):
        out = engine._heuristic_lgd(
            pd.DataFrame({"ORIGINAL_LTV": [20.0, 80.0, 150.0]}), BASELINE_MACRO
        )
        assert ((out >= 0.05) & (out <= 0.90)).all()

    def test_higher_ltv_gives_higher_lgd(self, engine):
        out = engine._heuristic_lgd(
            pd.DataFrame({"ORIGINAL_LTV": [60.0, 120.0]}), BASELINE_MACRO
        )
        assert out[1] >= out[0]


class TestEADFallback:
    def test_no_exposure_column_does_not_raise(self, engine):
        """Regression: `result.get("ORIGINAL_UPB", 0).fillna(0)` raised
        AttributeError, because DataFrame.get returns the bare int default."""
        df = pd.DataFrame({"CREDIT_SCORE": [700, 720]})
        out = engine.compute_ecl(df)
        assert (out["ead"] == 0).all()

    def test_falls_back_to_original_upb(self, engine):
        df = pd.DataFrame({"CREDIT_SCORE": [700], "ORIGINAL_UPB": [250_000.0]})
        out = engine.compute_ecl(df)
        assert out["ead"].iloc[0] == pytest.approx(250_000.0)

    def test_prefers_current_upb(self, engine):
        df = pd.DataFrame(
            {
                "CREDIT_SCORE": [700],
                "ORIGINAL_UPB": [250_000.0],
                "CURRENT_ACTUAL_UPB": [200_000.0],
            }
        )
        out = engine.compute_ecl(df)
        assert out["ead"].iloc[0] == pytest.approx(200_000.0)


class TestIFRS9Staging:
    def test_delinquency_drives_stage(self, engine):
        df = pd.DataFrame({"DLQ_STATUS_INT": [0, 1, 2, 3]})
        low_pd = np.full(4, 0.001)
        assert list(engine.classify_ifrs9_stage(df, low_pd)) == [1, 2, 2, 3]

    def test_pd_alone_can_force_stage_3(self, engine):
        df = pd.DataFrame({"DLQ_STATUS_INT": [0]})
        assert engine.classify_ifrs9_stage(df, np.array([0.5]))[0] == 3

    def test_missing_dlq_column_defaults_to_current(self, engine):
        df = pd.DataFrame({"CREDIT_SCORE": [700, 720]})
        assert list(engine.classify_ifrs9_stage(df, np.full(2, 0.001))) == [1, 1]

    def test_thresholds_are_in_months_not_days(self):
        """DLQ_STATUS_INT counts months delinquent, so the cutoffs are small
        integers. The constants this replaced were written in days (0 and 60)
        and could not be compared against the data as-is."""
        assert STAGE2_MIN_DLQ == 1
        assert STAGE3_MIN_DLQ == 3
        assert STAGE2_MIN_PD < STAGE3_MIN_PD


class TestPortfolioAggregation:
    def test_minimal_portfolio_survives(self, engine):
        """Regression: pd.cut indexed CREDIT_SCORE and ORIGINAL_LTV directly and
        raised KeyError, even though agg_by guards for missing segments."""
        df = pd.DataFrame({"PROPERTY_STATE": ["NY", "CA", "NY"]})
        out = engine.compute_portfolio_ecl(df)

        assert out["loan_count"] == 3
        assert out["by_fico_band"] == []
        assert out["by_ltv_band"] == []
        assert len(out["by_state"]) == 2

    def test_bands_populated_when_columns_present(self, engine):
        df = pd.DataFrame(
            {
                "CREDIT_SCORE": [610, 705, 780],
                "ORIGINAL_LTV": [55.0, 85.0, 105.0],
                "ORIGINAL_UPB": [100_000.0, 200_000.0, 300_000.0],
            }
        )
        out = engine.compute_portfolio_ecl(df)

        assert len(out["by_fico_band"]) > 0
        assert len(out["by_ltv_band"]) > 0
        assert out["total_ead"] == pytest.approx(600_000.0)

    def test_ecl_rate_is_finite_on_zero_exposure(self, engine):
        df = pd.DataFrame({"PROPERTY_STATE": ["NY"]})
        out = engine.compute_portfolio_ecl(df)
        assert np.isfinite(out["ecl_rate"])

    def test_fico_band_boundaries_match_their_labels(self, engine):
        """Regression: pd.cut defaults to right-inclusive bins, so a credit
        score of exactly 660 fell into the "620-659" bucket instead of
        "660-699" (and 700 into "660-699", 740 into "700-739", 800 into
        "740-799") -- every band boundary was mislabeled one band low."""
        df = pd.DataFrame({
            "CREDIT_SCORE": [660, 700, 740, 800],
            "ORIGINAL_UPB": [100_000.0] * 4,
        })
        out = engine.compute_portfolio_ecl(df)
        band_labels = {b["FICO_BAND"] for b in out["by_fico_band"]}
        assert "660-699" in band_labels
        assert "700-739" in band_labels
        assert "740-799" in band_labels
        assert "800+" in band_labels
        # None of the boundary scores should have spilled into the band below.
        assert "620-659" not in band_labels

    def test_ltv_band_boundaries_match_their_labels(self, engine):
        """Same right-inclusive-bin bug as FICO_BAND: an LTV of exactly 70,
        80, 90, or 100 fell into the band below its label."""
        df = pd.DataFrame({
            "ORIGINAL_LTV": [70.0, 80.0, 90.0, 100.0],
            "ORIGINAL_UPB": [100_000.0] * 4,
        })
        out = engine.compute_portfolio_ecl(df)
        band_labels = {b["LTV_BAND"] for b in out["by_ltv_band"]}
        assert "70-79" in band_labels
        assert "80-89" in band_labels
        assert "90-99" in band_labels
        assert "100+" in band_labels
        assert "60-69" not in band_labels


class TestScenario:
    def test_shocks_are_additive(self, engine):
        stressed = engine._apply_scenario(
            BASELINE_MACRO, {"unemployment_shock": 2.0, "hpi_shock": -10.0}
        )
        assert stressed["unemployment_rate"] == pytest.approx(6.87)
        assert stressed["hpi_yoy_chg"] == pytest.approx(-4.4)

    def test_empty_scenario_is_a_no_op(self, engine):
        pd.testing.assert_series_equal(
            engine._apply_scenario(BASELINE_MACRO, {}), BASELINE_MACRO
        )

    def test_stress_increases_ecl(self, engine):
        df = pd.DataFrame(
            {
                "CREDIT_SCORE": [680, 700, 720],
                "ORIGINAL_LTV": [90.0, 80.0, 70.0],
                "ORIGINAL_UPB": [200_000.0] * 3,
            }
        )
        base = engine.compute_portfolio_ecl(df)
        stressed = engine.compute_portfolio_ecl(df, {"unemployment_shock": 4.0})
        assert stressed["total_ecl"] > base["total_ecl"]
