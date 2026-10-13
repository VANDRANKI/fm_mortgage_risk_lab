"""
Tests for src/models/scenarios.py.

The ECL engine is built without model artifacts (heuristic PD and LGD) and the
output directory is redirected to a temp path, so no trained models or Freddie
Mac data are needed.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.settings import SCENARIOS  # noqa: E402
from src.models import scenarios  # noqa: E402
from src.models.ecl_engine import ECLEngine  # noqa: E402

BASELINE_MACRO = pd.Series(
    {
        "unemployment_rate": 4.87,
        "hpi_yoy_chg": 5.6,
        "mortgage_rate_30y": 3.65,
        "financial_stress_idx": -0.5,
        "unrate_chg_mom": 0.0,
    }
)


def _heuristic_engine() -> ECLEngine:
    e = ECLEngine.__new__(ECLEngine)
    e.dataset_name = "pd_12m"
    e.pd_model = None
    e.lgd_model = None
    e.lgd_pre = None
    e._macro_baseline = BASELINE_MACRO.copy()
    return e


def _loans(n=4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "LOAN_SEQUENCE_NUMBER": [f"L{i}" for i in range(n)],
            "CURRENT_ACTUAL_UPB": [200_000.0] * n,
            "ORIGINAL_UPB": [250_000.0] * n,
            "ORIGINAL_INTEREST_RATE": [4.0] * n,
            "CREDIT_SCORE": [720] * n,
            "ORIGINAL_LTV": [80] * n,
            "LOAN_AGE_MONTHS": [24] * n,
            "DLQ_STATUS_INT": [0] * n,
        }
    )


def _panel_rows(loan, vintage, dates, defaulted):
    return [
        {
            "LOAN_SEQUENCE_NUMBER": loan,
            "VINTAGE_YEAR": vintage,
            "REPORTING_DATE": pd.Timestamp(d),
            "IS_DEFAULTED": flag,
        }
        for d, flag in zip(dates, defaulted)
    ]


class TestRunScenario:
    def test_run_scenario_returns_portfolio_metrics(self):
        out = scenarios.run_scenario(_loans(), SCENARIOS["baseline"], _heuristic_engine())
        assert out["loan_count"] == 4
        assert out["total_ead"] == pytest.approx(4 * 200_000.0)
        assert out["total_ecl"] > 0

    def test_run_all_scenarios_tags_every_predefined_scenario(self):
        results = scenarios.run_all_scenarios(_loans(), _heuristic_engine())
        assert set(results) == set(SCENARIOS)
        for name, result in results.items():
            assert result["scenario_name"] == name
            assert result["scenario_label"] == SCENARIOS[name]["label"]
            assert result["color"] == SCENARIOS[name]["color"]

    def test_severe_scenario_costs_more_than_baseline(self):
        results = scenarios.run_all_scenarios(_loans(), _heuristic_engine())
        assert results["severe"]["total_ecl"] > results["baseline"]["total_ecl"]


class TestVintageCurves:
    @pytest.fixture
    def out_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(scenarios, "PORTFOLIO_BASELINE", tmp_path)
        return tmp_path

    def test_loan_defaulted_for_many_months_counts_once(self, out_dir):
        # One loan is 90+ days delinquent for all 12 months of 2020. The other
        # never defaults. Summing the monthly flag would give 12 / 2 = 600 percent.
        months = pd.date_range("2020-01-15", periods=12, freq="MS") + pd.Timedelta(days=14)
        rows = _panel_rows("L1", 2018, months, [True] * 12)
        rows += _panel_rows("L2", 2018, months, [False] * 12)
        scenarios._build_vintage_curves(pd.DataFrame(rows))

        curves = pd.read_parquet(out_dir / "vintage_curves.parquet")
        row = curves.iloc[0]
        assert row["total_loans"] == 2
        assert row["defaulted"] == 1
        assert row["default_rate"] == pytest.approx(0.5)

    def test_default_rate_never_exceeds_one(self, out_dir):
        months = pd.date_range("2021-01-15", periods=6, freq="MS")
        rows = []
        for i in range(3):
            rows += _panel_rows(f"L{i}", 2019, months, [True] * 6)
        scenarios._build_vintage_curves(pd.DataFrame(rows))

        curves = pd.read_parquet(out_dir / "vintage_curves.parquet")
        assert (curves["default_rate"] <= 1.0).all()
        assert curves.iloc[0]["default_rate"] == pytest.approx(1.0)

    def test_cell_without_defaults_has_zero_rate(self, out_dir):
        rows = _panel_rows("L1", 2017, ["2019-03-01", "2019-04-01"], [False, False])
        scenarios._build_vintage_curves(pd.DataFrame(rows))

        curves = pd.read_parquet(out_dir / "vintage_curves.parquet")
        assert curves.iloc[0]["defaulted"] == 0
        assert curves.iloc[0]["default_rate"] == 0.0

    def test_cells_are_split_by_vintage_and_observation_year(self, out_dir):
        rows = _panel_rows("L1", 2017, ["2019-12-01", "2020-01-01"], [False, True])
        rows += _panel_rows("L2", 2018, ["2020-01-01"], [False])
        scenarios._build_vintage_curves(pd.DataFrame(rows))

        curves = pd.read_parquet(out_dir / "vintage_curves.parquet")
        cell = curves.set_index(["vintage_year", "obs_year"])
        assert cell.loc[(2017, 2019), "defaulted"] == 0
        assert cell.loc[(2017, 2020), "defaulted"] == 1
        assert cell.loc[(2018, 2020), "total_loans"] == 1


class TestPrecompute:
    def test_snapshot_uses_latest_row_and_skips_zero_balance_loans(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "baseline"
        monkeypatch.setattr(scenarios, "PORTFOLIO_BASELINE", out_dir)
        monkeypatch.setattr(scenarios, "ECLEngine", _heuristic_engine)

        base = _loans(2).drop(columns="DLQ_STATUS_INT")
        old = base.assign(REPORTING_DATE=pd.Timestamp("2020-01-01"), IS_ZERO_BALANCE=False,
                          IS_DEFAULTED=False, VINTAGE_YEAR=2018, CURRENT_ACTUAL_UPB=999_999.0)
        new = base.assign(REPORTING_DATE=pd.Timestamp("2020-02-01"), IS_ZERO_BALANCE=[False, True],
                          IS_DEFAULTED=False, VINTAGE_YEAR=2018)
        path = tmp_path / "panel.parquet"
        pd.concat([old, new]).to_parquet(path, index=False)

        scenarios.precompute_and_cache_scenarios(path)

        results = json.loads((out_dir / "all_scenarios.json").read_text())
        assert set(results) == set(SCENARIOS)
        # Only L0 is still active, and its balance is the latest one, not the 999,999 row.
        assert results["baseline"]["loan_count"] == 1
        assert results["baseline"]["total_ead"] == pytest.approx(200_000.0)
        assert (out_dir / "vintage_curves.parquet").exists()
