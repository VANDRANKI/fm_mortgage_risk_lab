"""
Tests for the dataset builders in src/features (build_pd_dataset, add_macro).

Both read parquet files from the paths in src.config.settings; the tests point
those module-level paths at a temp directory and write tiny synthetic inputs.
The look-ahead window of the PD label is the part most worth pinning down:
an off-by-one there silently leaks the future into the observation snapshot or
drops real defaults from the label.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import add_macro, build_pd_dataset  # noqa: E402

MACRO = pd.DataFrame(
    {
        "date": pd.to_datetime(["2015-01-31", "2015-02-28", "2015-03-31"]),
        "unemployment_rate": [5.0, 5.1, 5.2],
        "hpi_us": [200.0, 201.0, 202.0],
        "hpi_yoy_chg": [4.0, 4.1, 4.2],
        "mortgage_rate_30y": [3.5, 3.6, 3.7],
        "financial_stress_idx": [-0.5, -0.4, -0.3],
        "unrate_chg_mom": [0.0, 0.1, 0.1],
    }
)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    processed = tmp_path / "processed"
    macro_dir = tmp_path / "macro"
    processed.mkdir()
    macro_dir.mkdir()
    for module in (add_macro, build_pd_dataset):
        monkeypatch.setattr(module, "DATA_PROCESSED", processed)
        monkeypatch.setattr(module, "DATA_MACRO", macro_dir)
    MACRO.to_parquet(macro_dir / "fred_macro.parquet", index=False)
    return processed


def _panel_rows(loan, ages, defaulted_ages=(), zero_ages=(), start="2015-01-15"):
    start = pd.Timestamp(start)
    rows = []
    for age in ages:
        rows.append(
            {
                "LOAN_SEQUENCE_NUMBER": loan,
                "LOAN_AGE_MONTHS": age,
                "REPORTING_DATE": start + pd.DateOffset(months=age),
                "IS_DEFAULTED": age in defaulted_ages,
                "IS_ZERO_BALANCE": age in zero_ages,
                "VINTAGE_YEAR": 2014,
                "CREDIT_SCORE": 720,
                "ORIGINAL_INTEREST_RATE": 4.0,
                "CURRENT_ACTUAL_UPB": 100000.0,
            }
        )
    return rows


def _write_panel(processed, rows):
    pd.DataFrame(rows).to_parquet(processed / "loan_monthly_panel.parquet", index=False)


class TestPdLabelWindow:
    def test_default_inside_the_horizon_is_labelled_one(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 30), defaulted_ages={20}))
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert out["PD_TARGET"].tolist() == [1]

    def test_default_at_the_horizon_edge_counts(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 30), defaulted_ages={24}))
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert out["PD_TARGET"].tolist() == [1]

    def test_default_after_the_horizon_is_ignored(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 30), defaulted_ages={25}))
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert out["PD_TARGET"].tolist() == [0]

    def test_a_longer_horizon_picks_up_the_later_default(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 40), defaulted_ages={30}))
        short = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        long = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=24)
        assert short["PD_TARGET"].tolist() == [0]
        assert long["PD_TARGET"].tolist() == [1]

    def test_default_before_or_at_the_snapshot_is_not_in_the_window(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 30), defaulted_ages={5, 12}))
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert out["PD_TARGET"].tolist() == [0]

    def test_loan_with_no_future_rows_gets_label_zero(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 13)))
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert out["PD_TARGET"].tolist() == [0]
        assert out["PD_TARGET"].dtype.kind == "i"


class TestPdSnapshot:
    def test_one_row_per_loan_and_prepaid_loans_are_excluded(self, dirs):
        rows = (
            _panel_rows("A", range(0, 30))
            + _panel_rows("B", range(0, 30), defaulted_ages={15})
            + _panel_rows("PREPAID", range(0, 13), zero_ages={12})
            + _panel_rows("YOUNG", range(0, 6))
        )
        _write_panel(dirs, rows)
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        by_loan = out.set_index("LOAN_SEQUENCE_NUMBER")["PD_TARGET"].to_dict()
        assert by_loan == {"A": 0, "B": 1}

    def test_duplicate_snapshot_rows_collapse_to_one(self, dirs):
        rows = _panel_rows("A", range(0, 30)) + _panel_rows("A", [12])
        _write_panel(dirs, rows)
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert len(out) == 1

    def test_macro_is_joined_on_the_observation_month_and_spread_computed(self, dirs):
        # Origination in 2014-12, so age 1 is reported in 2015-01 (macro rate 3.5).
        rows = _panel_rows("A", range(0, 20), start="2014-12-15")
        _write_panel(dirs, rows)
        out = build_pd_dataset._build_pd_dataset(observation_age=1, horizon=12)
        row = out.iloc[0]
        assert row["mortgage_rate_30y"] == 3.5
        assert row["RATE_SPREAD"] == pytest.approx(4.0 - 3.5)
        assert row["LOAN_AGE_AT_OBS"] == 1

    def test_months_without_macro_data_give_nan_not_a_dropped_loan(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 30), start="2020-01-15"))
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert len(out) == 1
        assert np.isnan(out.iloc[0]["unemployment_rate"])

    def test_output_has_no_helper_columns(self, dirs):
        _write_panel(dirs, _panel_rows("A", range(0, 30)))
        out = build_pd_dataset._build_pd_dataset(observation_age=12, horizon=12)
        assert "obs_month" not in out.columns
        assert "IS_ZERO_BALANCE" not in out.columns


class TestAddMacroToPanel:
    def test_macro_is_attached_by_reporting_month(self, dirs):
        panel = pd.DataFrame(
            {
                "LOAN_SEQUENCE_NUMBER": ["A", "A", "A"],
                "REPORTING_DATE": pd.to_datetime(["2015-01-15", "2015-02-01", "2015-03-31"]),
            }
        )
        panel.to_parquet(dirs / "loan_monthly_panel.parquet", index=False)
        add_macro.add_macro_to_panel()
        out = pd.read_parquet(dirs / "loan_monthly_panel_with_macro.parquet")
        assert out["unemployment_rate"].tolist() == [5.0, 5.1, 5.2]
        assert "report_month" not in out.columns
        assert len(out) == 3

    def test_unmatched_month_keeps_the_row_with_nan_macro(self, dirs):
        panel = pd.DataFrame(
            {
                "LOAN_SEQUENCE_NUMBER": ["A", "A"],
                "REPORTING_DATE": pd.to_datetime(["2015-01-15", "2016-06-15"]),
            }
        )
        panel.to_parquet(dirs / "loan_monthly_panel.parquet", index=False)
        add_macro.add_macro_to_panel()
        out = pd.read_parquet(dirs / "loan_monthly_panel_with_macro.parquet")
        assert len(out) == 2
        assert out["unemployment_rate"].isna().tolist() == [False, True]


class TestAddMacroToOutcomes:
    def test_origination_macro_and_rate_spread(self, dirs):
        outcomes = pd.DataFrame(
            {
                "LOAN_SEQUENCE_NUMBER": ["A", "B"],
                "ORIGINATION_DATE": pd.to_datetime(["2015-02-10", "2015-03-01"]),
                "ORIGINAL_INTEREST_RATE": [4.0, 3.5],
            }
        )
        outcomes.to_parquet(dirs / "loan_outcomes.parquet", index=False)
        add_macro.add_macro_to_outcomes()
        out = pd.read_parquet(dirs / "loan_outcomes_with_macro.parquet")
        assert out["rate_at_orig"].tolist() == [3.6, 3.7]
        assert out["unrate_at_orig"].tolist() == [5.1, 5.2]
        assert out["RATE_SPREAD"].tolist() == pytest.approx([0.4, -0.2])
        assert "orig_month" not in out.columns
        assert "unrate_chg_mom" not in out.columns
