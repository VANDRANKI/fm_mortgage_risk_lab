"""
Tests for src/features/build_lgd_dataset.py.

The builder reads loan_outcomes / loan_monthly_panel / fred_macro from the
paths in src.config.settings; the tests point those at a temp directory and
use a few hand-made loans so every branch (observed LGD, synthetic LGD, the
default-timing fallback, the final quality filter) has a known answer.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import build_lgd_dataset  # noqa: E402

MACRO = pd.DataFrame(
    {
        "date": pd.date_range("2014-01-01", periods=24, freq="MS"),
        "unemployment_rate": [6.0 + 0.1 * i for i in range(24)],
        "hpi_yoy_chg": [3.0] * 24,
        "mortgage_rate_30y": [4.0 + 0.05 * i for i in range(24)],
        "financial_stress_idx": [0.0] * 24,
    }
)


def _outcome(loan, **overrides):
    row = {
        "LOAN_SEQUENCE_NUMBER": loan,
        "VINTAGE_YEAR": 2014,
        "defaulted": True,
        "liquidated": True,
        "lgd_observed": 0.4,
        "ORIGINATION_DATE": pd.Timestamp("2014-01-15"),
        "time_to_default_months": 6.0,
        "observed_months": 10.0,
        "CREDIT_SCORE": 700,
        "ORIGINAL_LTV": 80.0,
        "ORIGINAL_CLTV": 80.0,
        "ORIGINAL_DTI": 35.0,
        "ORIGINAL_UPB": 200000.0,
        "ORIGINAL_INTEREST_RATE": 4.5,
        "ORIGINAL_LOAN_TERM": 360,
        "OCC_STATUS": "P",
        "PROPERTY_TYPE": "SF",
        "PROPERTY_STATE": "CA",
        "LOAN_PURPOSE": "C",
        "MI_PCT": 0.0,
        "EAD": 150000.0,
    }
    row.update(overrides)
    return row


def _build(tmp_path, monkeypatch, outcomes):
    processed = tmp_path / "processed"
    macro_dir = tmp_path / "macro"
    processed.mkdir()
    macro_dir.mkdir()
    monkeypatch.setattr(build_lgd_dataset, "DATA_PROCESSED", processed)
    monkeypatch.setattr(build_lgd_dataset, "DATA_MACRO", macro_dir)
    pd.DataFrame(outcomes).to_parquet(processed / "loan_outcomes.parquet", index=False)
    pd.DataFrame({"LOAN_SEQUENCE_NUMBER": ["x"]}).to_parquet(
        processed / "loan_monthly_panel.parquet", index=False
    )
    MACRO.to_parquet(macro_dir / "fred_macro.parquet", index=False)
    build_lgd_dataset.build_lgd_dataset()
    return pd.read_parquet(processed / "lgd_dataset.parquet").set_index("LOAN_SEQUENCE_NUMBER")


def test_observed_lgd_is_kept_as_is(tmp_path, monkeypatch):
    out = _build(tmp_path, monkeypatch, [_outcome("A", lgd_observed=0.37)])
    assert out.loc["A", "lgd_observed"] == pytest.approx(0.37)


def test_loans_that_never_defaulted_are_excluded(tmp_path, monkeypatch):
    out = _build(
        tmp_path,
        monkeypatch,
        [_outcome("A"), _outcome("PERFORMING", defaulted=False, liquidated=False, lgd_observed=None)],
    )
    assert list(out.index) == ["A"]


@pytest.mark.parametrize(
    "ltv, expected",
    [
        (20.0, 0.30),  # low LTV never goes below the 0.30 base
        (80.0, 0.30),
        (100.0, 0.50),
        (110.0, 0.60),
        (200.0, 0.90),  # capped
    ],
)
def test_synthetic_lgd_follows_the_ltv_formula(tmp_path, monkeypatch, ltv, expected):
    loan = _outcome("S", liquidated=False, lgd_observed=None, ORIGINAL_LTV=ltv)
    out = _build(tmp_path, monkeypatch, [loan])
    assert out.loc["S", "lgd_observed"] == pytest.approx(expected)


def test_missing_ltv_uses_the_80_percent_default(tmp_path, monkeypatch):
    loan = _outcome("S", liquidated=False, lgd_observed=None, ORIGINAL_LTV=None)
    out = _build(tmp_path, monkeypatch, [loan])
    assert out.loc["S", "lgd_observed"] == pytest.approx(0.30)


def test_liquidated_loan_with_missing_loss_gets_a_synthetic_value(tmp_path, monkeypatch):
    out = _build(tmp_path, monkeypatch, [_outcome("L", liquidated=True, lgd_observed=None)])
    assert out.loc["L", "lgd_observed"] == pytest.approx(0.30)


def test_out_of_range_observed_lgd_is_dropped(tmp_path, monkeypatch):
    out = _build(
        tmp_path,
        monkeypatch,
        [_outcome("OK"), _outcome("GAIN", lgd_observed=-0.1), _outcome("OVER", lgd_observed=1.2)],
    )
    assert list(out.index) == ["OK"]


def test_missing_ead_falls_back_to_original_upb_and_zero_ead_is_dropped(tmp_path, monkeypatch):
    out = _build(
        tmp_path,
        monkeypatch,
        [
            _outcome("NOEAD", EAD=None, ORIGINAL_UPB=123000.0),
            _outcome("ZERO", EAD=0.0),
            _outcome("OK"),
        ],
    )
    assert out.loc["NOEAD", "EAD"] == 123000.0
    assert "ZERO" not in out.index
    assert "OK" in out.index


def test_macro_is_taken_from_the_default_month(tmp_path, monkeypatch):
    # Origination 2014-01-15 + 6 months (182.64 days) lands in July 2014 (index 6).
    out = _build(tmp_path, monkeypatch, [_outcome("A", time_to_default_months=6.0)])
    assert out.loc["A", "unemployment_rate_at_default"] == pytest.approx(6.0 + 0.1 * 6)
    assert out.loc["A", "rate_at_default"] == pytest.approx(4.0 + 0.05 * 6)
    assert out.loc["A", "LOAN_AGE_AT_DEFAULT"] == 6.0


def test_unknown_default_timing_falls_back_to_observed_months(tmp_path, monkeypatch):
    """Regression guard: macro-at-default and LOAN_AGE_AT_DEFAULT must use the
    same fallback (observed_months), not origination month for one of them."""
    loan = _outcome(
        "F", liquidated=False, lgd_observed=None, time_to_default_months=None, observed_months=10.0
    )
    out = _build(tmp_path, monkeypatch, [loan])
    assert out.loc["F", "LOAN_AGE_AT_DEFAULT"] == 10.0
    assert out.loc["F", "unemployment_rate_at_default"] == pytest.approx(6.0 + 0.1 * 10)


def test_output_columns_are_only_the_model_features(tmp_path, monkeypatch):
    out = _build(tmp_path, monkeypatch, [_outcome("A")])
    for helper in ("default_date", "default_time", "lgd_source", "liquidated", "defaulted"):
        assert helper not in out.columns
    assert {"lgd_observed", "LOAN_AGE_AT_DEFAULT", "fsi_at_default", "EAD"} <= set(out.columns)
