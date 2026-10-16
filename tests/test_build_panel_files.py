"""
End-to-end tests for build_monthly_panel() and build_loan_outcomes().

The Freddie Mac raw files are not available in CI, so these tests write small
synthetic interim parquet files into a temp directory and point the module's
path constants at it. They check the per-loan outcome summary: default,
prepayment and liquidation flags, timing, EAD from the last non-zero balance,
and the LGD formula.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features import build_panel  # noqa: E402


def _svcg_row(loan, age, upb, dlq="0", zb="", mod=None, **extra):
    row = {
        "LOAN_SEQUENCE_NUMBER": loan,
        "MONTHLY_REPORTING_PERIOD": "202001",
        "REPORTING_DATE": pd.Timestamp("2020-01-01") + pd.DateOffset(months=age),
        "CURRENT_ACTUAL_UPB": upb,
        "CURRENT_LOAN_DELINQUENCY_STATUS": dlq,
        "LOAN_AGE": age,
        "REMAINING_MONTHS_TO_MATURITY": 360 - age,
        "CURRENT_INTEREST_RATE": 4.0,
        "ZERO_BALANCE_CODE": zb,
        "ZERO_BALANCE_DATE": pd.NaT,
        "IS_SERIOUSLY_DELINQUENT": dlq in {"3", "4", "5"},
        "IS_ZERO_BALANCE": zb != "",
        "IS_PREPAID": zb == "01",
        "IS_LIQUIDATED": zb in {"02", "03", "04", "05", "06", "09"},
        "IS_DEFAULTED": False,
        "MODIFICATION_FLAG": mod,
        "NET_SALES_PROCEEDS": np.nan,
        "ACTUAL_LOSS_CALCULATION": np.nan,
        "EXPENSES": np.nan,
        "MI_RECOVERIES": np.nan,
        "NON_MI_RECOVERIES": np.nan,
    }
    row.update(extra)
    return row


def _orig_row(loan):
    row = {c: np.nan for c in build_panel.ORIG_KEEP}
    row.update(
        LOAN_SEQUENCE_NUMBER=loan,
        VINTAGE_YEAR=2020,
        ORIGINATION_DATE=pd.Timestamp("2020-01-01"),
        CREDIT_SCORE=720,
        ORIGINAL_UPB=100000,
    )
    return row


@pytest.fixture
def built(tmp_path, monkeypatch):
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir()
    processed.mkdir()
    monkeypatch.setattr(build_panel, "DATA_INTERIM", interim)
    monkeypatch.setattr(build_panel, "DATA_PROCESSED", processed)

    rows = [
        # PREPAID: pays off at age 24.
        _svcg_row("PREPAID", 22, 80000),
        _svcg_row("PREPAID", 23, 79000),
        _svcg_row("PREPAID", 24, 0, zb="01"),
        # DEFAULT_LIQ: 3+ months delinquent at age 10, liquidated at age 12.
        _svcg_row("DEFAULT_LIQ", 9, 100000),
        _svcg_row("DEFAULT_LIQ", 10, 99000, dlq="3"),
        _svcg_row("DEFAULT_LIQ", 11, 98000, dlq="4"),
        _svcg_row(
            "DEFAULT_LIQ", 12, 0, dlq="5", zb="09",
            NET_SALES_PROCEEDS=60000.0, MI_RECOVERIES=5000.0,
            NON_MI_RECOVERIES=1000.0, EXPENSES=2000.0,
        ),
        # QUIET_LIQ: never seriously delinquent, liquidated at age 5, proceeds
        # above the balance so the raw loss is negative.
        _svcg_row("QUIET_LIQ", 4, 50000, dlq="1"),
        _svcg_row("QUIET_LIQ", 5, 0, dlq="1", zb="03", NET_SALES_PROCEEDS=60000.0),
        # CURRENT: performing, modified once.
        _svcg_row("CURRENT", 1, 90000),
        _svcg_row("CURRENT", 2, 89000, mod="Y"),
        _svcg_row("CURRENT", 3, 88000),
        # REPORTED_LOSS: has a pre-computed ACTUAL_LOSS_CALCULATION.
        _svcg_row("REPORTED_LOSS", 7, 40000, dlq="3"),
        _svcg_row("REPORTED_LOSS", 8, 0, dlq="3", zb="02", ACTUAL_LOSS_CALCULATION=10000.0),
    ]
    pd.DataFrame(rows).to_parquet(interim / "svcg_all.parquet", index=False)
    loans = ["PREPAID", "DEFAULT_LIQ", "QUIET_LIQ", "CURRENT", "REPORTED_LOSS"]
    pd.DataFrame([_orig_row(loan) for loan in loans]).to_parquet(
        interim / "orig_all.parquet", index=False
    )

    build_panel.build_monthly_panel()
    out = build_panel.build_loan_outcomes()
    df = pd.read_parquet(out).set_index("LOAN_SEQUENCE_NUMBER")
    return df, pd.read_parquet(processed / "loan_monthly_panel.parquet")


def test_one_outcome_row_per_loan(built):
    outcomes, _ = built
    assert sorted(outcomes.index) == [
        "CURRENT", "DEFAULT_LIQ", "PREPAID", "QUIET_LIQ", "REPORTED_LOSS"
    ]


def test_monthly_panel_adds_age_and_delinquency_int(built):
    _, panel = built
    assert "LOAN_AGE_MONTHS" in panel.columns
    row = panel[(panel["LOAN_SEQUENCE_NUMBER"] == "DEFAULT_LIQ") & (panel["LOAN_AGE"] == 11)]
    assert int(row["DLQ_STATUS_INT"].iloc[0]) == 4


def test_prepaid_loan(built):
    outcomes, _ = built
    row = outcomes.loc["PREPAID"]
    assert row["prepaid"]
    assert not row["defaulted"]
    assert not row["liquidated"]
    assert row["time_to_prepay_months"] == 24


def test_defaulted_loan_timing_is_first_serious_delinquency(built):
    outcomes, _ = built
    row = outcomes.loc["DEFAULT_LIQ"]
    assert row["defaulted"]
    assert row["liquidated"]
    assert row["time_to_default_months"] == 10
    assert row["zero_balance_age_months"] == 12
    assert row["observed_months"] == 12
    assert row["max_dlq_status"] == 5


def test_ead_is_last_nonzero_balance_and_lgd_follows_the_formula(built):
    outcomes, _ = built
    row = outcomes.loc["DEFAULT_LIQ"]
    # The final row has a zero balance, so EAD must come from age 11.
    assert row["EAD"] == 98000
    # 98000 - 60000 - 5000 - 1000 + 2000 = 34000
    assert row["lgd_observed"] == pytest.approx(34000 / 98000)


def test_liquidated_without_serious_delinquency_still_counts_as_default(built):
    outcomes, _ = built
    row = outcomes.loc["QUIET_LIQ"]
    assert row["defaulted"]
    assert row["liquidated"]
    # No serious-delinquency month, so timing falls back to the zero-balance age.
    assert row["time_to_default_months"] == 5


def test_negative_loss_is_clipped_to_zero_lgd(built):
    outcomes, _ = built
    assert outcomes.loc["QUIET_LIQ", "EAD"] == 50000
    assert outcomes.loc["QUIET_LIQ", "lgd_observed"] == 0.0


def test_precomputed_actual_loss_is_preferred(built):
    outcomes, _ = built
    # EAD 40000, ACTUAL_LOSS_CALCULATION 10000 -> 0.25, ignoring the components.
    assert outcomes.loc["REPORTED_LOSS", "lgd_observed"] == pytest.approx(0.25)
    # QUIET_LIQ has no reported loss and falls back to the formula.
    assert outcomes.loc["QUIET_LIQ", "lgd_observed"] == 0.0


def test_performing_loan_has_no_event_fields(built):
    outcomes, _ = built
    row = outcomes.loc["CURRENT"]
    assert not row["defaulted"]
    assert not row["prepaid"]
    assert not row["liquidated"]
    assert pd.isna(row["time_to_default_months"])
    assert pd.isna(row["lgd_observed"])
    assert row["observed_months"] == 3


def test_ever_modified_flag(built):
    outcomes, _ = built
    assert bool(outcomes.loc["CURRENT", "ever_modified"]) is True
    assert bool(outcomes.loc["PREPAID", "ever_modified"]) is False
