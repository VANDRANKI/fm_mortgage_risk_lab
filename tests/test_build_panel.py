"""
Unit tests for src/features/build_panel.py's default-flag computation.

build_loan_outcomes() itself reads/writes real Parquet files and is not
tested directly here. The two pieces that had the bug -- _build_default_flags
and _fill_missing_default_timing -- are pure and are exercised in isolation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.build_panel import (  # noqa: E402
    _build_default_flags,
    _fill_missing_default_timing,
)


class TestBuildDefaultFlags:
    """Regression: `defaulted` was computed only from loans that were ever
    IS_SERIOUSLY_DELINQUENT (`set(first_default.index)`). A loan liquidated
    via a short sale / deed-in-lieu / third-party sale zero-balance code
    without ever accumulating 3+ months of *reported* delinquency (a
    lender-agreed hardship resolution, or a reporting lag on its final month)
    ended up with liquidated=True but defaulted=False. build_lgd_dataset.py
    requires defaulted=True before it will admit a loan's observed LGD into
    the training set, so such loans -- with a fully computed, real loss --
    were silently dropped instead of trained on."""

    def test_liquidated_loan_never_seriously_delinquent_is_now_defaulted(self):
        loan_ids = pd.Series(["L1", "L2", "L3"])
        first_default_index = pd.Index(["L1"])       # only L1 was ever 3+ months dpd
        liquidated_index = pd.Index(["L1", "L2"])     # L1 and L2 were both liquidated

        flags = _build_default_flags(loan_ids, first_default_index, liquidated_index)

        assert flags.tolist() == [True, True, False]

    def test_old_buggy_definition_would_have_missed_it(self):
        """Documents the exact bug: set(first_default.index) alone excludes
        a liquidated-only loan."""
        first_default_index = pd.Index(["L1"])
        old_defaulted_set = set(first_default_index)  # the expression that shipped
        assert "L2" not in old_defaulted_set

    def test_loan_with_neither_signal_is_not_defaulted(self):
        loan_ids = pd.Series(["L4"])
        flags = _build_default_flags(loan_ids, pd.Index(["L1"]), pd.Index(["L2"]))
        assert flags.tolist() == [False]

    def test_loan_flagged_by_either_signal_alone(self):
        loan_ids = pd.Series(["A", "B"])
        # A only seriously delinquent, B only liquidated.
        flags = _build_default_flags(loan_ids, pd.Index(["A"]), pd.Index(["B"]))
        assert flags.tolist() == [True, True]


class TestFillMissingDefaultTiming:
    """Regression: a loan that defaulted only via liquidation has no
    time_to_default_months from the IS_SERIOUSLY_DELINQUENT-based join (it's
    NaN), which build_lgd_dataset.py needs for LOAN_AGE_AT_DEFAULT and the
    macro-at-default lookup. Falls back to the loan's zero-balance age --
    only for liquidated loans, since non-liquidated loans have no
    zero-balance event to fall back to."""

    def test_missing_timing_falls_back_to_zero_balance_age_for_liquidated_loan(self):
        time_to_default = pd.Series([np.nan])
        zero_balance_age = pd.Series([7])
        liquidated = pd.Series([True])

        out = _fill_missing_default_timing(time_to_default, zero_balance_age, liquidated)

        assert out.iloc[0] == 7

    def test_existing_timing_is_not_overwritten(self):
        time_to_default = pd.Series([3.0])
        zero_balance_age = pd.Series([9])
        liquidated = pd.Series([True])

        out = _fill_missing_default_timing(time_to_default, zero_balance_age, liquidated)

        assert out.iloc[0] == 3.0

    def test_non_liquidated_loan_with_missing_timing_stays_missing(self):
        """A loan that's neither seriously delinquent nor liquidated has no
        default event at all; there's nothing to fall back to."""
        time_to_default = pd.Series([np.nan])
        zero_balance_age = pd.Series([np.nan])
        liquidated = pd.Series([False])

        out = _fill_missing_default_timing(time_to_default, zero_balance_age, liquidated)

        assert pd.isna(out.iloc[0])

    def test_mixed_batch(self):
        time_to_default = pd.Series([np.nan, 5.0, np.nan])
        zero_balance_age = pd.Series([4, 8, np.nan])
        liquidated = pd.Series([True, True, False])

        out = _fill_missing_default_timing(time_to_default, zero_balance_age, liquidated)

        assert out.iloc[0] == 4      # filled from zero-balance age
        assert out.iloc[1] == 5.0    # kept the real value, not overwritten
        assert pd.isna(out.iloc[2])  # not liquidated, nothing to fall back to
