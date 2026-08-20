"""
Unit tests for the delinquency classification in src/ingest/load_svcg.py.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.load_svcg import _is_seriously_delinquent  # noqa: E402


class TestSeriouslyDelinquentThreshold:
    """Regression: CURRENT_LOAN_DELINQUENCY_STATUS is an unpadded count of
    months delinquent ('0'=current, '1'=30dpd, '2'=60dpd, '3'=90dpd, and it
    keeps counting past single digits for a loan still delinquent 300+ days).
    The old check was `dlq.isin({"3",...,"9","XX","RA"})`, a fixed string set
    that can only ever match single characters, so "10", "11", "24" and any
    other double-digit month count silently fell outside it even though they
    describe loans further past due than the "3" already included."""

    @pytest.mark.parametrize("value", ["3", "4", "9"])
    def test_single_digit_threshold_and_above_flagged(self, value):
        assert _is_seriously_delinquent(pd.Series([value])).iloc[0]

    @pytest.mark.parametrize("value", ["0", "1", "2"])
    def test_below_threshold_not_flagged(self, value):
        assert not _is_seriously_delinquent(pd.Series([value])).iloc[0]

    @pytest.mark.parametrize("value", ["10", "11", "24", "99"])
    def test_double_digit_months_are_flagged(self, value):
        assert _is_seriously_delinquent(pd.Series([value])).iloc[0]

    @pytest.mark.parametrize("value", ["XX", "RA"])
    def test_non_numeric_status_codes_are_flagged(self, value):
        assert _is_seriously_delinquent(pd.Series([value])).iloc[0]

    def test_empty_string_is_not_flagged(self):
        """fillna("") is applied by the caller before this runs, so an empty
        string (originally missing) must not parse as a delinquency month
        count or be caught by the non-numeric code set."""
        assert not _is_seriously_delinquent(pd.Series([""])).iloc[0]

    def test_mixed_batch_matches_expected_pattern(self):
        dlq = pd.Series(["0", "1", "2", "3", "9", "10", "11", "24", "XX", "RA", ""])
        expected = [False, False, False, True, True, True, True, True, True, True, False]
        assert _is_seriously_delinquent(dlq).tolist() == expected
