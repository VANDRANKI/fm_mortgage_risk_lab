"""
Unit tests for src/ingest/load_orig.py's string-column cleaning.

This is split out from load_orig_year so it can be tested without a synthetic
Freddie Mac raw file: load_orig_year reads from disk and does numeric casting,
date parsing and dtype conversion in the same function.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest.load_orig import _clean_string_columns  # noqa: E402


class TestLoanPurposeCodeNine:
    """Regression: a generic sentinel rule mapped "9"/"99" to missing across
    every string column. LOAN_PURPOSE uses "9" as a real "not applicable /
    other" code in some vintage years, and a line meant to restore it after
    the fact repeated .replace("", pd.NA), which cannot recover a value that
    had already become <NA>. Every LOAN_PURPOSE of "9" was silently lost."""

    def test_loan_purpose_nine_is_kept(self):
        df = pd.DataFrame({"LOAN_PURPOSE": ["P", "C", "9", " 9 ", ""]})
        out = _clean_string_columns(df, ["LOAN_PURPOSE"])
        assert out["LOAN_PURPOSE"].tolist() == ["P", "C", "9", "9", pd.NA]

    def test_loan_purpose_blank_is_still_missing(self):
        df = pd.DataFrame({"LOAN_PURPOSE": ["", "   "]})
        out = _clean_string_columns(df, ["LOAN_PURPOSE"])
        assert out["LOAN_PURPOSE"].isna().all()

    def test_other_columns_still_treat_nine_as_missing(self):
        """Only LOAN_PURPOSE is exempt. OCC_STATUS and similar fields use "9"
        as a genuine missing sentinel and must keep that behavior."""
        df = pd.DataFrame({"OCC_STATUS": ["P", "9", "99", "S", ""]})
        out = _clean_string_columns(df, ["OCC_STATUS"])
        assert out["OCC_STATUS"].tolist()[0] == "P"
        assert out["OCC_STATUS"].tolist()[3] == "S"
        assert out["OCC_STATUS"].isna().sum() == 3

    def test_column_not_in_frame_is_skipped_without_error(self):
        df = pd.DataFrame({"LOAN_PURPOSE": ["P"]})
        out = _clean_string_columns(df, ["LOAN_PURPOSE", "NOT_PRESENT"])
        assert out["LOAN_PURPOSE"].tolist() == ["P"]

    def test_both_columns_cleaned_together_independently(self):
        df = pd.DataFrame({
            "LOAN_PURPOSE": ["9", "P"],
            "OCC_STATUS":   ["9", "S"],
        })
        out = _clean_string_columns(df, ["LOAN_PURPOSE", "OCC_STATUS"])
        assert out["LOAN_PURPOSE"].tolist() == ["9", "P"]
        assert out["OCC_STATUS"].iloc[0] is pd.NA or pd.isna(out["OCC_STATUS"].iloc[0])
        assert out["OCC_STATUS"].iloc[1] == "S"
