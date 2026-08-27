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

from src.ingest.load_orig import (  # noqa: E402
    SENTINEL_MAP,
    _clean_string_columns,
    _replace_numeric_sentinels,
)


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


class TestReplaceNumericSentinels:
    """Regression: ORIGINAL_LTV and ORIGINAL_CLTV use the same 999 "not
    available" sentinel as ORIGINAL_DTI in the Freddie Mac layout, but
    SENTINEL_MAP only listed CREDIT_SCORE and ORIGINAL_DTI. A loan with a
    genuinely missing LTV/CLTV kept the literal value 999 (i.e. 999% LTV)
    instead of becoming NaN, so it was fed straight into the PD/LGD models
    as a wild numeric outlier instead of being median-imputed like every
    other missing numeric feature."""

    def test_ltv_and_cltv_sentinels_become_missing(self):
        df = pd.DataFrame({
            "ORIGINAL_LTV":  [80.0, 999.0],
            "ORIGINAL_CLTV": [80.0, 999.0],
            "ORIGINAL_DTI":  [30.0, 999.0],
            "CREDIT_SCORE":  [720.0, 9999.0],
        })
        sentinel_map = {
            "CREDIT_SCORE":  9999,
            "ORIGINAL_DTI":  999,
            "ORIGINAL_LTV":  999,
            "ORIGINAL_CLTV": 999,
        }
        out = _replace_numeric_sentinels(df, sentinel_map)

        assert out["ORIGINAL_LTV"].iloc[0] == 80.0
        assert pd.isna(out["ORIGINAL_LTV"].iloc[1])
        assert out["ORIGINAL_CLTV"].iloc[0] == 80.0
        assert pd.isna(out["ORIGINAL_CLTV"].iloc[1])
        assert pd.isna(out["ORIGINAL_DTI"].iloc[1])
        assert pd.isna(out["CREDIT_SCORE"].iloc[1])

    def test_old_sentinel_map_would_have_missed_ltv_cltv(self):
        """Documents the exact bug: the map that shipped had no entry for
        ORIGINAL_LTV or ORIGINAL_CLTV, so 999 passed straight through."""
        df = pd.DataFrame({"ORIGINAL_LTV": [999.0], "ORIGINAL_CLTV": [999.0]})
        old_sentinel_map = {"CREDIT_SCORE": 9999, "ORIGINAL_DTI": 999}  # the map that shipped

        out = _replace_numeric_sentinels(df.copy(), old_sentinel_map)

        assert out["ORIGINAL_LTV"].iloc[0] == 999.0   # bug: not converted to missing
        assert out["ORIGINAL_CLTV"].iloc[0] == 999.0  # bug: not converted to missing

    def test_missing_column_is_skipped_without_error(self):
        df = pd.DataFrame({"ORIGINAL_LTV": [80.0]})
        out = _replace_numeric_sentinels(df, {"ORIGINAL_LTV": 999, "NOT_PRESENT": 1})
        assert out["ORIGINAL_LTV"].tolist() == [80.0]

    def test_production_sentinel_map_covers_ltv_and_cltv(self):
        """Guards against the actual regression: the module-level SENTINEL_MAP
        used by load_orig_year must list ORIGINAL_LTV and ORIGINAL_CLTV, not
        just CREDIT_SCORE and ORIGINAL_DTI."""
        assert SENTINEL_MAP.get("ORIGINAL_LTV") == 999
        assert SENTINEL_MAP.get("ORIGINAL_CLTV") == 999
