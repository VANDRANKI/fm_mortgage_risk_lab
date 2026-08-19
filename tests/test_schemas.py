"""
Unit tests for src/api/schemas.py.

LoanPredictRequest and ScenarioRequest both expose unemployment_shock,
hpi_shock and rate_shock as macro overrides for the same underlying model,
so their validation bounds are expected to match.
"""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.schemas import LoanPredictRequest, ScenarioRequest  # noqa: E402


class TestLoanPredictShockBounds:
    """Regression: these three fields had no Field() bounds at all, so
    /loan/predict accepted values like unemployment_shock=1_000_000 that
    /scenario/run already rejects for the identical physical quantity. The
    result was not an error: the heuristic PD saturated near 1.0 and the
    heuristic LGD clipped to its ceiling instead of surfacing a 422."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("unemployment_shock", 1_000_000),
            ("unemployment_shock", -1_000_000),
            ("hpi_shock", 1_000_000),
            ("hpi_shock", -1_000_000),
            ("rate_shock", 1_000_000),
            ("rate_shock", -1_000_000),
        ],
    )
    def test_extreme_shock_is_rejected(self, field, value):
        with pytest.raises(ValidationError):
            LoanPredictRequest(**{field: value})

    def test_typical_shocks_are_accepted(self):
        req = LoanPredictRequest(unemployment_shock=2.0, hpi_shock=-10.0, rate_shock=1.0)
        assert req.unemployment_shock == 2.0
        assert req.hpi_shock == -10.0
        assert req.rate_shock == 1.0

    def test_bounds_match_scenario_request(self):
        """The two schemas stress the same quantities and should reject the
        same values, so their bounds are pinned equal here rather than left
        to drift independently."""
        for field in ("unemployment_shock", "hpi_shock", "rate_shock"):
            scenario_field = ScenarioRequest.model_fields[field]
            loan_field = LoanPredictRequest.model_fields[field]
            assert scenario_field.metadata == loan_field.metadata, field


class TestLoanPredictOtherBounds:
    def test_credit_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            LoanPredictRequest(credit_score=200)
        with pytest.raises(ValidationError):
            LoanPredictRequest(credit_score=900)

    def test_defaults_construct_a_valid_request(self):
        req = LoanPredictRequest()
        assert req.credit_score == 700
        assert req.original_upb == 200000
