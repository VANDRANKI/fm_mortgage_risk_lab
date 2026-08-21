"""
Tests for src/models/train_pd.py's calibrated-pipeline assembly.

train_pd_models() itself reads a real Freddie Mac dataset from disk and is not
tested directly here. build_calibrated_pipeline is the piece that had the bug
and is exercised in isolation with synthetic data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.train_pd import build_calibrated_pipeline  # noqa: E402


def _preprocessor():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]),
         ["CREDIT_SCORE"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="UNK")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         ["PROPERTY_STATE"]),
    ])


@pytest.fixture
def split_with_a_category_missing_from_validation():
    """Training spans NY/CA/TX; the validation slice (one vintage year, as in
    the real pipeline) only has NY/CA. This is the exact shape that used to
    crash: a ColumnTransformer refit on the validation split alone sees fewer
    categories than one fit on the full training split."""
    rng = np.random.default_rng(0)

    n_train = 500
    X_train = pd.DataFrame({
        "CREDIT_SCORE": rng.normal(700, 40, n_train),
        "PROPERTY_STATE": rng.choice(["NY", "CA", "TX"], n_train),
    })
    y_train = (rng.random(n_train) < 0.3).astype(int)

    n_val = 80
    X_val = pd.DataFrame({
        "CREDIT_SCORE": rng.normal(700, 40, n_val),
        "PROPERTY_STATE": rng.choice(["NY", "CA"], n_val),
    })
    y_val = (rng.random(n_val) < 0.3).astype(int)

    pre = _preprocessor()
    pre_fitted = Pipeline([("pre", pre)])
    pre_fitted.fit(X_train)
    X_train_t = pre_fitted.named_steps["pre"].transform(X_train)
    X_val_t = pre_fitted.named_steps["pre"].transform(X_val)

    clf = LogisticRegression().fit(X_train_t, y_train)

    return {
        "pre_fitted": pre_fitted,
        "X_val": X_val,
        "X_val_t": X_val_t,
        "y_val": y_val,
        "clf": clf,
    }


class TestBuildCalibratedPipeline:
    """Regression: the calibrated pipeline used to be assembled by calling
    Pipeline.fit(X_val, y_val) on a Pipeline whose first step was the
    train-fitted preprocessor. Pipeline.fit() refits every non-final step, so
    that call silently refit the preprocessor on X_val instead of reusing the
    fit from X_train. With a validation split covering fewer categories than
    training (a single vintage year versus a multi-year training window),
    that produced a narrower one-hot output than the classifier expects and
    crashed with a feature-count mismatch, or, had the width happened to
    match, would have deployed a preprocessor that silently drops any
    training-only category at serving time."""

    def test_no_crash_when_validation_is_missing_a_category(
        self, split_with_a_category_missing_from_validation
    ):
        d = split_with_a_category_missing_from_validation
        pipe = build_calibrated_pipeline(
            d["pre_fitted"].named_steps["pre"], d["X_val_t"], d["y_val"], d["clf"]
        )
        # Must not raise, and must work on the raw (untransformed) validation
        # frame exactly as _evaluate() calls it in train_pd.py.
        prob = pipe.predict_proba(d["X_val"])[:, 1]
        assert prob.shape == (len(d["X_val"]),)

    def test_deployed_preprocessor_keeps_the_full_training_vocabulary(
        self, split_with_a_category_missing_from_validation
    ):
        d = split_with_a_category_missing_from_validation
        pipe = build_calibrated_pipeline(
            d["pre_fitted"].named_steps["pre"], d["X_val_t"], d["y_val"], d["clf"]
        )
        ohe = pipe.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
        assert set(ohe.categories_[0]) == {"CA", "NY", "TX"}

    def test_a_category_absent_from_validation_is_still_scored_correctly(
        self, split_with_a_category_missing_from_validation
    ):
        """A loan in a state (TX) that appeared in training but not in the
        validation split must still be encoded, not silently zeroed out."""
        d = split_with_a_category_missing_from_validation
        pipe = build_calibrated_pipeline(
            d["pre_fitted"].named_steps["pre"], d["X_val_t"], d["y_val"], d["clf"]
        )
        tx_loan = pd.DataFrame({"CREDIT_SCORE": [710.0], "PROPERTY_STATE": ["TX"]})
        encoded = pipe.named_steps["pre"].transform(tx_loan)
        cat_columns = encoded[:, 1:]  # first column is the scaled numeric feature
        assert cat_columns.sum() == 1.0, "the TX category must be one-hot encoded, not dropped"

        # Must also run predict_proba end to end without raising.
        prob = pipe.predict_proba(tx_loan)[:, 1]
        assert prob.shape == (1,)

    def test_returned_pipeline_is_not_refit_by_the_caller(
        self, split_with_a_category_missing_from_validation
    ):
        """The whole point of the fix: the caller must never call .fit() on
        the assembled pipeline. This test documents that build_calibrated_
        pipeline itself never does so either, by checking the preprocessor
        object identity is preserved rather than replaced by a new fit."""
        d = split_with_a_category_missing_from_validation
        original_pre = d["pre_fitted"].named_steps["pre"]
        pipe = build_calibrated_pipeline(original_pre, d["X_val_t"], d["y_val"], d["clf"])
        assert pipe.named_steps["pre"] is original_pre
