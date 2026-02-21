"""
Train PD (Probability of Default) models.

Models trained:
  1. Logistic Regression (baseline, interpretable)
  2. XGBoost Gradient Boosting (advanced)
  3. Random Forest (ensemble diversity)
  4. Calibrated XGBoost (for well-calibrated PD output)

Validation strategy: time-based split by vintage year
  Train:    2010-2013
  Validate: 2014
  Test:     2015-2016

Outputs (saved to models/):
  pd_logit.joblib          – fitted sklearn Pipeline
  pd_xgb.joblib            – fitted XGBoost Pipeline (calibrated)
  pd_rf.joblib             – fitted Random Forest Pipeline
  pd_preprocessor.joblib   – shared preprocessing pipeline
  pd_metrics.json          – evaluation metrics
  pd_feature_importance.parquet
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    brier_score_loss, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from src.config.settings import (
    DATA_PROCESSED, MODELS_DIR, FIGURES_DIR,
    TRAIN_YEARS, VALID_YEARS, TEST_YEARS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Feature schema ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
    "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
    "MI_PCT", "RATE_SPREAD", "LOAN_AGE_AT_OBS",
    "unemployment_rate", "hpi_yoy_chg", "mortgage_rate_30y",
    "financial_stress_idx", "unrate_chg_mom",
    "DLQ_STATUS_INT",
]

CATEGORICAL_FEATURES = [
    "OCC_STATUS", "LOAN_PURPOSE", "PROPERTY_TYPE", "PROPERTY_STATE",
    "PRODUCT_TYPE", "CHANNEL", "FIRST_TIME_HOMEBUYER_FLAG",
]

TARGET = "PD_TARGET"


def _build_preprocessor() -> ColumnTransformer:
    """Shared preprocessing pipeline: impute → scale / encode."""
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", cat_pipe,     CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic (max separation between CDFs)."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def _evaluate(name: str, model, X: pd.DataFrame, y: pd.Series) -> dict:
    prob = model.predict_proba(X)[:, 1]
    auc  = roc_auc_score(y, prob)
    ap   = average_precision_score(y, prob)
    ks   = _ks_statistic(y.values, prob)
    brier = brier_score_loss(y, prob)
    log.info(
        "  %s | AUC=%.4f | AP=%.4f | KS=%.4f | Brier=%.4f",
        name, auc, ap, ks, brier
    )
    return {"model": name, "auc": auc, "ap": ap, "ks": ks, "brier": brier}


def load_split_data(dataset_path: Path):
    """Load dataset and split into train / valid / test by vintage year."""
    df = pd.read_parquet(dataset_path)
    log.info("Loaded %d rows from %s", len(df), dataset_path)
    log.info("  Default rate: %.2f%%", df[TARGET].mean() * 100)

    avail_num = [c for c in NUMERIC_FEATURES      if c in df.columns]
    avail_cat = [c for c in CATEGORICAL_FEATURES   if c in df.columns]

    X = df[avail_num + avail_cat].copy()
    y = df[TARGET].astype(int)

    train_mask = df["VINTAGE_YEAR"].isin(TRAIN_YEARS)
    valid_mask = df["VINTAGE_YEAR"].isin(VALID_YEARS)
    test_mask  = df["VINTAGE_YEAR"].isin(TEST_YEARS)

    return (
        X[train_mask], y[train_mask],
        X[valid_mask], y[valid_mask],
        X[test_mask],  y[test_mask],
        avail_num, avail_cat,
    )


def train_pd_models(dataset_name: str = "pd_12m") -> None:
    """Train all PD models for a given dataset."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_path = DATA_PROCESSED / f"{dataset_name}_dataset.parquet"
    (X_train, y_train,
     X_val,   y_val,
     X_test,  y_test,
     avail_num, avail_cat) = load_split_data(dataset_path)

    log.info(
        "Split sizes: train=%d val=%d test=%d",
        len(X_train), len(X_val), len(X_test)
    )

    # ── Build preprocessor ───────────────────────────────────────────────────
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="UNK")),
        ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, avail_num),
            ("cat", cat_pipe,     avail_cat),
        ],
        remainder="drop",
    )

    # ── 1. Logistic Regression ───────────────────────────────────────────────
    log.info("Training Logistic Regression ...")
    logit_pipe = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            C=0.1, max_iter=1000, solver="lbfgs", class_weight="balanced",
            random_state=42,
        )),
    ])
    logit_pipe.fit(X_train, y_train)
    joblib.dump(logit_pipe, MODELS_DIR / f"{dataset_name}_logit.joblib")

    # ── 2. XGBoost ───────────────────────────────────────────────────────────
    log.info("Training XGBoost ...")
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / max(pos, 1)

    xgb_pipe = Pipeline([
        ("pre", pre),
        ("clf", xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            early_stopping_rounds=30,
            random_state=42,
            verbosity=0,
        )),
    ])

    # Fit XGBoost with eval set (need to transform data first)
    pre_fitted = Pipeline([("pre", pre)])
    pre_fitted.fit(X_train)
    X_train_t = pre_fitted.named_steps["pre"].transform(X_train)
    X_val_t   = pre_fitted.named_steps["pre"].transform(X_val)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    xgb_clf.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )
    log.info("  Best XGB iteration: %d", xgb_clf.best_iteration)

    # Wrap in calibrated pipeline for well-calibrated PD
    xgb_pipe_final = Pipeline([
        ("pre", pre),
        ("clf", CalibratedClassifierCV(xgb_clf, cv="prefit", method="isotonic")),
    ])
    # Calibrate on validation set
    X_val_transformed = pre.fit_transform(X_train)  # pre is already fitted via pre_fitted
    # Re-use pre from pre_fitted for the final pipeline
    xgb_pipe_final.steps[0] = ("pre", pre_fitted.named_steps["pre"])
    xgb_pipe_final.fit(X_val, y_val)
    joblib.dump(xgb_pipe_final, MODELS_DIR / f"{dataset_name}_xgb.joblib")

    # Also save raw XGBoost for feature importance
    joblib.dump(xgb_clf, MODELS_DIR / f"{dataset_name}_xgb_raw.joblib")

    # ── 3. Random Forest ────────────────────────────────────────────────────
    log.info("Training Random Forest ...")
    rf_pipe = Pipeline([
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=50,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    rf_pipe.fit(X_train, y_train)
    joblib.dump(rf_pipe, MODELS_DIR / f"{dataset_name}_rf.joblib")

    # ── Evaluate all models ───────────────────────────────────────────────────
    log.info("Evaluating on VALIDATION set (%d rows):", len(X_val))
    metrics_val = [
        _evaluate("logit_val",  logit_pipe,    X_val, y_val),
        _evaluate("xgb_val",    xgb_pipe_final, X_val, y_val),
        _evaluate("rf_val",     rf_pipe,        X_val, y_val),
    ]

    log.info("Evaluating on TEST set (%d rows):", len(X_test))
    metrics_test = [
        _evaluate("logit_test", logit_pipe,    X_test, y_test),
        _evaluate("xgb_test",   xgb_pipe_final, X_test, y_test),
        _evaluate("rf_test",    rf_pipe,        X_test, y_test),
    ]

    metrics = {"validation": metrics_val, "test": metrics_test}
    metrics_path = MODELS_DIR / f"{dataset_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved -> %s", metrics_path)

    # ── Feature importance (XGBoost) ─────────────────────────────────────────
    try:
        # Get feature names from the preprocessor
        ohe = pre_fitted.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
        cat_feat_names = list(ohe.get_feature_names_out(avail_cat))
        all_feat_names = avail_num + cat_feat_names

        importance = xgb_clf.feature_importances_
        if len(importance) == len(all_feat_names):
            fi_df = pd.DataFrame({
                "feature":    all_feat_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)
            fi_df.to_parquet(MODELS_DIR / f"{dataset_name}_feature_importance.parquet", index=False)
            log.info("Feature importance saved.")
    except Exception as e:
        log.warning("Could not save feature importance: %s", e)

    # Save the preprocessor separately for the ECL engine
    joblib.dump(pre_fitted.named_steps["pre"], MODELS_DIR / f"{dataset_name}_preprocessor.joblib")
    log.info("Preprocessor saved.")

    log.info("PD model training complete for dataset: %s", dataset_name)


if __name__ == "__main__":
    train_pd_models("pd_12m")
