"""
Train LGD (Loss Given Default) models.

Models:
  1. XGBoost Regressor (main model)
  2. Linear Regression baseline (for comparison)
  3. Two-stage model: classifier (loss vs no-loss) + regressor

Output saved to models/:
  lgd_xgb.joblib
  lgd_linear.joblib
  lgd_preprocessor.joblib
  lgd_metrics.json
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb

from src.config.settings import (
    DATA_PROCESSED, MODELS_DIR, FIGURES_DIR,
    TRAIN_YEARS, VALID_YEARS, TEST_YEARS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

TARGET = "lgd_observed"

NUMERIC_FEATURES = [
    "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
    "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
    "MI_PCT", "LOAN_AGE_AT_DEFAULT",
    "unemployment_rate_at_default", "hpi_yoy_chg_at_default",
    "rate_at_default", "fsi_at_default",
]

CATEGORICAL_FEATURES = [
    "OCC_STATUS", "PROPERTY_TYPE", "PROPERTY_STATE", "LOAN_PURPOSE",
]


def _evaluate_regression(name: str, pred: np.ndarray, actual: np.ndarray) -> dict:
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    r2   = r2_score(actual, pred)
    log.info("  %s | RMSE=%.4f | MAE=%.4f | R2=%.4f", name, rmse, mae, r2)
    return {"model": name, "rmse": rmse, "mae": mae, "r2": r2}


def _decile_calibration(pred: np.ndarray, actual: np.ndarray) -> pd.DataFrame:
    """Compare predicted vs actual LGD by predicted decile."""
    df = pd.DataFrame({"pred": pred, "actual": actual})
    df["decile"] = pd.qcut(df["pred"], 10, labels=False, duplicates="drop")
    return (
        df.groupby("decile")[["pred", "actual"]]
        .mean()
        .reset_index()
        .rename(columns={"decile": "decile_bin", "pred": "mean_pred", "actual": "mean_actual"})
    )


def train_lgd_models() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(DATA_PROCESSED / "lgd_dataset.parquet")
    log.info("LGD dataset: %d rows, mean LGD=%.4f", len(df), df[TARGET].mean())

    avail_num = [c for c in NUMERIC_FEATURES      if c in df.columns]
    avail_cat = [c for c in CATEGORICAL_FEATURES   if c in df.columns]

    X = df[avail_num + avail_cat].copy()
    y = df[TARGET].astype(float)

    train_mask = df["VINTAGE_YEAR"].isin(TRAIN_YEARS)
    valid_mask = df["VINTAGE_YEAR"].isin(VALID_YEARS)
    test_mask  = df["VINTAGE_YEAR"].isin(TEST_YEARS)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[valid_mask], y[valid_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    log.info(
        "Split: train=%d | val=%d | test=%d",
        len(X_train), len(X_val), len(X_test)
    )

    if len(X_train) == 0:
        log.error("No training data found. Check that the LGD dataset is populated.")
        return

    # If val/test are empty, use a cross-val fold from train data instead
    if len(X_val) == 0:
        log.warning("Validation set empty; using 20%% of train data as proxy val.")
        split_idx = int(len(X_train) * 0.8)
        X_val, y_val = X_train.iloc[split_idx:], y_train.iloc[split_idx:]
        X_train, y_train = X_train.iloc[:split_idx], y_train.iloc[:split_idx]
    if len(X_test) == 0:
        log.warning("Test set empty; metrics will be reported on val set only.")
        X_test, y_test = X_val, y_val

    # ── Preprocessor ─────────────────────────────────────────────────────────
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

    # ── 1. Ridge Regression baseline ─────────────────────────────────────────
    log.info("Training Ridge Regression baseline ...")
    ridge_pipe = Pipeline([("pre", pre), ("reg", Ridge(alpha=1.0))])
    ridge_pipe.fit(X_train, y_train)
    joblib.dump(ridge_pipe, MODELS_DIR / "lgd_linear.joblib")

    # ── 2. XGBoost Regressor ─────────────────────────────────────────────────
    log.info("Training XGBoost Regressor ...")

    pre_fit = Pipeline([("pre", pre)])
    pre_fit.fit(X_train)
    X_train_t = pre_fit.named_steps["pre"].transform(X_train)
    X_val_t   = pre_fit.named_steps["pre"].transform(X_val)

    xgb_reg = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=20,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=40,
        random_state=42,
        verbosity=0,
    )
    xgb_reg.fit(
        X_train_t, y_train,
        eval_set=[(X_val_t, y_val)],
        verbose=False,
    )
    log.info("  Best XGB iteration: %d", xgb_reg.best_iteration)

    # Clip predictions to [0, 1]
    class ClippedXGBRegressor:
        def __init__(self, model, pre):
            self.model = model
            self.pre   = pre

        def predict(self, X):
            Xt = self.pre.transform(X)
            return self.model.predict(Xt).clip(0, 1)

    xgb_pipe = {"preprocessor": pre_fit.named_steps["pre"], "model": xgb_reg}
    joblib.dump(xgb_pipe, MODELS_DIR / "lgd_xgb.joblib")

    # Save preprocessor
    joblib.dump(pre_fit.named_steps["pre"], MODELS_DIR / "lgd_preprocessor.joblib")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    log.info("Validation metrics:")
    metrics = {}
    for split_name, Xs, ys in [
        ("val",  X_val,  y_val),
        ("test", X_test, y_test),
    ]:
        ridge_pred = ridge_pipe.predict(Xs).clip(0, 1)
        xgb_pred   = xgb_reg.predict(
            pre_fit.named_steps["pre"].transform(Xs)
        ).clip(0, 1)

        metrics[f"ridge_{split_name}"] = _evaluate_regression(
            f"Ridge/{split_name}", ridge_pred, ys.values
        )
        metrics[f"xgb_{split_name}"] = _evaluate_regression(
            f"XGBoost/{split_name}", xgb_pred, ys.values
        )

        # Decile calibration
        calib = _decile_calibration(xgb_pred, ys.values)
        calib.to_parquet(
            MODELS_DIR / f"lgd_calibration_{split_name}.parquet", index=False
        )

    with open(MODELS_DIR / "lgd_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved -> lgd_metrics.json")

    # ── Feature importance ────────────────────────────────────────────────────
    try:
        ohe = pre_fit.named_steps["pre"].named_transformers_["cat"].named_steps["ohe"]
        cat_feat_names = list(ohe.get_feature_names_out(avail_cat))
        all_feat_names = avail_num + cat_feat_names
        importance = xgb_reg.feature_importances_
        if len(importance) == len(all_feat_names):
            fi_df = pd.DataFrame({
                "feature":    all_feat_names,
                "importance": importance,
            }).sort_values("importance", ascending=False)
            fi_df.to_parquet(MODELS_DIR / "lgd_feature_importance.parquet", index=False)
    except Exception as e:
        log.warning("Feature importance not saved: %s", e)

    log.info("LGD model training complete.")


if __name__ == "__main__":
    train_lgd_models()
