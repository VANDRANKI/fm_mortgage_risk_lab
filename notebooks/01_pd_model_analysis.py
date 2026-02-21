"""
PD Model Analysis and Validation
Run after training: py -3 src/models/train_pd.py
Then: py -3 notebooks/01_pd_model_analysis.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve, CalibrationDisplay

from src.config.settings import MODELS_DIR, DATA_PROCESSED, FIGURES_DIR, TEST_YEARS

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "figure.facecolor": "#0f172a", "axes.facecolor": "#0f172a",
    "axes.edgecolor": "#1e3a5f", "text.color": "#94a3b8",
    "axes.labelcolor": "#94a3b8", "xtick.color": "#64748b",
    "ytick.color": "#64748b", "grid.color": "#1e3a5f",
})
CYAN = "#22d3ee"; AMBER = "#f59e0b"; RED = "#ef4444"; PURPLE = "#a78bfa"


def load_test_data():
    df = pd.read_parquet(DATA_PROCESSED / "pd_12m_dataset.parquet")
    test = df[df["VINTAGE_YEAR"].isin(TEST_YEARS)]
    print(f"Test set: {len(test):,} loans, default rate: {test['PD_TARGET'].mean():.2%}")
    return test


def main():
    # Load test data
    test = load_test_data()
    if len(test) == 0:
        print("No test data found. Run the pipeline first.")
        return

    y_test = test["PD_TARGET"].astype(int)

    # Load metrics JSON
    metrics_path = MODELS_DIR / "pd_12m_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
        print("\n-- Model Metrics -------------------------------------")
        for m in metrics.get("test", []):
            print(f"  {m['model']:20s} | AUC={m['auc']:.4f} | KS={m['ks']:.4f} | Brier={m['brier']:.4f}")
    else:
        print("Metrics file not found. Run train_pd.py first.")
        return

    # Load best model (XGBoost)
    xgb_path = MODELS_DIR / "pd_12m_xgb.joblib"
    if not xgb_path.exists():
        print("Model not found. Run train_pd.py first.")
        return

    model = joblib.load(xgb_path)

    # Build feature matrix
    numeric_cols = [
        "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
        "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
        "MI_PCT", "RATE_SPREAD", "LOAN_AGE_AT_OBS",
        "unemployment_rate", "hpi_yoy_chg", "mortgage_rate_30y",
        "financial_stress_idx", "unrate_chg_mom", "DLQ_STATUS_INT",
    ]
    cat_cols = [
        "OCC_STATUS", "LOAN_PURPOSE", "PROPERTY_TYPE", "PROPERTY_STATE",
        "PRODUCT_TYPE", "CHANNEL", "FIRST_TIME_HOMEBUYER_FLAG",
    ]
    avail = [c for c in numeric_cols + cat_cols if c in test.columns]
    X_test = test[avail]

    try:
        prob = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Prediction failed: {e}")
        return

    # -- 1. ROC Curve ---------------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    ks  = np.max(tpr - fpr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.plot(fpr, tpr, color=CYAN, lw=2, label=f"XGBoost (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#334155", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color=CYAN)
    ax.set_title("ROC Curve — PD Model (Test 2015–2016)", color="white")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(facecolor="#0f172a")
    ax.text(0.6, 0.15, f"KS = {ks:.4f}", color=AMBER, fontsize=12)

    # -- 2. Calibration Curve -------------------------------------------------
    ax = axes[1]
    fraction_pos, mean_pred = calibration_curve(y_test, prob, n_bins=10)
    ax.plot(mean_pred, fraction_pos, "s-", color=CYAN, lw=2, label="XGBoost")
    ax.plot([0, 1], [0, 1], color="#334155", lw=1, linestyle="--", label="Perfect calibration")
    ax.set_title("Calibration Plot", color="white")
    ax.set_xlabel("Mean Predicted PD")
    ax.set_ylabel("Fraction of Defaults")
    ax.legend(facecolor="#0f172a")

    plt.suptitle("PD Model Validation — XGBoost", color="white", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "pd_validation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: pd_validation.png")

    # -- 3. Feature Importance -------------------------------------------------
    fi_path = MODELS_DIR / "pd_12m_feature_importance.parquet"
    if fi_path.exists():
        fi = pd.read_parquet(fi_path).head(20)
        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(fi["feature"][::-1], fi["importance"][::-1],
                       color=CYAN, alpha=0.85)
        ax.set_title("Top 20 Features — PD XGBoost Model", color="white")
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "pd_feature_importance.png", dpi=150)
        plt.close()
        print("  Saved: pd_feature_importance.png")

    print("\nPD model analysis complete. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
