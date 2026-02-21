"""
ECL (Expected Credit Loss) computation engine.

ECL = PD × LGD × EAD

Supports:
  - IFRS 9 Stage 1/2/3 classification
  - Scenario-adjusted macro inputs
  - Portfolio-level aggregations by state, FICO band, vintage, product type

Usage:
  from src.models.ecl_engine import ECLEngine
  engine = ECLEngine()
  results = engine.compute_portfolio_ecl(df_loans, macro_scenario={"unemployment_shock": 2.0, ...})
"""
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.config.settings import MODELS_DIR, DATA_MACRO

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Delinquency thresholds for IFRS 9 staging
STAGE1_MAX_DLQ = 0    # Current or 1-29 dpd
STAGE2_MAX_DLQ = 60   # 30-89 dpd (significant increase in credit risk)
# Stage 3: 90+ dpd or in workout


class ECLEngine:
    """Loads trained PD and LGD models and computes ECL for a loan portfolio."""

    def __init__(self, dataset_name: str = "pd_12m"):
        self.dataset_name = dataset_name
        self.pd_model   = None
        self.lgd_model  = None
        self.lgd_pre    = None
        self._macro_baseline: Optional[pd.Series] = None
        self._load_models()

    def _load_models(self) -> None:
        """Load fitted model artifacts from disk."""
        pd_path  = MODELS_DIR / f"{self.dataset_name}_xgb.joblib"
        lgd_path = MODELS_DIR / "lgd_xgb.joblib"

        if pd_path.exists():
            self.pd_model = joblib.load(pd_path)
            log.info("PD model loaded from %s", pd_path)
        else:
            log.warning("PD model not found at %s – using heuristic fallback", pd_path)

        if lgd_path.exists():
            lgd_data = joblib.load(lgd_path)
            self.lgd_model = lgd_data["model"]
            self.lgd_pre   = lgd_data["preprocessor"]
            log.info("LGD model loaded from %s", lgd_path)
        else:
            log.warning("LGD model not found at %s – using heuristic fallback", lgd_path)

    def _load_baseline_macro(self) -> pd.Series:
        """Get latest available macro observation as the baseline."""
        if self._macro_baseline is not None:
            return self._macro_baseline

        macro_path = DATA_MACRO / "fred_macro.parquet"
        if macro_path.exists():
            macro = pd.read_parquet(macro_path)
            macro["date"] = pd.to_datetime(macro["date"])
            # Use 2016 average as baseline (matches our loan cohort)
            base = macro[macro["date"].dt.year == 2016].mean(numeric_only=True)
        else:
            # Hard-coded 2016 baseline if no macro file
            base = pd.Series({
                "unemployment_rate":    4.87,
                "hpi_yoy_chg":          5.6,
                "mortgage_rate_30y":    3.65,
                "financial_stress_idx": -0.5,
                "unrate_chg_mom":        0.0,
                "hpi_us":              199.0,
            })
        self._macro_baseline = base
        return base

    def _apply_scenario(
        self,
        macro_base: pd.Series,
        scenario: dict,
    ) -> pd.Series:
        """
        Apply scenario shocks to baseline macro.

        scenario keys:
          unemployment_shock  (percentage points added)
          hpi_shock           (percentage points added to YoY HPI change)
          rate_shock          (percentage points added to mortgage rate)
        """
        stressed = macro_base.copy()
        unemp_shock = scenario.get("unemployment_shock", 0.0)
        stressed["unemployment_rate"]    += unemp_shock
        stressed["hpi_yoy_chg"]          += scenario.get("hpi_shock", 0.0)
        stressed["mortgage_rate_30y"]    += scenario.get("rate_shock", 0.0)
        # Propagate rising-unemployment signal
        stressed["unrate_chg_mom"]       += unemp_shock / 12.0   # spread over 12 months
        return stressed

    def _heuristic_pd(self, df: pd.DataFrame, macro: pd.Series) -> np.ndarray:
        """
        Simple logistic heuristic when no trained model is available.
        Used for demo/fallback only.
        """
        # Base PD from FICO and LTV
        fico    = df.get("CREDIT_SCORE", pd.Series([700] * len(df))).fillna(700)
        ltv     = df.get("ORIGINAL_LTV", pd.Series([80] * len(df))).fillna(80)
        dti     = df.get("ORIGINAL_DTI",  pd.Series([35] * len(df))).fillna(35)
        unrate  = macro.get("unemployment_rate", 5.0)

        logit = (
            -4.5
            + (700 - fico) * 0.01
            + (ltv  - 80)  * 0.03
            + (dti  - 35)  * 0.01
            + (unrate - 5) * 0.15
        )
        return 1 / (1 + np.exp(-logit))

    def _heuristic_lgd(self, df: pd.DataFrame, macro: pd.Series) -> np.ndarray:
        """Simple LTV-based LGD heuristic."""
        ltv      = df.get("ORIGINAL_LTV",     pd.Series([80] * len(df))).fillna(80)
        hpi_chg  = macro.get("hpi_yoy_chg",  0.0)
        # Stressed LTV = original LTV / (1 + HPI change %)
        stressed_ltv = ltv / (1 + hpi_chg / 100).clip(0.5, 1.5)
        # LGD ≈ max(0, stressed LTV – 1) + expenses buffer
        lgd = (stressed_ltv / 100 - 0.85).clip(0, 0.8) + 0.15
        return lgd.clip(0.05, 0.90).values

    def predict_pd(self, X_loan: pd.DataFrame, macro: pd.Series) -> np.ndarray:
        """Predict 12-month PD for each loan given macro conditions."""
        Xm = X_loan.copy()

        # Inject current macro values
        for col in ["unemployment_rate", "hpi_yoy_chg", "mortgage_rate_30y",
                    "financial_stress_idx", "unrate_chg_mom"]:
            Xm[col] = macro.get(col, np.nan)

        # Derive features that the model expects but the raw panel may not have
        if "LOAN_AGE_AT_OBS" not in Xm.columns:
            if "LOAN_AGE_MONTHS" in Xm.columns:
                Xm["LOAN_AGE_AT_OBS"] = Xm["LOAN_AGE_MONTHS"]
            elif "LOAN_AGE" in Xm.columns:
                Xm["LOAN_AGE_AT_OBS"] = Xm["LOAN_AGE"]

        if "RATE_SPREAD" not in Xm.columns and "ORIGINAL_INTEREST_RATE" in Xm.columns:
            # RATE_SPREAD captures adverse selection at origination, not current macro.
            # Use the 2016 baseline market rate (constant) so that stress scenarios
            # (which change mortgage_rate_30y) don't perversely reduce RATE_SPREAD.
            baseline_rate = self._load_baseline_macro().get("mortgage_rate_30y", 3.65)
            Xm["RATE_SPREAD"] = Xm["ORIGINAL_INTEREST_RATE"] - baseline_rate

        # DLQ_STATUS_INT: if the panel has it, use it; else default to 0
        if "DLQ_STATUS_INT" not in Xm.columns:
            Xm["DLQ_STATUS_INT"] = 0

        if self.pd_model is not None:
            try:
                return self.pd_model.predict_proba(Xm)[:, 1]
            except Exception as e:
                log.warning("PD model prediction failed: %s – using heuristic", e)

        return self._heuristic_pd(Xm, macro)

    def predict_lgd(self, X_loan: pd.DataFrame, macro: pd.Series) -> np.ndarray:
        """Predict LGD for each loan given macro conditions."""
        Xm = X_loan.copy()

        # Map current macro to the "at_default" names the LGD model was trained on
        macro_map = {
            "unemployment_rate_at_default": "unemployment_rate",
            "hpi_yoy_chg_at_default":       "hpi_yoy_chg",
            "rate_at_default":              "mortgage_rate_30y",
            "fsi_at_default":               "financial_stress_idx",
        }
        for dest, src in macro_map.items():
            Xm[dest] = macro.get(src, np.nan)

        # Derive LOAN_AGE_AT_DEFAULT if missing
        if "LOAN_AGE_AT_DEFAULT" not in Xm.columns:
            if "LOAN_AGE_MONTHS" in Xm.columns:
                Xm["LOAN_AGE_AT_DEFAULT"] = Xm["LOAN_AGE_MONTHS"]
            elif "LOAN_AGE" in Xm.columns:
                Xm["LOAN_AGE_AT_DEFAULT"] = Xm["LOAN_AGE"]
            else:
                Xm["LOAN_AGE_AT_DEFAULT"] = 24.0  # reasonable fallback

        if self.lgd_model is not None and self.lgd_pre is not None:
            try:
                Xt   = self.lgd_pre.transform(Xm)
                pred = self.lgd_model.predict(Xt).clip(0, 1)
                return pred
            except Exception as e:
                log.warning("LGD model prediction failed: %s – using heuristic", e)

        return self._heuristic_lgd(Xm, macro)

    def classify_ifrs9_stage(self, df: pd.DataFrame, pd_arr: np.ndarray) -> np.ndarray:
        """
        Assign IFRS 9 stage based on delinquency and PD increase.
          Stage 1: PD < 1% and DLQ < 30dpd  → 12-month ECL
          Stage 2: PD 1-10% or DLQ 30-89dpd → Lifetime ECL
          Stage 3: PD > 10% or DLQ 90+ dpd  → Lifetime ECL (impaired)
        """
        dlq = df.get("DLQ_STATUS_INT", pd.Series(np.zeros(len(df)))).fillna(0)
        stages = np.where(
            (pd_arr > 0.10) | (dlq >= 3),  3,
            np.where(
                (pd_arr > 0.01) | (dlq >= 1), 2, 1
            )
        )
        return stages

    def compute_ecl(
        self,
        df_loans: pd.DataFrame,
        macro_scenario: dict | None = None,
        ead_col: str = "CURRENT_ACTUAL_UPB",
    ) -> pd.DataFrame:
        """
        Compute ECL per loan.

        Parameters
        ----------
        df_loans      : DataFrame with loan features
        macro_scenario: dict with shock values (unemployment_shock, hpi_shock, rate_shock)
        ead_col       : column to use as Exposure at Default

        Returns
        -------
        DataFrame with added columns: pd_pred, lgd_pred, ead, ecl, ifrs9_stage
        """
        macro_base = self._load_baseline_macro()
        scenario   = macro_scenario or {}
        macro      = self._apply_scenario(macro_base, scenario)

        result = df_loans.copy()

        # EAD
        if ead_col in result.columns:
            result["ead"] = result[ead_col].fillna(result.get("ORIGINAL_UPB", 0)).fillna(0)
        else:
            result["ead"] = result.get("ORIGINAL_UPB", 0).fillna(0)

        # PD
        result["pd_pred"] = self.predict_pd(result, macro).clip(0, 1)

        # LGD  (add lifetime multiplier for Stage 2/3)
        result["lgd_pred"] = self.predict_lgd(result, macro).clip(0, 1)

        # IFRS 9 Stage
        result["ifrs9_stage"] = self.classify_ifrs9_stage(result, result["pd_pred"].values)

        # Lifetime multiplier: Stage 1 uses 12m PD, Stage 2/3 use lifetime
        lifetime_mult = np.where(result["ifrs9_stage"] == 1, 1.0, 3.0)  # rough 3x for lifetime
        pd_effective  = (result["pd_pred"] * lifetime_mult).clip(0, 1)

        # ECL
        result["ecl"] = pd_effective * result["lgd_pred"] * result["ead"]

        return result

    def compute_portfolio_ecl(
        self,
        df_loans: pd.DataFrame,
        macro_scenario: dict | None = None,
    ) -> dict:
        """
        Compute ECL for full portfolio and return aggregated metrics.
        """
        df = self.compute_ecl(df_loans, macro_scenario)

        total_ead = df["ead"].sum()
        total_ecl = df["ecl"].sum()
        ecl_rate  = total_ecl / max(total_ead, 1)

        # By segment
        def agg_by(col: str, df: pd.DataFrame) -> list[dict]:
            if col not in df.columns:
                return []
            g = (
                df.groupby(col, observed=True)
                .agg(
                    ead=("ead", "sum"),
                    ecl=("ecl", "sum"),
                    loan_count=(col, "count"),
                    mean_pd=("pd_pred", "mean"),
                    mean_lgd=("lgd_pred", "mean"),
                )
                .reset_index()
            )
            g["ecl_rate"] = g["ecl"] / g["ead"].replace(0, np.nan)
            return g.to_dict(orient="records")

        fico_bands = pd.cut(
            df["CREDIT_SCORE"].fillna(650),
            bins=[0, 620, 660, 700, 740, 800, 900],
            labels=["<620", "620-659", "660-699", "700-739", "740-799", "800+"],
        )
        df["FICO_BAND"] = fico_bands

        ltv_bands = pd.cut(
            df["ORIGINAL_LTV"].fillna(80),
            bins=[0, 60, 70, 80, 90, 100, 200],
            labels=["<60", "60-69", "70-79", "80-89", "90-99", "100+"],
        )
        df["LTV_BAND"] = ltv_bands

        stage_summary = (
            df.groupby("ifrs9_stage")
            .agg(
                loan_count=("ecl", "count"),
                ead=("ead", "sum"),
                ecl=("ecl", "sum"),
            )
            .reset_index()
            .to_dict(orient="records")
        )

        return {
            "total_ead":          float(total_ead),
            "total_ecl":          float(total_ecl),
            "ecl_rate":           float(ecl_rate),
            "loan_count":         int(len(df)),
            "mean_pd":            float(df["pd_pred"].mean()),
            "mean_lgd":           float(df["lgd_pred"].mean()),
            "by_state":           agg_by("PROPERTY_STATE", df),
            "by_fico_band":       agg_by("FICO_BAND", df),
            "by_ltv_band":        agg_by("LTV_BAND", df),
            "by_vintage":         agg_by("VINTAGE_YEAR", df),
            "by_product_type":    agg_by("PRODUCT_TYPE", df),
            "by_ifrs9_stage":     stage_summary,
        }
