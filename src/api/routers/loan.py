"""
Single-loan prediction endpoint.

POST /loan/predict  – predict PD, LGD, ECL for a single loan
"""
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter

from src.api.schemas import LoanPredictRequest, LoanPredictResponse
from src.models.ecl_engine import ECLEngine

router = APIRouter(prefix="/loan", tags=["Loan"])
log    = logging.getLogger(__name__)

_ENGINE: ECLEngine | None = None


def _get_engine() -> ECLEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = ECLEngine()
    return _ENGINE


def _risk_level(pd: float) -> str:
    if pd < 0.01:
        return "LOW"
    if pd < 0.03:
        return "MEDIUM"
    if pd < 0.08:
        return "HIGH"
    return "VERY HIGH"


@router.post("/predict", response_model=LoanPredictResponse)
def predict_loan(req: LoanPredictRequest):
    """Predict PD, LGD, ECL and IFRS 9 stage for a single loan."""
    engine = _get_engine()

    # Build a one-row DataFrame matching the expected feature schema
    loan_df = pd.DataFrame([{
        "CREDIT_SCORE":          req.credit_score,
        "ORIGINAL_LTV":          req.original_ltv,
        "ORIGINAL_CLTV":         req.original_cltv,
        "ORIGINAL_DTI":          req.original_dti,
        "ORIGINAL_UPB":          req.original_upb,
        "ORIGINAL_INTEREST_RATE":req.original_interest_rate,
        "ORIGINAL_LOAN_TERM":    req.original_loan_term,
        "OCC_STATUS":            req.occ_status,
        "LOAN_PURPOSE":          req.loan_purpose,
        "PROPERTY_TYPE":         req.property_type,
        "PROPERTY_STATE":        req.property_state,
        "PRODUCT_TYPE":          req.product_type,
        "CHANNEL":               req.channel,
        "FIRST_TIME_HOMEBUYER_FLAG": req.first_time_homebuyer,
        "MI_PCT":                req.mi_pct,
        "CURRENT_ACTUAL_UPB":    req.original_upb,  # EAD = original UPB
        "DLQ_STATUS_INT":        0,
        "LOAN_AGE_AT_OBS":       0,                 # new loan at origination
        # Note: RATE_SPREAD deliberately omitted so ECLEngine computes it from
        # ORIGINAL_INTEREST_RATE minus baseline market rate.
        "unrate_chg_mom":        0.0,
    }])

    macro_shock = {
        "unemployment_shock": req.unemployment_shock,
        "hpi_shock":          req.hpi_shock,
        "rate_shock":         req.rate_shock,
    }

    result_df = engine.compute_ecl(loan_df, macro_scenario=macro_shock)
    row = result_df.iloc[0]

    pd_val   = float(row["pd_pred"])
    lgd_val  = float(row["lgd_pred"])
    ead_val  = float(row["ead"])
    ecl_val  = float(row["ecl"])
    stage    = int(row["ifrs9_stage"])

    return LoanPredictResponse(
        pd_12m       = round(pd_val, 6),
        lgd          = round(lgd_val, 4),
        ead          = round(ead_val, 2),
        ecl          = round(ecl_val, 2),
        ecl_rate     = round(ecl_val / max(ead_val, 1), 6),
        ifrs9_stage  = stage,
        risk_level   = _risk_level(pd_val),
    )
