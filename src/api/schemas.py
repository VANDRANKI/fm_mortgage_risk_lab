"""
Pydantic schemas for request/response validation.
"""
from typing import Optional
from pydantic import BaseModel, Field


# ── Scenario ──────────────────────────────────────────────────────────────────
class ScenarioRequest(BaseModel):
    unemployment_shock: float = Field(0.0, ge=-5, le=20,  description="pp change in unemployment")
    hpi_shock:          float = Field(0.0, ge=-50, le=20, description="pp change in HPI YoY")
    rate_shock:         float = Field(0.0, ge=-3, le=10,  description="pp change in mortgage rate")
    scenario_name:      str   = Field("custom", description="Label for this scenario")


class ScenarioKPIs(BaseModel):
    total_ead:    float
    total_ecl:    float
    ecl_rate:     float
    loan_count:   int
    mean_pd:      float
    mean_lgd:     float
    scenario_name: str


# ── Loan Prediction ────────────────────────────────────────────────────────────
class LoanPredictRequest(BaseModel):
    credit_score:           int     = Field(700,    ge=300, le=850)
    original_ltv:           float   = Field(80.0,   ge=0,   le=200)
    original_cltv:          float   = Field(80.0,   ge=0,   le=200)
    original_dti:           float   = Field(35.0,   ge=0,   le=70)
    original_upb:           float   = Field(200000, ge=1000)
    original_interest_rate: float   = Field(4.5,    ge=0,   le=20)
    original_loan_term:     int     = Field(360,    ge=60,  le=480)
    occ_status:             str     = Field("P",    description="P=Primary, I=Investment, S=Second")
    loan_purpose:           str     = Field("C",    description="C=Purchase, N=Refinance, R=Cash-out")
    property_type:          str     = Field("SF",   description="SF/CO/PU/MH")
    property_state:         str     = Field("CA")
    product_type:           str     = Field("FRM",  description="FRM or ARM")
    channel:                str     = Field("R",    description="R=Retail, B=Broker, C=Correspondent")
    first_time_homebuyer:   str     = Field("N",    description="Y/N")
    mi_pct:                 float   = Field(0.0,    ge=0,   le=35)
    # Macro overrides (optional)
    unemployment_shock:     float   = Field(0.0)
    hpi_shock:              float   = Field(0.0)
    rate_shock:             float   = Field(0.0)


class LoanPredictResponse(BaseModel):
    pd_12m:        float
    lgd:           float
    ead:           float
    ecl:           float
    ecl_rate:      float
    ifrs9_stage:   int
    risk_level:    str    # LOW / MEDIUM / HIGH / VERY HIGH
    pd_drivers:    Optional[list[dict]] = None
