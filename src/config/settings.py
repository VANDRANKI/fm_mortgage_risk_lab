"""
Central configuration for the Mortgage Credit Risk & Stress Testing Lab.
All paths, column definitions, and constants live here.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ── Root paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]

DATA_RAW        = ROOT / "data" / "raw" / "freddie"
DATA_INTERIM    = ROOT / "data" / "interim"
DATA_PROCESSED  = ROOT / "data" / "processed"
DATA_MACRO      = ROOT / "data" / "macro"
MODELS_DIR      = ROOT / "models"
REPORTS_DIR     = ROOT / "reports"
FIGURES_DIR     = ROOT / "reports" / "figures"
LOGS_DIR        = ROOT / "logs"
PORTFOLIO_BASELINE = DATA_PROCESSED / "portfolio_baseline"

# ── Vintage years available ────────────────────────────────────────────────────
VINTAGE_YEARS = list(range(2010, 2017))  # 2010-2016

# ── Origination file column names (32 columns, pipe-delimited, no header) ─────
ORIG_COLS = [
    "CREDIT_SCORE",                    # 01 – FICO score
    "FIRST_PAYMENT_DATE",              # 02 – YYYYMM
    "FIRST_TIME_HOMEBUYER_FLAG",       # 03 – Y/N/9
    "MATURITY_DATE",                   # 04 – YYYYMM
    "MSA",                             # 05 – Metropolitan Statistical Area code
    "MI_PCT",                          # 06 – Mortgage Insurance %
    "NUMBER_OF_UNITS",                 # 07 – 1-4
    "OCC_STATUS",                      # 08 – P/I/S (Primary/Investment/Second)
    "ORIGINAL_CLTV",                   # 09 – Combined LTV at origination
    "ORIGINAL_DTI",                    # 10 – Debt-to-Income ratio
    "ORIGINAL_UPB",                    # 11 – Unpaid Principal Balance
    "ORIGINAL_LTV",                    # 12 – Loan-to-Value ratio
    "ORIGINAL_INTEREST_RATE",          # 13 – Interest rate %
    "CHANNEL",                         # 14 – R/B/C/T (Retail/Broker/Correspondent/TPO)
    "PPM_FLAG",                        # 15 – Prepayment Penalty Y/N
    "PRODUCT_TYPE",                    # 16 – FRM/ARM
    "PROPERTY_STATE",                  # 17 – Two-letter state code
    "PROPERTY_TYPE",                   # 18 – SF/CO/PU/MH/CP
    "POSTAL_CODE",                     # 19 – 3-digit zip prefix
    "LOAN_SEQUENCE_NUMBER",            # 20 – Primary key
    "LOAN_PURPOSE",                    # 21 – C/N/R/U (Purchase/Refi/etc.)
    "ORIGINAL_LOAN_TERM",              # 22 – Months (e.g., 360)
    "NUMBER_OF_BORROWERS",             # 23 – 01/02
    "SELLER_NAME",                     # 24
    "SERVICER_NAME",                   # 25
    "SUPER_CONFORMING_FLAG",           # 26 – Y/N/blank
    "PRE_HARP_LOAN_SEQUENCE_NUMBER",   # 27 – Prior loan ref
    "PROGRAM_INDICATOR",               # 28 – H/F/R/S/Y/9
    "HARP_INDICATOR",                  # 29 – Y/N/blank
    "PROPERTY_VALUATION_METHOD",       # 30 – 1/2/3/7/9
    "IO_INDICATOR",                    # 31 – Y/N/9
    "MI_CANCELLATION_INDICATOR",       # 32 – Y/N/9
]

# ── Origination dtype mapping ─────────────────────────────────────────────────
ORIG_DTYPES = {
    "CREDIT_SCORE":             "Int16",
    "FIRST_PAYMENT_DATE":       "str",
    "FIRST_TIME_HOMEBUYER_FLAG":"str",
    "MATURITY_DATE":            "str",
    "MSA":                      "str",
    "MI_PCT":                   "float32",
    "NUMBER_OF_UNITS":          "Int8",
    "OCC_STATUS":               "str",
    "ORIGINAL_CLTV":            "float32",
    "ORIGINAL_DTI":             "float32",
    "ORIGINAL_UPB":             "float64",
    "ORIGINAL_LTV":             "float32",
    "ORIGINAL_INTEREST_RATE":   "float32",
    "CHANNEL":                  "str",
    "PPM_FLAG":                 "str",
    "PRODUCT_TYPE":             "str",
    "PROPERTY_STATE":           "str",
    "PROPERTY_TYPE":            "str",
    "POSTAL_CODE":              "str",
    "LOAN_SEQUENCE_NUMBER":     "str",
    "LOAN_PURPOSE":             "str",
    "ORIGINAL_LOAN_TERM":       "Int16",
    "NUMBER_OF_BORROWERS":      "str",
    "SELLER_NAME":              "str",
    "SERVICER_NAME":            "str",
    "SUPER_CONFORMING_FLAG":    "str",
    "PRE_HARP_LOAN_SEQUENCE_NUMBER": "str",
    "PROGRAM_INDICATOR":        "str",
    "HARP_INDICATOR":           "str",
    "PROPERTY_VALUATION_METHOD":"str",
    "IO_INDICATOR":             "str",
    "MI_CANCELLATION_INDICATOR":"str",
}

# ── Servicing file column names (32 columns, pipe-delimited, no header) ────────
SVCG_COLS = [
    "LOAN_SEQUENCE_NUMBER",                           # 01 – Foreign key to orig
    "MONTHLY_REPORTING_PERIOD",                       # 02 – YYYYMM
    "CURRENT_ACTUAL_UPB",                             # 03 – Outstanding balance
    "CURRENT_LOAN_DELINQUENCY_STATUS",                # 04 – 0,1,2,...,XX
    "LOAN_AGE",                                       # 05 – Months since origination
    "REMAINING_MONTHS_TO_MATURITY",                   # 06
    "REPURCHASE_MAKE_WHOLE_PROCEEDS_FLAG",            # 07
    "MODIFICATION_FLAG",                              # 08 – Y/P/blank
    "ZERO_BALANCE_CODE",                              # 09 – 01-09
    "ZERO_BALANCE_EFFECTIVE_DATE",                    # 10 – YYYYMM
    "CURRENT_INTEREST_RATE",                          # 11
    "CURRENT_DEFERRED_UPB",                           # 12
    "DDLPI",                                          # 13 – Due Date of Last Paid Installment
    "MI_RECOVERIES",                                  # 14
    "NET_SALES_PROCEEDS",                             # 15
    "NON_MI_RECOVERIES",                              # 16
    "EXPENSES",                                       # 17
    "LEGAL_COSTS",                                    # 18
    "MAINTENANCE_PRESERVATION_COSTS",                 # 19
    "TAXES_INSURANCE",                                # 20
    "MISCELLANEOUS_EXPENSES",                         # 21
    "ACTUAL_LOSS_CALCULATION",                        # 22
    "MODIFICATION_RELATED_NON_INTEREST_BEARING_UPB",  # 23
    "PRINCIPAL_FORGIVENESS_UPB",                      # 24
    "ORIGINAL_LIST_START_DATE",                       # 25
    "ORIGINAL_LIST_PRICE",                            # 26
    "CURRENT_LIST_START_DATE",                        # 27
    "CURRENT_LIST_PRICE",                             # 28
    "BORROWER_ASSISTANCE_PLAN",                       # 29
    "VALUE_OF_DELINQUENT_ACCRUED_INTEREST",           # 30
    "CURRENT_NON_INTEREST_BEARING_UPB",               # 31
    "CURRENT_UPB_SCHEDULED",                          # 32
]

# ── Servicing dtype mapping ───────────────────────────────────────────────────
SVCG_DTYPES = {
    "LOAN_SEQUENCE_NUMBER":         "str",
    "MONTHLY_REPORTING_PERIOD":     "str",
    "CURRENT_ACTUAL_UPB":           "float64",
    "CURRENT_LOAN_DELINQUENCY_STATUS": "str",
    "LOAN_AGE":                     "Int16",
    "REMAINING_MONTHS_TO_MATURITY": "Int16",
    "REPURCHASE_MAKE_WHOLE_PROCEEDS_FLAG": "str",
    "MODIFICATION_FLAG":            "str",
    "ZERO_BALANCE_CODE":            "str",
    "ZERO_BALANCE_EFFECTIVE_DATE":  "str",
    "CURRENT_INTEREST_RATE":        "float32",
    "CURRENT_DEFERRED_UPB":         "float64",
    "DDLPI":                        "str",
    "MI_RECOVERIES":                "float64",
    "NET_SALES_PROCEEDS":           "float64",
    "NON_MI_RECOVERIES":            "float64",
    "EXPENSES":                     "float64",
    "LEGAL_COSTS":                  "float64",
    "MAINTENANCE_PRESERVATION_COSTS": "float64",
    "TAXES_INSURANCE":              "float64",
    "MISCELLANEOUS_EXPENSES":       "float64",
    "ACTUAL_LOSS_CALCULATION":      "float64",
    "MODIFICATION_RELATED_NON_INTEREST_BEARING_UPB": "float64",
    "PRINCIPAL_FORGIVENESS_UPB":    "float64",
    "ORIGINAL_LIST_START_DATE":     "str",
    "ORIGINAL_LIST_PRICE":          "float64",
    "CURRENT_LIST_START_DATE":      "str",
    "CURRENT_LIST_PRICE":           "float64",
    "BORROWER_ASSISTANCE_PLAN":     "str",
    "VALUE_OF_DELINQUENT_ACCRUED_INTEREST": "float64",
    "CURRENT_NON_INTEREST_BEARING_UPB": "float64",
    "CURRENT_UPB_SCHEDULED":        "float64",
}

# ── Business definitions ──────────────────────────────────────────────────────
# Delinquency: '0'=Current, '1'=30dpd, '2'=60dpd, '3'=90dpd, etc., 'XX'=Unknown/Foreclosure
SERIOUS_DELINQUENCY_CODES = {"3", "4", "5", "6", "7", "8", "9",
                              "XX", "RA"}  # 90+dpd or in foreclosure/REO

# Zero Balance Codes meaning loss/liquidation (not voluntary payoff)
LIQUIDATION_ZERO_BALANCE_CODES = {"02", "03", "04", "05", "06", "09"}
PREPAYMENT_ZERO_BALANCE_CODE   = "01"

# ── FRED macro series to pull ─────────────────────────────────────────────────
FRED_SERIES = {
    "UNRATE":   "unemployment_rate",     # Civilian Unemployment Rate (monthly)
    "USSTHPI":  "hpi_us",                # U.S. House Price Index (quarterly → ffill)
    "MORTGAGE30US": "mortgage_rate_30y", # 30-yr Fixed Mortgage Rate (weekly → monthly)
    "STLFSI4":  "financial_stress_idx",  # St. Louis Fed Financial Stress Index (weekly)
    "GDPC1":    "real_gdp",              # Real GDP (quarterly → ffill)
    "CPIAUCSL": "cpi",                   # Consumer Price Index
}

FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# ── Model training splits ─────────────────────────────────────────────────────
TRAIN_YEARS   = [2010, 2011, 2012, 2013]
VALID_YEARS   = [2014]
TEST_YEARS    = [2015, 2016]

# ── Feature lists for PD model ────────────────────────────────────────────────
PD_LOAN_FEATURES = [
    "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_CLTV", "ORIGINAL_DTI",
    "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE", "ORIGINAL_LOAN_TERM",
    "OCC_STATUS", "LOAN_PURPOSE", "PROPERTY_TYPE", "PROPERTY_STATE",
    "PRODUCT_TYPE", "CHANNEL", "NUMBER_OF_UNITS",
    "FIRST_TIME_HOMEBUYER_FLAG", "MI_PCT",
    # Derived
    "RATE_SPREAD",          # rate above market rate at origination
    "LOAN_AGE_AT_OBS",      # months since origination at observation
]

PD_MACRO_FEATURES = [
    "unemployment_rate", "hpi_yoy_chg", "mortgage_rate_30y", "financial_stress_idx",
]

PD_FEATURES = PD_LOAN_FEATURES + PD_MACRO_FEATURES

# ── Feature lists for LGD model ───────────────────────────────────────────────
LGD_FEATURES = [
    "CREDIT_SCORE", "ORIGINAL_LTV", "ORIGINAL_UPB", "ORIGINAL_INTEREST_RATE",
    "ORIGINAL_LOAN_TERM", "OCC_STATUS", "PROPERTY_TYPE", "PROPERTY_STATE",
    "LOAN_PURPOSE", "LOAN_AGE_AT_DEFAULT",
    # Macro at default
    "unemployment_rate_at_default", "hpi_yoy_chg_at_default",
]

# ── Scenario definitions ──────────────────────────────────────────────────────
SCENARIOS = {
    "baseline": {
        "unemployment_shock": 0.0,
        "hpi_shock":          0.0,
        "rate_shock":         0.0,
        "label":              "Baseline",
        "color":              "#22d3ee",   # cyan
    },
    "mild": {
        "unemployment_shock": 2.0,
        "hpi_shock":         -5.0,
        "rate_shock":         1.0,
        "label":              "Mild Stress",
        "color":              "#f59e0b",   # amber
    },
    "severe": {
        "unemployment_shock": 5.0,
        "hpi_shock":        -20.0,
        "rate_shock":         2.5,
        "label":              "Severe Stress",
        "color":              "#ef4444",   # red
    },
    "gfc": {
        "unemployment_shock": 8.0,
        "hpi_shock":        -33.0,
        "rate_shock":         0.5,
        "label":              "GFC-like",
        "color":              "#7c3aed",   # purple
    },
}
