# Freddie Mac Single-Family Loan-Level Dataset — Field Reference

## Source
Freddie Mac Single-Family Loan-Level Dataset (Sample Files)
- Registration & download: https://www.freddiemac.com/research/datasets/sf-loanlevel-dataset
- User Guide: https://www.freddiemac.com/fmac-resources/research/pdf/user_guide.pdf

---

## Origination File (`sample_orig_YYYY.txt`)

Pipe-delimited, no header. One row per loan at origination. 32 columns.

| # | Column Name | Type | Description |
|---|-------------|------|-------------|
| 1 | CREDIT_SCORE | Int16 | FICO credit score at origination. 9999 = missing. |
| 2 | FIRST_PAYMENT_DATE | datetime | YYYYMM — date of first scheduled payment. |
| 3 | FIRST_TIME_HOMEBUYER_FLAG | str | Y/N/9 — borrower is a first-time homebuyer. |
| 4 | MATURITY_DATE | datetime | YYYYMM — scheduled maturity of loan. |
| 5 | MSA | str | Metropolitan Statistical Area code. Blank = non-metro. |
| 6 | MI_PCT | float32 | Mortgage insurance percentage (0–55). 000 = no MI. |
| 7 | NUMBER_OF_UNITS | Int8 | 1–4 units. |
| 8 | OCC_STATUS | str | P=Primary, I=Investment, S=Second home. |
| 9 | ORIGINAL_CLTV | float32 | Combined LTV at origination (includes all liens). |
| 10 | ORIGINAL_DTI | float32 | Debt-to-income ratio. 999 = missing. |
| 11 | ORIGINAL_UPB | float64 | Original Unpaid Principal Balance ($). |
| 12 | ORIGINAL_LTV | float32 | First-lien LTV at origination. |
| 13 | ORIGINAL_INTEREST_RATE | float32 | Initial interest rate (%). |
| 14 | CHANNEL | str | R=Retail, B=Broker, C=Correspondent, T=TPO. |
| 15 | PPM_FLAG | str | Y=Prepayment penalty, N=None. |
| 16 | PRODUCT_TYPE | str | FRM=Fixed rate, ARM=Adjustable. |
| 17 | PROPERTY_STATE | str | Two-letter US state abbreviation. |
| 18 | PROPERTY_TYPE | str | SF=Single-family, CO=Condo, PU=PUD, MH=Manufactured. |
| 19 | POSTAL_CODE | str | 3-digit ZIP code prefix. |
| 20 | LOAN_SEQUENCE_NUMBER | str | **Primary key.** Freddie Mac unique loan identifier. |
| 21 | LOAN_PURPOSE | str | C=Purchase, N=No-cash refinance, R=Cash-out refinance. |
| 22 | ORIGINAL_LOAN_TERM | Int16 | Scheduled term in months (e.g., 360 = 30 years). |
| 23 | NUMBER_OF_BORROWERS | str | 01 or 02. 99 = missing. |
| 24 | SELLER_NAME | str | Selling institution name. |
| 25 | SERVICER_NAME | str | Servicing institution name. |
| 26 | SUPER_CONFORMING_FLAG | str | Y=Super-conforming loan. |
| 27 | PRE_HARP_LOAN_SEQUENCE_NUMBER | str | Prior loan reference (HARP refinances). |
| 28 | PROGRAM_INDICATOR | str | H=Home Possible, F=HFA, etc. 9=Not applicable. |
| 29 | HARP_INDICATOR | str | Y=HARP loan. |
| 30 | PROPERTY_VALUATION_METHOD | str | 1=Full appraisal, 2=Drive-by, 7=BPO. |
| 31 | IO_INDICATOR | str | Y=Interest-only. N=Fully amortizing. |
| 32 | MI_CANCELLATION_INDICATOR | str | Y=MI cancelled. |

---

## Servicing File (`sample_svcg_YYYY.txt`)

Pipe-delimited, no header. One row per loan per reporting month. 32 columns.

| # | Column Name | Type | Description |
|---|-------------|------|-------------|
| 1 | LOAN_SEQUENCE_NUMBER | str | **Foreign key** to origination file. |
| 2 | MONTHLY_REPORTING_PERIOD | str | YYYYMM — reporting month. |
| 3 | CURRENT_ACTUAL_UPB | float64 | Outstanding balance at end of period ($). |
| 4 | CURRENT_LOAN_DELINQUENCY_STATUS | str | 0=Current, 1=30dpd, 2=60dpd, 3=90dpd, ..., XX=Foreclosure/REO. |
| 5 | LOAN_AGE | Int16 | Months since origination. |
| 6 | REMAINING_MONTHS_TO_MATURITY | Int16 | Months remaining to scheduled maturity. |
| 7 | REPURCHASE_MAKE_WHOLE_PROCEEDS_FLAG | str | Flag for repurchase proceeds. |
| 8 | MODIFICATION_FLAG | str | Y=Modified, P=Prior modification. |
| 9 | ZERO_BALANCE_CODE | str | 01=Prepaid, 02-09=Loss/liquidation codes. |
| 10 | ZERO_BALANCE_EFFECTIVE_DATE | str | YYYYMM — date balance reached zero. |
| 11 | CURRENT_INTEREST_RATE | float32 | Current note rate (%). |
| 12 | CURRENT_DEFERRED_UPB | float64 | Deferred principal balance ($). |
| 13 | DDLPI | str | Due Date of Last Paid Installment. |
| 14 | MI_RECOVERIES | float64 | Mortgage insurance recovery ($). |
| 15 | NET_SALES_PROCEEDS | float64 | Net proceeds from property sale ($). |
| 16 | NON_MI_RECOVERIES | float64 | Other recoveries ($). |
| 17 | EXPENSES | float64 | Servicer expenses ($). |
| 18 | LEGAL_COSTS | float64 | Legal costs ($). |
| 19 | MAINTENANCE_PRESERVATION_COSTS | float64 | Property maintenance ($). |
| 20 | TAXES_INSURANCE | float64 | Taxes and insurance paid ($). |
| 21 | MISCELLANEOUS_EXPENSES | float64 | Other expenses ($). |
| 22 | ACTUAL_LOSS_CALCULATION | float64 | Freddie Mac calculated actual loss ($). |
| 23 | MODIFICATION_RELATED_NON_INTEREST_BEARING_UPB | float64 | Deferred interest on modified loans ($). |
| 24 | PRINCIPAL_FORGIVENESS_UPB | float64 | Principal forgiven ($). |
| 25–31 | Various | — | REO list prices, borrower assistance plan, delinquent interest. |
| 32 | CURRENT_UPB_SCHEDULED | float64 | Scheduled principal balance at period end ($). |

---

## Business Definitions (This Project)

| Term | Definition |
|------|-----------|
| **Default** | Loan reaches 90+ days past due (delinquency ≥ 3) OR receives a liquidation zero balance code (02–09). |
| **Prepayment** | Loan receives zero balance code 01 (voluntary payoff). |
| **Liquidation** | Loan receives zero balance codes 02–09 (3rd party sale, short sale, forfeiture, deed-in-lieu, REO, etc.). |
| **EAD** | Exposure at Default = current outstanding UPB at time of default/liquidation. |
| **LGD** | Loss Given Default = (EAD − Net Proceeds − MI Recoveries − Non-MI Recoveries + Expenses) / EAD, clamped to [0,1]. |
| **ECL** | Expected Credit Loss = PD × LGD × EAD. |
| **Serious Delinquency** | 90+ days past due or foreclosure/REO status. |

---

## Zero Balance Code Reference

| Code | Meaning |
|------|---------|
| 01 | Voluntary prepayment / payoff |
| 02 | Third-party sale (loss) |
| 03 | Short sale (loss) |
| 04 | Forfeiture / deed from borrower (loss) |
| 05 | Repurchase prior to disposition (loss) |
| 06 | Repurchase (loss) |
| 09 | Real estate owned (REO) disposition (loss) |
