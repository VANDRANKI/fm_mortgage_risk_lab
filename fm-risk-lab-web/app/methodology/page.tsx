"use client";
import { motion } from "framer-motion";

const SECTIONS = [
  {
    title: "Data",
    content: `
      In this lab, I use the Freddie Mac Single-Family Loan-Level dataset (2010–2016 vintages).

      There are two main files:
      - Origination data: FICO, LTV, DTI, loan amount, interest rate, state, property type, etc.
      - Servicing data: monthly loan status, delinquency, balance updates, and loss information.

      Each year has around 350,000 loans. Over 7 years, that’s about 2.4 million loans.

      I also added macroeconomic data from the FRED API, such as unemployment rate, home price index (HPI), and mortgage rates.
    `,
  },
  {
    title: "Default Definition",
    content: `
      A loan is treated as defaulted if:
      - It becomes 90+ days past due (delinquency status ≥ 3), or
      - It has a zero balance code related to loss (codes 02–09 like short sale, foreclosure, REO, etc.).

      If the zero balance code is 01, that means the borrower prepaid the loan voluntarily.
    `,
  },
  {
    title: "PD Model",
    content: `
      The PD model predicts the probability that a loan will default in the next 12 months.

      I observe each loan after 12 months of seasoning.
      The target is: does the loan default in the following 12 months?

      I train three models using time-based splits:
      - Logistic Regression (baseline and easy to interpret)
      - XGBoost (with calibration so probabilities are realistic)
      - Random Forest (ensemble-based model)

      The data split is:
      Train: 2010–2013  
      Validate: 2014  
      Test: 2015–2016  

      Important features include FICO, LTV, DTI, interest rate, state, occupancy, loan purpose,
      and macro variables like unemployment and HPI changes.
    `,
  },
  {
    title: "LGD Model",
    content: `
      The LGD model estimates how much money is lost when a loan defaults.

      LGD is calculated as:
      (Exposure − Recoveries + Expenses) / Exposure

      The value is limited between 0 and 1.

      I only use loans that were actually liquidated and had positive exposure.
      An XGBoost regression model is trained to predict LGD.

      I evaluate it using RMSE and check calibration by grouping predictions into deciles.
    `,
  },
  {
    title: "ECL Engine",
    content: `
      Expected Credit Loss (ECL) is calculated as:

      ECL = PD × LGD × EAD

      I follow IFRS 9 staging rules:
      - Stage 1: Low risk → 12-month ECL
      - Stage 2: Increased risk → Lifetime ECL
      - Stage 3: Defaulted/impaired → Lifetime ECL

      EAD is the current outstanding loan balance.
    `,
  },
  {
    title: "Stress Scenarios",
    content: `
    To test how the portfolio behaves under stress, I apply macroeconomic shocks:

      - Increase unemployment rate :  +N percentage points to baseline unemployment rate
      - Decrease home prices (HPI) :  +N% to baseline HPI year-over-year change (negative = price decline)
      - Increase mortgage rates : +N percentage points to 30-year fixed rate

      I define four scenarios:
      - Baseline (no change)
      - Mild stress
      - Severe stress
      - GFC-like stress

      These shocks are applied to macro variables before running predictions
    `,
  },
  {
    title: "Validation",
    content: `
      I use time-based splitting instead of random splits to avoid look-ahead bias.

      Train: 2010–2013  
      Validate: 2014  
      Test: 2015–2016  

      For PD: I check AUC, KS, Brier score, and calibration plots.

      For LGD: I evaluate RMSE, MAE, R², and decile calibration.
    `,
  },
];

export default function MethodologyPage() {
  return (
    <div className="space-y-8 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-200">Methodology</h1>
        <p className="text-sm text-gray-500 mt-1">
          How PD, LGD, and ECL are estimated — and how the stress testing works.
        </p>
      </div>

      {SECTIONS.map((s, i) => (
        <motion.div
          key={s.title}
          className="card"
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.06 }}
        >
          <h2 className="text-[#22d3ee] font-semibold text-sm mb-3 uppercase tracking-widest border-b border-[#1e3a5f] pb-2">
            {s.title}
          </h2>
          <div className="text-sm text-gray-400 leading-relaxed space-y-2 whitespace-pre-wrap">
            {s.content.trim().split("\n").map((line, j) => {
              if (line.startsWith("**") && line.endsWith("**")) {
                return <p key={j} className="font-semibold text-gray-200">{line.replace(/\*\*/g, "")}</p>;
              }
              if (line.startsWith("- ")) {
                return <p key={j} className="pl-4 text-gray-400">· {line.slice(2)}</p>;
              }
              if (line.startsWith("|")) {
                return (
                  <p key={j} className="font-mono text-xs text-gray-500 bg-[#080d1a] px-2 py-0.5 rounded">
                    {line}
                  </p>
                );
              }
              return <p key={j}>{line}</p>;
            })}
          </div>
        </motion.div>
      ))}
    </div>
  );
}
