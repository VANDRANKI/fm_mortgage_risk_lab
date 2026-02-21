# Mortgage Credit Risk & Stress Testing Lab

**Live Demo:** *(add URLs after deployment)*

---

## What is this project?

Imagine a bank gives out thousands of home loans. Some people will miss payments and default. The bank needs to know **how much money it might lose** — this is called the **Expected Credit Loss (ECL)**.

This project builds a full system that:
1. Takes **real mortgage data** from Freddie Mac (350,000 loans, 2010–2016)
2. Trains **machine learning models** to predict who will default and how much will be lost
3. Lets you **simulate bad economic scenarios** — what if unemployment doubles? what if home prices crash by 30%?
4. Shows everything on a **live website dashboard** with charts and sliders

---

## How it works (simply)

```
Freddie Mac loan data
      ↓
Clean it + add real economic data from the US Federal Reserve
      ↓
Train two ML models:
  - PD model  → probability that a loan defaults  (AUC = 0.85)
  - LGD model → how much money is lost if it does (R² = 0.30)
      ↓
Combine them:  ECL = PD × LGD × Loan Balance
      ↓
FastAPI backend serves results as a JSON API
      ↓
Next.js website shows it all in real-time with interactive sliders
```

---

## The 4 stress scenarios

| Scenario | Unemployment rise | Home prices | ECL result |
|----------|-------------------|-------------|------------|
| Baseline | No change | No change | $60.2M |
| Mild | +2 pp | -5% | $63.6M |
| Severe | +5 pp | -20% | $61.1M |
| GFC-Like (2008-style) | +8 pp | -33% | $60.7M |

---

## Run it yourself

### You need
- Python 3.10+
- Node.js 18+
- Freddie Mac loan data files (free to download from their website)
- A free FRED API key (from the US Federal Reserve website)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/VANDRANKI/fm_mortgage_risk_lab.git
cd fm_mortgage_risk_lab

# 2. Install Python packages
pip install -r requirements.txt

# 3. Add your FRED API key
cp .env.example .env
# open .env and paste your FRED_API_KEY

# 4. Place Freddie Mac .txt files inside data/raw/

# 5. Run the full pipeline (builds everything, ~30 min)
py -3 run_pipeline.py

# 6. Start the API
py -3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 7. Start the website (in a second terminal)
cd fm-risk-lab-web
npm install
node node_modules/next/dist/bin/next dev --port 3000
```

Open http://localhost:3000 — you're live!

---

## Website pages

| Page | What it shows |
|------|---------------|
| **Portfolio** | Overall risk summary — total ECL, PD, LGD, loan count |
| **Risk Lab** | Move sliders to shock the economy and see ECL change live |
| **Loan Explorer** | Enter a single loan's details and get its risk score |
| **Methodology** | How the models work, what data was used |

---

## Tech used

| What | Tools |
|------|-------|
| Data processing | Python, pandas, PyArrow |
| Machine learning | XGBoost, scikit-learn |
| Macro data | FRED API (US Federal Reserve) |
| Backend API | FastAPI, uvicorn |
| Website | Next.js 16, React 19, TypeScript, Tailwind CSS |
| Charts | Recharts, Framer Motion |
| Hosting | Render (backend) + Vercel (frontend) |

---
> Built by **Prabhu ** — [LinkedIn](https://www.linkedin.com/in/vandranki-prabhu-kiran-4b75b4215/) · [GitHub](https://github.com/VANDRANKI) | © 2026 
