"""
Exploratory Data Analysis — Mortgage Credit Risk Lab
Run as a plain Python script or convert to Jupyter with jupytext.

Usage:
  py -3 notebooks/00_eda.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import pandas as pd
import numpy as np

from src.config.settings import DATA_PROCESSED, DATA_INTERIM, FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="dark", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#0f172a",
    "axes.edgecolor":   "#1e3a5f",
    "text.color":       "#94a3b8",
    "axes.labelcolor":  "#94a3b8",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "grid.color":       "#1e3a5f",
    "grid.alpha":       0.5,
})
CYAN  = "#22d3ee"
AMBER = "#f59e0b"
RED   = "#ef4444"

print("Loading origination data ...")
orig = pd.read_parquet(DATA_INTERIM / "orig_all.parquet")
print(f"  {len(orig):,} loans across {orig['VINTAGE_YEAR'].nunique()} vintages")

outcomes = pd.read_parquet(DATA_PROCESSED / "loan_outcomes.parquet")

# ── 1. FICO distribution ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for yr, grp in orig.groupby("VINTAGE_YEAR"):
    ax.hist(grp["CREDIT_SCORE"].dropna(), bins=50, alpha=0.5, label=str(yr))
ax.set_title("FICO Score Distribution by Vintage", color="white", fontsize=14)
ax.set_xlabel("FICO Score")
ax.legend(title="Vintage", fontsize=8)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fico_distribution.png", dpi=150)
plt.close()
print("  Saved: fico_distribution.png")

# ── 2. LTV distribution ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(orig["ORIGINAL_LTV"].dropna(), bins=40, color=CYAN, alpha=0.8, edgecolor="none")
ax.axvline(80, color=AMBER, linestyle="--", label="80% (PMI threshold)")
ax.axvline(orig["ORIGINAL_LTV"].median(), color=RED, linestyle="--",
           label=f"Median {orig['ORIGINAL_LTV'].median():.0f}%")
ax.set_title("Original LTV Distribution", color="white", fontsize=14)
ax.set_xlabel("LTV (%)")
ax.legend()
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "ltv_distribution.png", dpi=150)
plt.close()
print("  Saved: ltv_distribution.png")

# ── 3. Default rate by vintage ───────────────────────────────────────────────
dr = outcomes.groupby("VINTAGE_YEAR")["defaulted"].mean().reset_index()
dr["default_rate_pct"] = dr["defaulted"] * 100
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(dr["VINTAGE_YEAR"].astype(str), dr["default_rate_pct"],
              color=CYAN, alpha=0.9, width=0.6, edgecolor="none")
for bar, val in zip(bars, dr["default_rate_pct"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.1f}%", ha="center", fontsize=9, color="white")
ax.set_title("Default Rate by Origination Vintage", color="white", fontsize=14)
ax.set_xlabel("Vintage Year")
ax.set_ylabel("Default Rate (%)")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}%"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "default_rate_by_vintage.png", dpi=150)
plt.close()
print("  Saved: default_rate_by_vintage.png")

# ── 4. Default rate by FICO band ─────────────────────────────────────────────
outcomes["FICO_BAND"] = pd.cut(
    outcomes["CREDIT_SCORE"].fillna(650),
    bins=[0, 620, 660, 700, 740, 800, 900],
    labels=["<620", "620–659", "660–699", "700–739", "740–799", "800+"],
)
fico_dr = outcomes.groupby("FICO_BAND", observed=True)["defaulted"].mean() * 100
fig, ax = plt.subplots(figsize=(10, 5))
colors = [RED if v > 5 else AMBER if v > 2 else CYAN for v in fico_dr.values]
bars = ax.bar(fico_dr.index.astype(str), fico_dr.values, color=colors, alpha=0.9, width=0.6)
for bar, val in zip(bars, fico_dr.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{val:.1f}%", ha="center", fontsize=9, color="white")
ax.set_title("Default Rate by FICO Band", color="white", fontsize=14)
ax.set_xlabel("FICO Band")
ax.set_ylabel("Default Rate (%)")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "default_rate_by_fico.png", dpi=150)
plt.close()
print("  Saved: default_rate_by_fico.png")

# ── 5. LGD distribution ──────────────────────────────────────────────────────
lgd_obs = outcomes["lgd_observed"].dropna()
if len(lgd_obs) > 0:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lgd_obs, bins=40, color=RED, alpha=0.8, edgecolor="none")
    ax.axvline(lgd_obs.mean(), color=AMBER, linestyle="--",
               label=f"Mean LGD = {lgd_obs.mean():.3f}")
    ax.axvline(lgd_obs.median(), color=CYAN, linestyle="--",
               label=f"Median LGD = {lgd_obs.median():.3f}")
    ax.set_title("Observed LGD Distribution (Liquidated Defaulters)", color="white", fontsize=14)
    ax.set_xlabel("LGD")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "lgd_distribution.png", dpi=150)
    plt.close()
    print("  Saved: lgd_distribution.png")

# ── 6. Top states by loan count ───────────────────────────────────────────────
state_counts = orig.groupby("PROPERTY_STATE").size().sort_values(ascending=False).head(15)
fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(state_counts.index, state_counts.values, color=CYAN, alpha=0.85, width=0.7)
ax.set_title("Top 15 States by Loan Count", color="white", fontsize=14)
ax.set_xlabel("State")
ax.set_ylabel("Loan Count")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
plt.tight_layout()
plt.savefig(FIGURES_DIR / "top_states_loan_count.png", dpi=150)
plt.close()
print("  Saved: top_states_loan_count.png")

print("\nEDA complete. All figures saved to:", FIGURES_DIR)

# ── Summary statistics ─────────────────────────────────────────────────────────
print("\n-- Portfolio Summary -------------------------------------------")
print(f"Total loans:       {len(orig):>12,}")
print(f"Total UPB:         ${orig['ORIGINAL_UPB'].sum():>12,.0f}")
print(f"Avg FICO:          {orig['CREDIT_SCORE'].mean():>12.0f}")
print(f"Avg LTV:           {orig['ORIGINAL_LTV'].mean():>12.1f}%")
print(f"Avg Rate:          {orig['ORIGINAL_INTEREST_RATE'].mean():>12.2f}%")
print(f"Default rate:      {outcomes['defaulted'].mean():>12.2%}")
print(f"Prepay rate:       {outcomes['prepaid'].mean():>12.2%}")
if len(lgd_obs) > 0:
    print(f"Mean LGD:          {lgd_obs.mean():>12.4f}")
print(f"FRM share:         {(orig['PRODUCT_TYPE'] == 'FRM').mean():>12.1%}")
print(f"Purchase share:    {(orig['LOAN_PURPOSE'] == 'C').mean():>12.1%}")
