# scripts/generate_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)
Path("data/raw").mkdir(parents=True, exist_ok=True)

# ── 1. Quarterly P&L (5 years) ─────────────────────────────────────────────
quarters = pd.date_range("2020-01-01", periods=20, freq="QE")
revenue = 400 + np.cumsum(np.random.normal(15, 8, 20))          # growing trend
cogs    = revenue * np.random.uniform(0.52, 0.58, 20)           # 52–58% of revenue
opex    = revenue * np.random.uniform(0.18, 0.24, 20)
ebitda  = revenue - cogs - opex
tax     = ebitda * 0.22
net_profit = ebitda - tax

# Inject one bad quarter (Q3 2022 = index 10)
revenue[10]    *= 0.78
net_profit[10] *= 0.45

pl = pd.DataFrame({
    "quarter":    quarters.strftime("%Y-Q%q"),
    "revenue":    revenue.round(2),
    "cogs":       cogs.round(2),
    "opex":       opex.round(2),
    "ebitda":     ebitda.round(2),
    "tax":        tax.round(2),
    "net_profit": net_profit.round(2),
    "net_margin": (net_profit / revenue * 100).round(2),
})
pl.to_csv("data/raw/quarterly_pl.csv", index=False)
print(f" quarterly_pl.csv — {len(pl)} rows")

# ── 2. Monthly Revenue by Segment (3 years) ───────────────────────────────
months = pd.date_range("2022-01-01", periods=36, freq="ME")
segments = ["Enterprise", "SMB", "Consumer"]
rows = []
base = {"Enterprise": 180, "SMB": 90, "Consumer": 60}
growth = {"Enterprise": 1.018, "SMB": 1.012, "Consumer": 1.008}

for i, month in enumerate(months):
    for seg in segments:
        val = base[seg] * (growth[seg] ** i) + np.random.normal(0, base[seg] * 0.03)
        rows.append({
            "month": month.strftime("%Y-%m"),
            "segment": seg,
            "revenue": round(max(val, 0), 2),
            "customers": int(val / np.random.uniform(1.8, 2.2)),
            "churn_rate": round(np.random.uniform(0.01, 0.04), 4),
        })

segment_rev = pd.DataFrame(rows)
segment_rev.to_csv("data/raw/segment_revenue.csv", index=False)
print(f" segment_revenue.csv — {len(segment_rev)} rows")

# ── 3. Cost & Headcount (monthly, 3 years) ────────────────────────────────
headcount = np.round(120 + np.cumsum(np.random.randint(1, 5, 36))).astype(int)
salaries  = headcount * np.random.uniform(7.8, 8.5, 36)
infra     = 40 + np.cumsum(np.random.normal(0.8, 0.3, 36))
marketing = 30 + np.random.normal(0, 4, 36)
rd        = 55 + np.cumsum(np.random.normal(0.5, 0.2, 36))

costs = pd.DataFrame({
    "month":        months.strftime("%Y-%m"),
    "headcount":    headcount,
    "salary_cost":  salaries.round(2),
    "infra_cost":   infra.round(2),
    "marketing":    marketing.round(2),
    "rd_spend":     rd.round(2),
    "total_cost":   (salaries + infra + marketing + rd).round(2),
})
costs.to_csv("data/raw/cost_headcount.csv", index=False)
print(f" cost_headcount.csv — {len(costs)} rows")

# ── 4. Narrative report as plain text (for RAG) ───────────────────────────
report = """
ANNUAL FINANCIAL REPORT — FY 2023

EXECUTIVE SUMMARY
Total revenue for FY 2023 reached $2,847M, representing 14.2% year-over-year growth.
EBITDA margin improved to 22.4%, up from 19.8'%' in FY 2022.
Net profit was $412M with a net margin of 14.5%.

REVENUE BREAKDOWN
Enterprise segment contributed $1,520M (53'%' of total), growing 18% YoY driven by
5 new Fortune 500 contracts signed in Q2 and Q4.
SMB segment generated $890M (31%), up 11% YoY. Churn remained elevated at 3.2%.
Consumer segment delivered $437M (16%), growing 8% YoY.

Q3 2022 ANOMALY
Revenue dipped significantly in Q3 2022 due to a supply chain disruption that
delayed enterprise deal closures. Three major contracts worth $68M were pushed
into Q4. This was a one-time event; the pipeline recovered fully by Q4 2022.

COST STRUCTURE
Total operating expenses were $2,210M. Headcount grew from 120 to 245 employees.
R&D investment increased 28% to $198M as the company accelerated product development.
Infrastructure costs rose 19'%' due to cloud migration completed in Q2 2023.

OUTLOOK FY 2024
Management guides revenue of $3,200M–$3,350M (12–18'%' growth).
Key risks: macroeconomic slowdown affecting SMB segment, rising cloud costs.
Key opportunities: AI product line launch in Q2, expansion into APAC markets.
""".strip()

Path("data/raw/annual_report_fy2023.txt").write_text(report)
print("annual_report_fy2023.txt — narrative report for RAG")

print("\n All files saved to data/raw/")