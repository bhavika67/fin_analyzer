# scripts/fetch_real_data.py
"""
Fetch real financial data for any ticker from Yahoo Finance.
Usage: python scripts/fetch_real_data.py --ticker MSFT --years 5
"""
import argparse
import yfinance as yf
import pandas as pd
from pathlib import Path

Path("data/raw").mkdir(parents=True, exist_ok=True)

def fetch(ticker: str, years: int = 5):
    print(f"\n📡 Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)

    # ── 1. Historical price + volume ──────────────────────────────────────
    hist = stock.history(period=f"{years}y")[["Close", "Volume", "High", "Low"]]
    hist.index = hist.index.strftime("%Y-%m-%d")
    hist.columns = ["close", "volume", "high", "low"]
    hist["daily_return"] = hist["close"].pct_change().round(6)
    hist["30d_ma"] = hist["close"].rolling(30).mean().round(2)
    hist.to_csv(f"data/raw/{ticker}_prices.csv")
    print(f"{ticker}_prices.csv — {len(hist)} trading days")

    # ── 2. Quarterly financials ────────────────────────────────────────────
    try:
        income = stock.quarterly_income_stmt
        if income is not None and not income.empty:
            # Transpose so each row = one quarter
            df = income.T[["Total Revenue", "Gross Profit",
                            "Operating Income", "Net Income"]].copy()
            df.index = df.index.strftime("%Y-%m-%d")
            df.columns = ["revenue", "gross_profit", "operating_income", "net_income"]
            df = df.apply(pd.to_numeric, errors="coerce").dropna()
            df["gross_margin"] = (df["gross_profit"] / df["revenue"] * 100).round(2)
            df["net_margin"]   = (df["net_income"]   / df["revenue"] * 100).round(2)
            df.to_csv(f"data/raw/{ticker}_financials.csv")
            print(f"{ticker}_financials.csv — {len(df)} quarters")
    except Exception as e:
        print(f"Financials unavailable: {e}")

    # ── 3. Key metrics as a text summary (for RAG ingestion) ──────────────
    try:
        info = stock.info
        keys = ["longName", "sector", "industry", "marketCap", "trailingPE",
                "forwardPE", "revenueGrowth", "grossMargins", "operatingMargins",
                "returnOnEquity", "debtToEquity", "currentRatio", "beta",
                "52WeekChange", "recommendationKey", "targetMeanPrice"]
        lines = [f"COMPANY PROFILE: {ticker}\n"]
        for k in keys:
            val = info.get(k)
            if val is not None:
                lines.append(f"{k}: {val}")
        summary = "\n".join(lines)
        Path(f"data/raw/{ticker}_profile.txt").write_text(summary)
        print(f"{ticker}_profile.txt — company summary for RAG")
    except Exception as e:
        print(f"Profile unavailable: {e}")

    print(f"\nDone. Files saved to data/raw/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="TCS.NS", help="Stock ticker e.g. MSFT, GOOGL, TSLA")
    parser.add_argument("--years",  type=int, default=5)
    args = parser.parse_args()
    fetch(args.ticker, args.years)