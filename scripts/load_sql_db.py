# scripts/load_sql_db.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import sqlite3
import pandas as pd
from loguru import logger

DB_PATH = ROOT / "data" / "processed" / "fin.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# Tables built from financial CSVs that have clean tabular structure
TABLES = {
    "quarterly_pl":     "data/raw/quarterly_pl.csv",
    "segment_revenue":  "data/raw/segment_revenue.csv",
    "cost_headcount":   "data/raw/cost_headcount.csv",
    "aapl_financials":  "data/raw/AAPL_financials.csv",
    "googl_financials": "data/raw/GOOGL_financials.csv",
    "msft_financials":  "data/raw/MSFT_financials.csv",
    "tsla_financials":  "data/raw/TSLA_financials.csv",
    "infy_financials":  "data/raw/INFY.NS_financials.csv",
    "tcs_financials":   "data/raw/TCS.NS_financials.csv",
}


def main():
    conn = sqlite3.connect(DB_PATH)

    for table, csv_path in TABLES.items():
        path = ROOT / csv_path
        if not path.exists():
            logger.warning(f"Skipping {table}: {csv_path} not found")
            continue

        # Financial statement CSVs have the date as the index column (unnamed)
        df = pd.read_csv(path)
        if df.columns[0].startswith("Unnamed") or df.columns[0] == "":
            df = df.rename(columns={df.columns[0]: "period"})

        df.to_sql(table, conn, if_exists="replace", index=False)
        logger.info(f"Loaded {table}: {len(df)} rows, columns: {list(df.columns)}")

    conn.close()
    logger.info(f"Database saved to {DB_PATH}")


if __name__ == "__main__":
    main()