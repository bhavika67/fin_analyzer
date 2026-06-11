# tests/conftest.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture
def sample_prices_df():
    """Realistic price DataFrame — 50 rows, enough for regression."""
    np.random.seed(42)
    close = 150 + np.cumsum(np.random.normal(0.3, 2, 50))
    return pd.DataFrame({
        "close":        close.round(2),
        "volume":       np.random.randint(50_000_000, 120_000_000, 50),
        "high":         (close + np.random.uniform(1, 4, 50)).round(2),
        "low":          (close - np.random.uniform(1, 4, 50)).round(2),
        "daily_return": np.random.normal(0.001, 0.02, 50).round(6),
    })

@pytest.fixture
def sample_pl_df():
    """Quarterly P&L — 20 rows with one injected anomaly."""
    np.random.seed(0)
    revenue = 400 + np.cumsum(np.random.normal(15, 8, 20))
    cost    = revenue * 0.55
    profit  = revenue - cost
    profit[10] *= 0.45   # injected anomaly
    return pd.DataFrame({
        "revenue":    revenue.round(2),
        "cost":       cost.round(2),
        "net_profit": profit.round(2),
        "net_margin": (profit / revenue * 100).round(2),
    })

@pytest.fixture
def sample_txt_file(tmp_path):
    """A small text file for parser testing."""
    f = tmp_path / "report.txt"
    f.write_text("Q3 Revenue: $500M. Net profit margin improved to 22%.")
    return f

@pytest.fixture
def sample_csv_file(tmp_path):
    """A small CSV file for parser testing."""
    f = tmp_path / "data.csv"
    f.write_text("revenue,cost,profit\n100,60,40\n200,110,90\n150,85,65\n")
    return f