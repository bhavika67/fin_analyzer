# tests/test_eda.py
import pytest
import pandas as pd
import numpy as np
from eda import EDAAnalyzer, RegressionAnalyzer, AnomalyDetector

# ── EDA Analyzer ─────────────────────────────────────────────────

def test_eda_returns_insights(sample_prices_df):
    result = EDAAnalyzer().analyze(sample_prices_df, target_col="close")
    assert len(result.insights) > 0
    assert "50 rows" in result.insights[0]

def test_eda_summary_has_all_columns(sample_prices_df):
    result = EDAAnalyzer().analyze(sample_prices_df)
    for col in sample_prices_df.select_dtypes("number").columns:
        assert col in result.summary_stats

def test_eda_detects_correlations(sample_prices_df):
    result = EDAAnalyzer().analyze(sample_prices_df)
    assert any("close" in k for k in result.correlations)

def test_eda_missing_values():
    df = pd.DataFrame({"a": [1, None, 3, 4, 5], "b": [1, 2, 3, 4, 5]})
    result = EDAAnalyzer().analyze(df)
    assert "a" in result.missing_values
    assert result.missing_values["a"]["count"] == 1

def test_eda_trends_direction(sample_pl_df):
    result = EDAAnalyzer().analyze(sample_pl_df, target_col="revenue")
    assert result.trends["revenue"]["direction"] in ("upward", "downward")

# ── Regression ───────────────────────────────────────────────────

def test_regression_r2_reasonable(sample_prices_df):
    reg = RegressionAnalyzer().fit(sample_prices_df, target_col="close")
    assert reg.r2 > 0.5
    assert reg.mae > 0
    assert "close" in reg.interpretation

def test_regression_too_few_rows():
    df = pd.DataFrame({"x": range(5), "y": range(5)})
    with pytest.raises(ValueError, match="Too few rows"):
        RegressionAnalyzer().fit(df, target_col="y")

def test_regression_missing_target(sample_prices_df):
    with pytest.raises(ValueError, match="not found"):
        RegressionAnalyzer().fit(sample_prices_df, target_col="nonexistent")

def test_regression_coefficients_present(sample_pl_df):
    reg = RegressionAnalyzer().fit(sample_pl_df, target_col="net_profit")
    assert "revenue" in reg.coefficients
    assert reg.r2 > 0.8

# ── Anomaly Detector ─────────────────────────────────────────────

def test_anomaly_detects_injected_outlier(sample_pl_df):
    results = AnomalyDetector().detect(sample_pl_df, method="iqr")
    cols_with_anomalies = [r.column for r in results]
    # Anomaly on index 10 shows up in net_margin (ratio amplifies the drop)
    # or net_profit depending on the spread — either is a correct detection
    assert len(cols_with_anomalies) > 0, "Expected at least one anomaly column"
    assert any(col in cols_with_anomalies for col in ["net_profit", "net_margin"]), \
        f"Expected net_profit or net_margin, got: {cols_with_anomalies}"

def test_anomaly_detects_injected_outlier_zscore(sample_pl_df):
    results = AnomalyDetector().detect(sample_pl_df, method="zscore")
    cols_with_anomalies = [r.column for r in results]
    assert any(col in cols_with_anomalies for col in ["net_profit", "net_margin"]), \
        f"Expected anomaly via z-score, got: {cols_with_anomalies}"

def test_anomaly_zscore(sample_pl_df):
    results = AnomalyDetector().detect(sample_pl_df, method="zscore")
    assert isinstance(results, list)

def test_anomaly_clean_data():
    df = pd.DataFrame({"x": [10, 11, 10, 12, 11, 10, 11, 12, 10, 11]})
    results = AnomalyDetector().detect(df, method="iqr")
    assert results == []

def test_anomaly_result_fields(sample_pl_df):
    results = AnomalyDetector().detect(sample_pl_df, method="iqr")
    for r in results:
        assert r.anomaly_count == len(r.anomaly_values)
        assert len(r.summary) > 0