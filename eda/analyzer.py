# eda/analyzer.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from loguru import logger


@dataclass
class EDAResult:
    summary_stats: dict
    missing_values: dict
    correlations: dict
    trends: dict
    insights: list[str]


class EDAAnalyzer:
    """Exploratory Data Analysis on financial DataFrames."""

    def analyze(self, df: pd.DataFrame, target_col: str | None = None) -> EDAResult:
        logger.info(f"Running EDA — shape: {df.shape}")
        numeric = df.select_dtypes(include=[np.number])

        return EDAResult(
            summary_stats  = self._summary(numeric),
            missing_values = self._missing(df),
            correlations   = self._correlations(numeric),
            trends         = self._trends(numeric, target_col),
            insights       = self._insights(df, numeric, target_col),
        )

    # ── helpers ───────────────────────────────────────────────────────────

    def _summary(self, numeric: pd.DataFrame) -> dict:
        desc = numeric.describe().round(4).to_dict()
        for col in numeric.columns:
            desc[col]["skewness"] = round(float(numeric[col].skew()), 4)
            desc[col]["kurtosis"] = round(float(numeric[col].kurt()), 4)
        return desc

    def _missing(self, df: pd.DataFrame) -> dict:
        counts = df.isnull().sum()
        pct    = (counts / len(df) * 100).round(2)
        return {
            col: {"count": int(counts[col]), "pct": float(pct[col])}
            for col in df.columns if counts[col] > 0
        }

    def _correlations(self, numeric: pd.DataFrame) -> dict:
        if numeric.shape[1] < 2:
            return {}
        corr = numeric.corr().round(3)
        seen, pairs = set(), {}
        for col in corr.columns:
            for idx in corr.index:
                if col == idx:
                    continue
                key = tuple(sorted([col, idx]))
                if key in seen:
                    continue
                seen.add(key)
                val = float(corr.loc[idx, col])
                if abs(val) > 0.5:
                    pairs[f"{key[0]} vs {key[1]}"] = val
        return dict(sorted(pairs.items(), key=lambda x: abs(x[1]), reverse=True))

    def _trends(self, numeric: pd.DataFrame, target_col: str | None) -> dict:
        cols = ([target_col] if target_col and target_col in numeric.columns
                else numeric.columns[:5])  # cap at 5 cols
        trends = {}
        for col in cols:
            s = numeric[col].dropna()
            if len(s) < 3:
                continue
            pct_chg = s.pct_change().dropna()
            trends[col] = {
                "mean":           round(float(s.mean()), 4),
                "std":            round(float(s.std()), 4),
                "min":            round(float(s.min()), 4),
                "max":            round(float(s.max()), 4),
                "avg_pct_change": round(float(pct_chg.mean()) * 100, 3),
                "direction":      "upward" if pct_chg.mean() > 0 else "downward",
                "volatility":     round(float(pct_chg.std()) * 100, 3),
            }
        return trends

    def _insights(self, df, numeric, target_col) -> list[str]:
        insights = []
        insights.append(f"Dataset: {len(df)} rows × {len(df.columns)} columns.")

        miss_pct = df.isnull().sum().sum() / df.size * 100
        if miss_pct > 0:
            insights.append(f"{miss_pct:.1f}% of values are missing overall.")

        for col in numeric.columns:
            skew = float(numeric[col].skew())
            if abs(skew) > 1:
                side = "right (positive)" if skew > 0 else "left (negative)"
                insights.append(
                    f"'{col}' is heavily {side}-skewed (skew={skew:.2f}) "
                    f"— consider log-transform before regression."
                )

        if target_col and target_col in numeric.columns:
            corr = numeric.corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
            for feat, val in corr.head(3).items():
                if val > 0.4:
                    insights.append(
                        f"'{feat}' has a strong correlation ({val:.2f}) with '{target_col}'."
                    )

        # detect if any column looks like it has a structural break
        for col in numeric.columns[:5]:
            s = numeric[col].dropna()
            if len(s) < 8:
                continue
            mid = len(s) // 2
            mean_first = s.iloc[:mid].mean()
            mean_second = s.iloc[mid:].mean()
            if mean_first != 0 and abs(mean_second - mean_first) / abs(mean_first) > 0.25:
                insights.append(
                    f"'{col}' shows a significant shift between first and second half "
                    f"({mean_first:.2f} → {mean_second:.2f}) — possible structural break."
                )

        return insights