# eda/anomaly.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
from loguru import logger


@dataclass
class AnomalyResult:
    column:         str
    method:         str
    anomaly_indices: list
    anomaly_values: list[float]
    anomaly_count:  int
    summary:        str


class AnomalyDetector:

    def detect(self, df: pd.DataFrame, method: str = "iqr") -> list[AnomalyResult]:
        numeric = df.select_dtypes(include=[np.number])
        results = []
        for col in numeric.columns:
            s = numeric[col].dropna()
            if len(s) < 5:
                continue
            result = self._iqr(s, col) if method == "iqr" else self._zscore(s, col)
            if result.anomaly_count > 0:
                logger.info(f"Anomaly in '{col}': {result.anomaly_count} outlier(s)")
                results.append(result)
        return results

    def _iqr(self, s: pd.Series, col: str) -> AnomalyResult:
        Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        mask = (s < lo) | (s > hi)
        out  = s[mask]
        return AnomalyResult(
            column=col, method="IQR",
            anomaly_indices=out.index.tolist(),
            anomaly_values=[round(v, 4) for v in out.values],
            anomaly_count=len(out),
            summary=f"{len(out)} outlier(s) in '{col}' outside [{lo:.2f}, {hi:.2f}].",
        )

    def _zscore(self, s: pd.Series, col: str) -> AnomalyResult:
        z   = np.abs(stats.zscore(s))
        out = s[z > 3]
        return AnomalyResult(
            column=col, method="Z-score",
            anomaly_indices=out.index.tolist(),
            anomaly_values=[round(v, 4) for v in out.values],
            anomaly_count=len(out),
            summary=f"{len(out)} outlier(s) in '{col}' with |z-score| > 3.",
        )