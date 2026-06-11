# eda/__init__.py
from .analyzer import EDAAnalyzer
from .regression import RegressionAnalyzer
from .anomaly import AnomalyDetector

__all__ = ["EDAAnalyzer", "RegressionAnalyzer", "AnomalyDetector"]