# eda/regression.py
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from loguru import logger


@dataclass
class RegressionResult:
    target:         str
    features:       list[str]
    r2:             float
    mae:            float
    rmse:           float
    coefficients:   dict[str, float]
    intercept:      float
    interpretation: str


class RegressionAnalyzer:

    def fit(self, df: pd.DataFrame, target_col: str) -> RegressionResult:
        numeric = df.select_dtypes(include=[np.number]).dropna()
        if target_col not in numeric.columns:
            raise ValueError(f"Target '{target_col}' not found or not numeric.")

        features = [c for c in numeric.columns if c != target_col]
        if not features:
            raise ValueError("No numeric feature columns available.")

        X = numeric[features].values
        y = numeric[target_col].values

        # ── guard: need at least 10 rows for a meaningful split ──────────
        if len(X) < 10:
            raise ValueError(
                f"Too few rows ({len(X)}) for regression — need at least 10. "
                f"Try quarterly_pl.csv or AAPL_prices.csv instead."
            )

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        model  = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = round(float(r2_score(y_test, y_pred)), 4)
        mae  = round(float(mean_absolute_error(y_test, y_pred)), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)

        coefs = {f: round(float(c), 4) for f, c in zip(features, model.coef_)}
        top   = max(coefs, key=lambda k: abs(coefs[k]))

        logger.info(f"Regression '{target_col}': R²={r2}, RMSE={rmse}")
        return RegressionResult(
            target=target_col, features=features,
            r2=r2, mae=mae, rmse=rmse,
            coefficients=coefs,
            intercept=round(float(model.intercept_), 4),
            interpretation=(
                f"Model explains {r2*100:.1f}% of variance in '{target_col}'. "
                f"'{top}' is the strongest predictor (coef={coefs[top]})."
            ),
        )