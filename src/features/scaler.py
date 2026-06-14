"""Feature normalization and scaling utilities."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


class FeatureScaler:
    """Handles feature normalization with persistence.

    Uses RobustScaler (median/IQR) which is more resistant to outliers
    than StandardScaler -- important for crypto data with extreme moves.
    """

    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler: RobustScaler | None = None
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeatureScaler":
        """Fit scaler on training data."""
        numeric = df.select_dtypes(include=[np.number])
        self.feature_names = numeric.columns.tolist()
        self.scaler = RobustScaler()
        self.scaler.fit(numeric.values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        numeric = df.select_dtypes(include=[np.number])
        # Handle columns that weren't in training
        common_cols = [c for c in self.feature_names if c in numeric.columns]
        missing_cols = [c for c in self.feature_names if c not in numeric.columns]

        result = numeric[common_cols].copy()
        for col in missing_cols:
            result[col] = 0.0
        result = result[self.feature_names]

        scaled_values = self.scaler.transform(result.values)
        return pd.DataFrame(scaled_values, index=result.index, columns=self.feature_names)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df)
        return self.transform(df)

    def save(self, name: str = "feature_scaler") -> None:
        path = self.model_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "feature_names": self.feature_names}, f)

    def load(self, name: str = "feature_scaler") -> "FeatureScaler":
        path = self.model_dir / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.scaler = data["scaler"]
                self.feature_names = data["feature_names"]
        return self
