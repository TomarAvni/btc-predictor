"""LSTM model for BTC time-series pattern recognition.

Captures sequential patterns in hourly price/volume data that XGBoost
cannot easily learn from tabular features alone.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for sequential BTC data."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_length: int = 168):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        x_seq = self.X[idx:idx + self.seq_length]
        y_val = self.y[idx + self.seq_length]
        return x_seq, y_val


class LSTMNetwork(nn.Module):
    """LSTM architecture for price prediction."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden).squeeze(-1)


class LSTMPredictor:
    """LSTM-based sequential price pattern predictor."""

    def __init__(self, model_dir: str = "data/models", seq_length: int = 168):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.seq_length = seq_length  # 168 hours = 7 days lookback
        self.models: dict[str, LSTMNetwork] = {}
        self.scalers: dict[str, dict] = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _select_lstm_features(self, df: pd.DataFrame) -> list[str]:
        """Select features suitable for LSTM sequential learning."""
        priority_features = [
            "close", "volume", "rsi_14", "macd_histogram",
            "bb_width", "atr_14", "volume_ratio",
            "momentum_24h", "volatility_24h",
        ]
        return [f for f in priority_features if f in df.columns]

    def _normalize(self, data: np.ndarray, horizon_label: str, fit: bool = False) -> np.ndarray:
        """Z-score normalization per feature."""
        if fit:
            mean = data.mean(axis=0)
            std = data.std(axis=0) + 1e-8
            self.scalers[horizon_label] = {"mean": mean, "std": std}
        else:
            mean = self.scalers[horizon_label]["mean"]
            std = self.scalers[horizon_label]["std"]
        return (data - mean) / std

    def train(
        self,
        features_df: pd.DataFrame,
        target: pd.Series,
        horizon_label: str = "24h",
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
    ) -> dict:
        """Train LSTM on sequential features.

        Uses the last 20% of data as validation (time-respecting split).
        """
        feature_cols = self._select_lstm_features(features_df)
        if not feature_cols:
            logger.warning("No suitable features for LSTM")
            return {"error": "no_features"}

        logger.info(f"Training LSTM for {horizon_label}, {len(features_df)} samples, "
                    f"{len(feature_cols)} sequential features, seq_len={self.seq_length}")

        X = features_df[feature_cols].values
        y = target.values

        # Normalize
        X = self._normalize(X, horizon_label, fit=True)

        # Time-respecting train/val split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_dataset = TimeSeriesDataset(X_train, y_train, self.seq_length)
        val_dataset = TimeSeriesDataset(X_val, y_val, self.seq_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = LSTMNetwork(input_size=len(feature_cols)).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                pred = model(x_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                    pred = model(x_batch)
                    val_loss += criterion(pred, y_batch).item()
                    val_correct += (torch.sign(pred) == torch.sign(y_batch)).sum().item()
                    val_total += len(y_batch)

            val_loss /= max(len(val_loader), 1)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.model_dir / f"lstm_{horizon_label}.pt")
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        # Load best model
        model.load_state_dict(torch.load(self.model_dir / f"lstm_{horizon_label}.pt", weights_only=True))
        self.models[horizon_label] = model

        # Save scaler
        with open(self.model_dir / f"lstm_scaler_{horizon_label}.pkl", "wb") as f:
            pickle.dump(self.scalers[horizon_label], f)

        direction_accuracy = val_correct / max(val_total, 1)
        metrics = {
            "horizon": horizon_label,
            "epochs_trained": epoch + 1,
            "best_val_loss": best_val_loss,
            "val_direction_accuracy": direction_accuracy,
            "seq_length": self.seq_length,
            "features_used": feature_cols,
        }
        logger.info(f"LSTM {horizon_label}: val direction accuracy = {direction_accuracy:.3f}")
        return metrics

    def predict(self, features_df: pd.DataFrame, horizon_label: str = "24h") -> dict:
        """Predict using the latest sequence of data."""
        model = self.models.get(horizon_label)
        if model is None:
            model = self._load_model(horizon_label)
        if model is None:
            return {"direction": "unknown", "magnitude": 0.0, "raw": 0.0}

        feature_cols = self._select_lstm_features(features_df)
        if not feature_cols or len(features_df) < self.seq_length:
            return {"direction": "unknown", "magnitude": 0.0, "raw": 0.0}

        X = features_df[feature_cols].tail(self.seq_length).values
        X = self._normalize(X, horizon_label, fit=False)

        model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device)
            pred = model(x_tensor).item()

        direction = "UP" if pred > 0 else "DOWN"
        return {
            "direction": direction,
            "magnitude": abs(pred),
            "raw": pred,
        }

    def _load_model(self, horizon_label: str) -> LSTMNetwork | None:
        model_path = self.model_dir / f"lstm_{horizon_label}.pt"
        scaler_path = self.model_dir / f"lstm_scaler_{horizon_label}.pkl"

        if not model_path.exists():
            return None

        # Need to know input size -- use saved scaler to determine
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scalers[horizon_label] = pickle.load(f)
            input_size = len(self.scalers[horizon_label]["mean"])
        else:
            return None

        model = LSTMNetwork(input_size=input_size).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.models[horizon_label] = model
        return model
