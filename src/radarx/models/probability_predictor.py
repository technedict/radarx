"""
Probability Predictor

Ensemble model combining gradient boosting and temporal neural networks
for multi-horizon probability predictions.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    import xgboost as xgb

    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False
    logger.warning("XGBoost/LightGBM not available. Using fallback models.")

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available. Temporal model disabled.")


class TemporalAttentionNetwork(nn.Module if HAS_TORCH else object):
    """
    Temporal neural network with attention mechanism for time-series patterns.
    Captures social-price lead-lag relationships and momentum patterns.
    """

    def __init__(
        self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 15, num_heads: int = 4
    ):
        if not HAS_TORCH:
            return
        super().__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        x = self.embedding(x)
        x, _ = self.attention(x, x, x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last time step
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class ProbabilityPredictor:
    """
    Hybrid ensemble model for probability prediction.

    Combines:
    - XGBoost/LightGBM for feature-based prediction and explainability
    - Temporal neural network for time-series patterns
    - Weighted ensemble for final predictions

    Predicts probability heatmaps for multiple multipliers (2x, 5x, 10x, 20x, 50x)
    across multiple horizons (24h, 7d, 30d).
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize probability predictor.

        Args:
            model_dir: Directory to load/save models
        """
        self.model_dir = model_dir or Path("models/probability")
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Model components
        self.xgb_models = {}  # {(horizon, multiplier): model}
        self.lgb_models = {}
        self.temporal_model = None
        self.ensemble_weights = {}

        # Feature names
        self.feature_names = []
        self.temporal_features = []

        # Metadata
        self.version = "1.0.0"
        self.trained_at = None
        self.is_trained = False

    def train(
        self,
        X_features: np.ndarray,
        X_temporal: Optional[np.ndarray],
        y_labels: Dict[Tuple[str, str], np.ndarray],
        feature_names: List[str],
        horizons: List[str] = ["24h", "7d", "30d"],
        multipliers: List[str] = ["2x", "5x", "10x", "20x", "50x"],
    ):
        """
        Train ensemble models.

        Args:
            X_features: Feature matrix (n_samples, n_features)
            X_temporal: Temporal sequences (n_samples, seq_len, temporal_features)
            y_labels: Labels dict {(horizon, multiplier): binary_labels}
            feature_names: List of feature names
            horizons: Time horizons to predict
            multipliers: Multiplier targets
        """
        logger.info("Training probability prediction models...")
        self.feature_names = feature_names
        self.is_trained = False

        # Train XGBoost models for each (horizon, multiplier) combination
        for horizon in horizons:
            for multiplier in multipliers:
                key = (horizon, multiplier)
                if key not in y_labels:
                    logger.warning(f"No labels for {key}, skipping")
                    continue

                y = y_labels[key]

                # XGBoost
                if HAS_BOOSTING:
                    logger.info(f"Training XGBoost for {horizon} {multiplier}...")
                    xgb_model = xgb.XGBClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    )
                    xgb_model.fit(X_features, y)
                    self.xgb_models[key] = xgb_model

                    # LightGBM
                    logger.info(f"Training LightGBM for {horizon} {multiplier}...")
                    lgb_model = lgb.LGBMClassifier(
                        n_estimators=200,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="binary",
                        metric="binary_logloss",
                        random_state=42,
                        n_jobs=-1,
                        verbose=-1,
                    )
                    lgb_model.fit(X_features, y)
                    self.lgb_models[key] = lgb_model
                else:
                    # Fallback to simple logistic regression
                    from sklearn.linear_model import LogisticRegression

                    model = LogisticRegression(max_iter=1000, random_state=42)
                    model.fit(X_features, y)
                    self.xgb_models[key] = model

        # Train temporal neural network if data available
        if HAS_TORCH and X_temporal is not None:
            logger.info("Training temporal attention network...")
            self._train_temporal_model(X_temporal, y_labels, horizons, multipliers)

        # Set ensemble weights (can be optimized via validation)
        self.ensemble_weights = {"xgb": 0.4, "lgb": 0.4, "temporal": 0.2}

        self.trained_at = datetime.utcnow()
        self.is_trained = True
        logger.info("Training complete!")

    def _train_temporal_model(
        self,
        X_temporal: np.ndarray,
        y_labels: Dict[Tuple[str, str], np.ndarray],
        horizons: List[str],
        multipliers: List[str],
    ):
        """Train temporal neural network."""
        if not HAS_TORCH:
            return

        # Prepare targets (flatten all horizons/multipliers)
        y_combined = []
        for horizon in horizons:
            for multiplier in multipliers:
                key = (horizon, multiplier)
                if key in y_labels:
                    y_combined.append(y_labels[key])

        if not y_combined:
            logger.warning("No labels for temporal model")
            return

        y_combined = np.stack(y_combined, axis=1)  # (n_samples, n_targets)

        # Initialize model
        input_dim = X_temporal.shape[2]
        output_dim = y_combined.shape[1]
        self.temporal_model = TemporalAttentionNetwork(
            input_dim=input_dim, hidden_dim=128, output_dim=output_dim, num_heads=4
        )

        # Training setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temporal_model = self.temporal_model.to(device)

        optimizer = torch.optim.Adam(self.temporal_model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_temporal).to(device)
        y_tensor = torch.FloatTensor(y_combined).to(device)

        # Training loop (simplified - in production use DataLoader, validation, etc.)
        epochs = 50
        batch_size = 32
        n_samples = X_tensor.shape[0]

        self.temporal_model.train()
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                batch_X = X_tensor[i : i + batch_size]
                batch_y = y_tensor[i : i + batch_size]

                optimizer.zero_grad()
                outputs = self.temporal_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

        self.temporal_model.eval()

    def predict_proba(
        self,
        X_features: np.ndarray,
        X_temporal: Optional[np.ndarray] = None,
        horizons: List[str] = ["24h", "7d", "30d"],
        multipliers: List[str] = ["2x", "5x", "10x", "20x", "50x"],
    ) -> Dict[Tuple[str, str], float]:
        """
        Predict probabilities for all (horizon, multiplier) combinations.

        Args:
            X_features: Feature matrix (n_samples, n_features)
            X_temporal: Temporal sequences (n_samples, seq_len, temporal_features)
            horizons: Horizons to predict
            multipliers: Multipliers to predict

        Returns:
            Dict {(horizon, multiplier): probability}
        """
        if not self.is_trained:
            logger.warning("Model not trained. Using dummy predictions.")
            return self._dummy_predictions(horizons, multipliers)

        predictions = {}

        for horizon in horizons:
            for multiplier in multipliers:
                key = (horizon, multiplier)

                # Get predictions from each model
                probs = []
                weights = []

                # XGBoost
                if key in self.xgb_models:
                    xgb_prob = self.xgb_models[key].predict_proba(X_features)[:, 1]
                    probs.append(xgb_prob)
                    weights.append(self.ensemble_weights.get("xgb", 0.4))

                # LightGBM
                if key in self.lgb_models:
                    lgb_prob = self.lgb_models[key].predict_proba(X_features)[:, 1]
                    probs.append(lgb_prob)
                    weights.append(self.ensemble_weights.get("lgb", 0.4))

                # Temporal (if available)
                if self.temporal_model is not None and X_temporal is not None:
                    temporal_prob = self._predict_temporal(X_temporal, key, horizons, multipliers)
                    if temporal_prob is not None:
                        probs.append(temporal_prob)
                        weights.append(self.ensemble_weights.get("temporal", 0.2))

                # Ensemble
                if probs:
                    weights = np.array(weights)
                    weights = weights / weights.sum()  # Normalize
                    ensemble_prob = sum(p * w for p, w in zip(probs, weights))
                    predictions[key] = float(ensemble_prob.mean())  # Average over batch
                else:
                    predictions[key] = 0.05  # Default fallback

        return predictions

    def _predict_temporal(
        self,
        X_temporal: np.ndarray,
        key: Tuple[str, str],
        horizons: List[str],
        multipliers: List[str],
    ) -> Optional[np.ndarray]:
        """Predict using temporal model."""
        if not HAS_TORCH or self.temporal_model is None:
            return None

        # Map key to output index
        all_keys = [(h, m) for h in horizons for m in multipliers]
        if key not in all_keys:
            return None

        idx = all_keys.index(key)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_temporal).to(device)
            outputs = self.temporal_model(X_tensor)
            return outputs[:, idx].cpu().numpy()

    def _dummy_predictions(
        self, horizons: List[str], multipliers: List[str]
    ) -> Dict[Tuple[str, str], float]:
        """Generate dummy predictions for demonstration."""
        # Realistic dummy probabilities that decay with multiplier
        multiplier_base = {"2x": 0.35, "5x": 0.15, "10x": 0.05, "20x": 0.02, "50x": 0.005}
        horizon_decay = {"24h": 1.0, "7d": 1.5, "30d": 2.0}

        predictions = {}
        for horizon in horizons:
            for multiplier in multipliers:
                base = multiplier_base.get(multiplier, 0.01)
                decay = horizon_decay.get(horizon, 1.0)
                predictions[(horizon, multiplier)] = base / decay

        return predictions

    def save(self, path: Optional[Path] = None):
        """Save models to disk."""
        save_path = path or self.model_dir
        save_path.mkdir(parents=True, exist_ok=True)

        # Save XGBoost models
        for key, model in self.xgb_models.items():
            filename = f"xgb_{key[0]}_{key[1]}.pkl"
            joblib.dump(model, save_path / filename)

        # Save LightGBM models
        for key, model in self.lgb_models.items():
            filename = f"lgb_{key[0]}_{key[1]}.pkl"
            joblib.dump(model, save_path / filename)

        # Save temporal model
        if HAS_TORCH and self.temporal_model is not None:
            torch.save(self.temporal_model.state_dict(), save_path / "temporal_model.pt")

        # Save metadata
        metadata = {
            "version": self.version,
            "trained_at": self.trained_at.isoformat() if self.trained_at else None,
            "feature_names": self.feature_names,
            "ensemble_weights": self.ensemble_weights,
            "is_trained": self.is_trained,
        }
        joblib.dump(metadata, save_path / "metadata.pkl")

        logger.info(f"Models saved to {save_path}")

    def load(self, path: Optional[Path] = None):
        """Load models from disk."""
        load_path = path or self.model_dir

        if not load_path.exists():
            logger.warning(f"Model path {load_path} does not exist")
            return

        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.version = metadata.get("version", "1.0.0")
            self.feature_names = metadata.get("feature_names", [])
            self.ensemble_weights = metadata.get("ensemble_weights", {})
            self.is_trained = metadata.get("is_trained", False)
            trained_at_str = metadata.get("trained_at")
            if trained_at_str:
                self.trained_at = datetime.fromisoformat(trained_at_str)

        # Load XGBoost models
        for model_file in load_path.glob("xgb_*.pkl"):
            # Parse filename: xgb_24h_2x.pkl -> ("24h", "2x")
            parts = model_file.stem.split("_")
            if len(parts) >= 3:
                key = (parts[1], parts[2])
                self.xgb_models[key] = joblib.load(model_file)

        # Load LightGBM models
        for model_file in load_path.glob("lgb_*.pkl"):
            parts = model_file.stem.split("_")
            if len(parts) >= 3:
                key = (parts[1], parts[2])
                self.lgb_models[key] = joblib.load(model_file)

        # Load temporal model
        temporal_path = load_path / "temporal_model.pt"
        if HAS_TORCH and temporal_path.exists():
            # Need to know architecture to load - in production, save architecture config
            logger.info("Temporal model found but architecture needed to load")

        logger.info(f"Models loaded from {load_path}")
