"""
Calibration Pipeline

Calibrates probability predictions using isotonic regression or Platt scaling.
"""

import numpy as np
import logging
from typing import Optional, Literal
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("sklearn not available. Calibration disabled.")


class CalibrationPipeline:
    """
    Probability calibration pipeline.
    
    Ensures predicted probabilities match observed frequencies.
    Uses isotonic regression (non-parametric) or Platt scaling (parametric).
    
    For each (horizon, multiplier) combination, maintains a separate calibrator.
    """
    
    def __init__(self, method: Literal["isotonic", "platt"] = "isotonic",
                 model_dir: Optional[Path] = None):
        """
        Initialize calibration pipeline.
        
        Args:
            method: Calibration method ("isotonic" or "platt")
            model_dir: Directory to load/save calibrators
        """
        self.method = method
        self.model_dir = model_dir or Path("models/calibration")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.calibrators = {}  # {(horizon, multiplier): calibrator}
        self.is_fitted = False
        
    def fit(self, y_true: dict, y_pred: dict):
        """
        Fit calibration models.
        
        Args:
            y_true: Dict {(horizon, multiplier): binary_labels}
            y_pred: Dict {(horizon, multiplier): predicted_probabilities}
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available. Skipping calibration.")
            return
        
        logger.info(f"Fitting calibration models with {self.method}...")
        
        for key in y_true:
            if key not in y_pred:
                logger.warning(f"No predictions for {key}, skipping calibration")
                continue
            
            true_labels = y_true[key]
            pred_probs = y_pred[key]
            
            # Ensure valid shapes
            if len(true_labels) != len(pred_probs):
                logger.warning(f"Shape mismatch for {key}, skipping")
                continue
            
            # Fit calibrator
            if self.method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(pred_probs, true_labels)
            else:  # platt
                # Platt scaling is logistic regression on prediction scores
                from sklearn.linear_model import LogisticRegression
                calibrator = LogisticRegression()
                # Reshape for sklearn
                X = pred_probs.reshape(-1, 1) if pred_probs.ndim == 1 else pred_probs
                calibrator.fit(X, true_labels)
            
            self.calibrators[key] = calibrator
            logger.info(f"Calibrator fitted for {key}")
        
        self.is_fitted = True
        logger.info("Calibration complete!")
    
    def transform(self, y_pred: dict) -> dict:
        """
        Apply calibration to predictions.
        
        Args:
            y_pred: Dict {(horizon, multiplier): predicted_probabilities}
            
        Returns:
            Dict {(horizon, multiplier): calibrated_probabilities}
        """
        if not self.is_fitted:
            logger.warning("Calibrators not fitted. Returning uncalibrated predictions.")
            return y_pred
        
        calibrated = {}
        
        for key, probs in y_pred.items():
            if key not in self.calibrators:
                logger.warning(f"No calibrator for {key}, using uncalibrated")
                calibrated[key] = probs
                continue
            
            calibrator = self.calibrators[key]
            
            # Apply calibration
            if self.method == "isotonic":
                cal_probs = calibrator.transform(probs)
            else:  # platt
                X = probs.reshape(-1, 1) if probs.ndim == 1 else probs
                cal_probs = calibrator.predict_proba(X)[:, 1]
            
            calibrated[key] = cal_probs
        
        return calibrated
    
    def get_calibration_curve(self, y_true: np.ndarray, y_pred: np.ndarray,
                              n_bins: int = 10) -> tuple:
        """
        Compute calibration curve for visualization.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted probabilities
            n_bins: Number of bins for grouping predictions
            
        Returns:
            (bin_centers, observed_frequencies, bin_counts)
        """
        if not HAS_SKLEARN:
            return None, None, None
        
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred, n_bins=n_bins, strategy='uniform'
        )
        
        # Count samples in each bin
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_counts = np.histogram(y_pred, bins=bin_edges)[0]
        
        return prob_pred, prob_true, bin_counts
    
    def evaluate_calibration(self, y_true: dict, y_pred: dict) -> dict:
        """
        Evaluate calibration quality using metrics.
        
        Args:
            y_true: Dict {(horizon, multiplier): binary_labels}
            y_pred: Dict {(horizon, multiplier): predicted_probabilities}
            
        Returns:
            Dict of calibration metrics per key
        """
        metrics = {}
        
        for key in y_true:
            if key not in y_pred:
                continue
            
            true_labels = y_true[key]
            pred_probs = y_pred[key]
            
            # Expected Calibration Error (ECE)
            ece = self._compute_ece(true_labels, pred_probs)
            
            # Brier score
            brier = np.mean((pred_probs - true_labels) ** 2)
            
            metrics[key] = {
                'ece': float(ece),
                'brier_score': float(brier)
            }
        
        return metrics
    
    def _compute_ece(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE measures the difference between predicted probabilities
        and observed frequencies across bins.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        ece = 0.0
        total_count = len(y_true)
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() == 0:
                continue
            
            bin_count = mask.sum()
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_pred[mask].mean()
            
            ece += (bin_count / total_count) * abs(bin_accuracy - bin_confidence)
        
        return ece
    
    def save(self, path: Optional[Path] = None):
        """Save calibrators to disk."""
        save_path = path or self.model_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        for key, calibrator in self.calibrators.items():
            filename = f"calibrator_{key[0]}_{key[1]}.pkl"
            joblib.dump(calibrator, save_path / filename)
        
        # Save metadata
        metadata = {
            'method': self.method,
            'is_fitted': self.is_fitted
        }
        joblib.dump(metadata, save_path / "metadata.pkl")
        
        logger.info(f"Calibrators saved to {save_path}")
    
    def load(self, path: Optional[Path] = None):
        """Load calibrators from disk."""
        load_path = path or self.model_dir
        
        if not load_path.exists():
            logger.warning(f"Calibration path {load_path} does not exist")
            return
        
        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.method = metadata.get('method', 'isotonic')
            self.is_fitted = metadata.get('is_fitted', False)
        
        # Load calibrators
        for model_file in load_path.glob("calibrator_*.pkl"):
            # Parse filename: calibrator_24h_2x.pkl -> ("24h", "2x")
            parts = model_file.stem.split('_')
            if len(parts) >= 3:
                key = (parts[1], parts[2])
                self.calibrators[key] = joblib.load(model_file)
        
        logger.info(f"Calibrators loaded from {load_path}")
