"""
Online Learner

Implements continual learning with incremental model updates.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

try:
    from river import forest, ensemble, metrics
    HAS_RIVER = True
except ImportError:
    HAS_RIVER = False
    logger.warning("River not available. Online learning limited.")


class OnlineLearner:
    """
    Online learning system for continual model updates.
    
    Features:
    - Incremental learning from new data
    - Weighted sampling (recent data weighted higher)
    - Periodic full retraining
    - Performance tracking over time
    - Automatic trigger for full retraining when drift detected
    """
    
    def __init__(self, base_models: Optional[dict] = None,
                 update_frequency: str = "daily",
                 model_dir: Optional[Path] = None):
        """
        Initialize online learner.
        
        Args:
            base_models: Initial models to update incrementally
            update_frequency: How often to update ("daily", "weekly")
            model_dir: Directory for model checkpoints
        """
        self.base_models = base_models or {}
        self.update_frequency = update_frequency
        self.model_dir = model_dir or Path("models/online")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Online models (if using River)
        self.online_models = {}
        
        # Performance tracking
        self.performance_history = []
        self.last_update = None
        self.update_count = 0
        
        # Retraining triggers
        self.retrain_threshold = 0.10  # Retrain if performance drops >10%
        self.samples_since_retrain = 0
        self.max_samples_before_retrain = 10000
        
    def should_update(self) -> bool:
        """
        Check if models should be updated based on schedule.
        
        Returns:
            True if update is due
        """
        if self.last_update is None:
            return True
        
        now = datetime.utcnow()
        
        if self.update_frequency == "daily":
            return (now - self.last_update) > timedelta(days=1)
        elif self.update_frequency == "weekly":
            return (now - self.last_update) > timedelta(weeks=1)
        else:
            return False
    
    def partial_fit(self, X: np.ndarray, y: dict, 
                   sample_weights: Optional[np.ndarray] = None):
        """
        Incrementally update models with new data.
        
        Args:
            X: Feature matrix
            y: Labels dict {(horizon, multiplier): labels}
            sample_weights: Optional weights (recent samples weighted higher)
        """
        logger.info("Performing partial fit with new data...")
        
        # Default: weight recent samples higher (exponential decay)
        if sample_weights is None:
            n_samples = X.shape[0]
            decay_rate = 0.99
            sample_weights = np.array([decay_rate ** (n_samples - i - 1) 
                                      for i in range(n_samples)])
            sample_weights /= sample_weights.sum()  # Normalize
        
        # Update each model
        for key, labels in y.items():
            if key not in self.base_models:
                logger.warning(f"No base model for {key}, skipping")
                continue
            
            model = self.base_models[key]
            
            # Check if model supports partial_fit
            if hasattr(model, 'partial_fit'):
                model.partial_fit(X, labels, sample_weight=sample_weights)
                logger.info(f"Partial fit for {key} complete")
            else:
                logger.warning(f"Model for {key} doesn't support partial_fit")
        
        self.update_count += 1
        self.last_update = datetime.utcnow()
        self.samples_since_retrain += len(X)
        
    def should_retrain(self, current_performance: float) -> bool:
        """
        Check if full retraining is needed.
        
        Args:
            current_performance: Current model performance metric
            
        Returns:
            True if retraining is recommended
        """
        # Trigger 1: Performance degradation
        if self.performance_history:
            # Extract performance values from history dicts
            perf_values = [h['performance'] for h in self.performance_history[-10:]]  # Last 10 measurements
            baseline = np.mean(perf_values)
            if current_performance < baseline * (1 - self.retrain_threshold):
                logger.warning(f"Performance drop detected: {current_performance:.3f} vs {baseline:.3f}")
                return True
        
        # Trigger 2: Too many samples since last retrain
        if self.samples_since_retrain >= self.max_samples_before_retrain:
            logger.info(f"Sample threshold reached: {self.samples_since_retrain}")
            return True
        
        return False
    
    def log_performance(self, performance: float):
        """
        Log performance metric for tracking.
        
        Args:
            performance: Performance metric value (e.g., accuracy, AUC)
        """
        from datetime import datetime
        
        self.performance_history.append({
            'timestamp': datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow(),
            'performance': performance,
            'update_count': self.update_count
        })
        
        # Keep last 100 measurements
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_performance_trend(self, window: int = 10) -> dict:
        """
        Get performance trend statistics.
        
        Args:
            window: Number of recent measurements to analyze
            
        Returns:
            Dict with trend statistics
        """
        if len(self.performance_history) < 2:
            return {'status': 'insufficient_data'}
        
        recent = self.performance_history[-window:]
        perfs = [h['performance'] for h in recent]
        
        return {
            'current': perfs[-1],
            'mean': np.mean(perfs),
            'std': np.std(perfs),
            'min': np.min(perfs),
            'max': np.max(perfs),
            'trend': 'improving' if perfs[-1] > np.mean(perfs[:-1]) else 'degrading'
        }
    
    def reset_retrain_counter(self):
        """Reset counter after full retraining."""
        self.samples_since_retrain = 0
        logger.info("Retrain counter reset")
    
    def save_checkpoint(self, path: Optional[Path] = None):
        """Save online learning state."""
        save_path = path or self.model_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save state
        state = {
            'performance_history': self.performance_history,
            'last_update': self.last_update,
            'update_count': self.update_count,
            'samples_since_retrain': self.samples_since_retrain,
            'update_frequency': self.update_frequency
        }
        joblib.dump(state, save_path / "online_state.pkl")
        
        logger.info(f"Online learning state saved to {save_path}")
    
    def load_checkpoint(self, path: Optional[Path] = None):
        """Load online learning state."""
        load_path = path or self.model_dir
        state_path = load_path / "online_state.pkl"
        
        if not state_path.exists():
            logger.warning("No checkpoint found")
            return
        
        state = joblib.load(state_path)
        self.performance_history = state.get('performance_history', [])
        self.last_update = state.get('last_update')
        self.update_count = state.get('update_count', 0)
        self.samples_since_retrain = state.get('samples_since_retrain', 0)
        self.update_frequency = state.get('update_frequency', 'daily')
        
        logger.info(f"Online learning state loaded from {load_path}")


class IncrementalEnsemble:
    """
    Incremental ensemble that maintains multiple online models.
    Uses River library for true streaming ML.
    """
    
    def __init__(self):
        """Initialize incremental ensemble."""
        if not HAS_RIVER:
            logger.warning("River not available. Incremental ensemble disabled.")
            return
        
        # Create online models using River
        self.models = {
            'forest': forest.ARFClassifier(n_models=10, seed=42),
            'adaptive': ensemble.AdaptiveRandomForestClassifier(n_models=10, seed=42)
        }
        
        # Metrics
        self.metrics = {
            'accuracy': metrics.Accuracy(),
            'f1': metrics.F1(),
            'auc': metrics.ROCAUC()
        }
    
    def learn_one(self, x: dict, y: int):
        """
        Learn from a single sample (streaming).
        
        Args:
            x: Feature dict
            y: True label
        """
        if not HAS_RIVER:
            return
        
        # Get predictions before learning
        y_pred = {}
        for name, model in self.models.items():
            pred = model.predict_proba_one(x)
            y_pred[name] = pred.get(1, 0.5) if pred else 0.5
        
        # Update metrics
        ensemble_pred = np.mean(list(y_pred.values()))
        for metric in self.metrics.values():
            metric.update(y, ensemble_pred > 0.5)
        
        # Learn from sample
        for model in self.models.values():
            model.learn_one(x, y)
    
    def predict_one(self, x: dict) -> float:
        """
        Predict probability for a single sample.
        
        Args:
            x: Feature dict
            
        Returns:
            Ensemble probability prediction
        """
        if not HAS_RIVER:
            return 0.5
        
        predictions = []
        for model in self.models.values():
            pred = model.predict_proba_one(x)
            predictions.append(pred.get(1, 0.5) if pred else 0.5)
        
        return np.mean(predictions)
    
    def get_metrics(self) -> dict:
        """Get current metric values."""
        return {name: metric.get() for name, metric in self.metrics.items()}
