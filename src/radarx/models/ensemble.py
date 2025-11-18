"""
Advanced Model Ensemble with Dynamic Weighting

Implements sophisticated ensemble strategies including:
- Dynamic weight adjustment based on recent performance
- Confidence-weighted voting
- Stacking with meta-learner
- Feature-based model selection
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DynamicEnsemble:
    """
    Dynamic ensemble that adjusts model weights based on recent performance.

    This addresses the issue where some models perform better on different
    market conditions or token types.
    """

    def __init__(
        self,
        models: Dict[str, any],
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.1,
        min_weight: float = 0.05,
    ):
        """
        Initialize dynamic ensemble.

        Args:
            models: Dictionary of models {name: model}
            initial_weights: Initial weights (default: equal)
            adaptation_rate: How quickly weights adapt (0-1)
            min_weight: Minimum weight for any model
        """
        self.models = models
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight

        # Initialize weights
        if initial_weights:
            self.weights = initial_weights
        else:
            n_models = len(models)
            self.weights = {name: 1.0 / n_models for name in models}

        # Normalize weights
        self._normalize_weights()

        # Performance tracking
        self.model_performance = {name: [] for name in models}
        self.recent_window = 100  # Track last 100 predictions

    def _normalize_weights(self):
        """Ensure weights sum to 1 and respect minimum."""
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] = max(self.min_weight, self.weights[name] / total)

            # Re-normalize after applying minimum
            total = sum(self.weights.values())
            if total > 0:
                for name in self.weights:
                    self.weights[name] /= total

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make ensemble prediction with dynamic weights.

        Args:
            X: Feature matrix

        Returns:
            Weighted ensemble predictions
        """
        predictions = {}

        # Get predictions from each model
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning(f"Model {name} failed to predict: {e}")
                continue

        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred

        return ensemble_pred

    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction with confidence estimates.

        Args:
            X: Feature matrix

        Returns:
            (predictions, confidence_scores)
        """
        predictions = {}

        # Get predictions from each model
        for name, model in self.models.items():
            try:
                predictions[name] = model.predict(X)
            except Exception as e:
                logger.warning(f"Model {name} failed to predict: {e}")
                continue

        if not predictions:
            return np.zeros(len(X)), np.zeros(len(X))

        # Convert to array for variance calculation
        pred_array = np.array([predictions[name] for name in predictions])

        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * pred

        # Confidence based on agreement (low variance = high confidence)
        variance = np.var(pred_array, axis=0)
        confidence = 1.0 / (1.0 + variance)  # Higher when models agree

        return ensemble_pred, confidence

    def update_weights(self, model_errors: Dict[str, float]):
        """
        Update model weights based on recent performance.

        Args:
            model_errors: Dictionary of model errors {name: error}
        """
        # Track performance
        for name, error in model_errors.items():
            if name in self.model_performance:
                self.model_performance[name].append(error)
                # Keep only recent window
                if len(self.model_performance[name]) > self.recent_window:
                    self.model_performance[name].pop(0)

        # Calculate average performance
        avg_performance = {}
        for name in self.models:
            if self.model_performance[name]:
                # Use reciprocal of error (lower error = better performance)
                avg_error = np.mean(self.model_performance[name])
                avg_performance[name] = 1.0 / (avg_error + 1e-6)

        if not avg_performance:
            return

        # Update weights with exponential moving average
        total_perf = sum(avg_performance.values())
        for name in self.weights:
            if name in avg_performance:
                target_weight = avg_performance[name] / total_perf
                # Exponential moving average
                self.weights[name] = (1 - self.adaptation_rate) * self.weights[
                    name
                ] + self.adaptation_rate * target_weight

        # Normalize and apply constraints
        self._normalize_weights()

        logger.info(f"Updated ensemble weights: {self.weights}")

    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions to ensemble prediction.

        Args:
            X: Feature matrix

        Returns:
            Dictionary of weighted predictions per model
        """
        contributions = {}

        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                contributions[name] = self.weights[name] * pred
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
                contributions[name] = np.zeros(len(X))

        return contributions


class StackedEnsemble:
    """
    Stacked ensemble with meta-learner.

    First level: Multiple base models
    Second level: Meta-learner that learns to combine base models
    """

    def __init__(
        self,
        base_models: Dict[str, any],
        meta_model: Optional[any] = None,
    ):
        """
        Initialize stacked ensemble.

        Args:
            base_models: Dictionary of base models
            meta_model: Meta-learner (defaults to LogisticRegression)
        """
        self.base_models = base_models
        self.meta_model = meta_model
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray, cv: int = 5):
        """
        Fit stacked ensemble using cross-validation.

        Args:
            X: Training features
            y: Training labels
            cv: Number of cross-validation folds
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict

            # Get out-of-fold predictions from base models
            meta_features = []

            for name, model in self.base_models.items():
                try:
                    # Get cross-validated predictions
                    oof_preds = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
                    # Take positive class probability
                    if oof_preds.ndim > 1:
                        oof_preds = oof_preds[:, 1]
                    meta_features.append(oof_preds)

                    # Fit on full data for future predictions
                    model.fit(X, y)

                    logger.info(f"Base model {name} fitted")
                except Exception as e:
                    logger.error(f"Failed to fit base model {name}: {e}")
                    continue

            if not meta_features:
                logger.error("No base models fitted successfully")
                return

            # Stack meta features
            meta_X = np.column_stack(meta_features)

            # Fit meta-learner
            if self.meta_model is None:
                self.meta_model = LogisticRegression()

            self.meta_model.fit(meta_X, y)
            self.is_fitted = True

            logger.info("Stacked ensemble fitted successfully")

        except ImportError:
            logger.error("sklearn not available for stacking")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction with stacked ensemble.

        Args:
            X: Feature matrix

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            logger.warning("Ensemble not fitted")
            return np.zeros(len(X))

        # Get predictions from base models
        base_predictions = []

        for name, model in self.base_models.items():
            try:
                pred = model.predict_proba(X)
                if pred.ndim > 1:
                    pred = pred[:, 1]
                base_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Base model {name} prediction failed: {e}")
                # Use zeros as fallback
                base_predictions.append(np.zeros(len(X)))

        if not base_predictions:
            return np.zeros(len(X))

        # Stack base predictions
        meta_X = np.column_stack(base_predictions)

        # Meta-learner prediction
        try:
            final_pred = self.meta_model.predict_proba(meta_X)
            if final_pred.ndim > 1:
                final_pred = final_pred[:, 1]
            return final_pred
        except Exception as e:
            logger.error(f"Meta-learner prediction failed: {e}")
            # Fallback to simple average
            return np.mean(meta_X, axis=1)


class FeatureBasedSelector:
    """
    Selects best model based on input features.

    Different models may perform better on different token types
    or market conditions. This selector learns which model to use
    based on input characteristics.
    """

    def __init__(self, models: Dict[str, any], routing_model: Optional[any] = None):
        """
        Initialize feature-based selector.

        Args:
            models: Dictionary of models
            routing_model: Model to route samples to best base model
        """
        self.models = models
        self.routing_model = routing_model
        self.model_assignments = {}  # Track which model used for each sample

    def fit_routing(self, X: np.ndarray, y: np.ndarray):
        """
        Fit routing model to learn which base model is best for each sample.

        Args:
            X: Training features
            y: Training labels
        """
        try:
            from sklearn.ensemble import RandomForestClassifier

            # Get predictions from all models
            model_preds = {}
            for name, model in self.models.items():
                try:
                    pred = model.predict(X)
                    model_preds[name] = pred
                except:
                    continue

            if not model_preds:
                return

            # Find best model for each sample
            best_models = []
            for i in range(len(y)):
                errors = {}
                for name, preds in model_preds.items():
                    errors[name] = abs(preds[i] - y[i])

                best_model = min(errors, key=errors.get)
                best_models.append(list(self.models.keys()).index(best_model))

            # Train routing model
            if self.routing_model is None:
                self.routing_model = RandomForestClassifier(n_estimators=50)

            self.routing_model.fit(X, best_models)
            logger.info("Routing model fitted")

        except ImportError:
            logger.warning("sklearn not available for routing")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make prediction using routed models.

        Args:
            X: Feature matrix

        Returns:
            Ensemble predictions
        """
        if self.routing_model is None:
            # Fallback to simple average
            preds = []
            for model in self.models.values():
                try:
                    preds.append(model.predict(X))
                except:
                    continue
            return np.mean(preds, axis=0) if preds else np.zeros(len(X))

        # Route samples to best models
        try:
            routes = self.routing_model.predict(X)
        except:
            # Fallback
            routes = np.zeros(len(X), dtype=int)

        # Get predictions from routed models
        predictions = np.zeros(len(X))
        model_list = list(self.models.values())

        for i, route in enumerate(routes):
            try:
                model = model_list[int(route)]
                predictions[i] = model.predict(X[i : i + 1])[0]
            except:
                # Use first model as fallback
                predictions[i] = model_list[0].predict(X[i : i + 1])[0]

        return predictions
