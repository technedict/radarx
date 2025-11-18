"""
Explainer Model

SHAP-based feature importance and contribution analysis for model explanations.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logger.warning("SHAP not available. Using fallback explanations.")


class ExplainerModel:
    """
    Model explainability using SHAP (SHapley Additive exPlanations).

    Provides:
    - Feature importance rankings
    - Individual prediction explanations
    - Feature contribution analysis (positive/negative)
    - Human-readable explanations
    """

    def __init__(self):
        """Initialize explainer."""
        self.explainers = {}  # {model_key: shap.Explainer}
        self.feature_names = []
        self.baseline_values = {}

    def fit(
        self, model, X_background: np.ndarray, feature_names: List[str], model_key: str = "default"
    ):
        """
        Fit SHAP explainer to a model.

        Args:
            model: Trained model (XGBoost, LightGBM, or sklearn)
            X_background: Background dataset for SHAP (typically training sample)
            feature_names: List of feature names
            model_key: Identifier for this model
        """
        self.feature_names = feature_names

        if HAS_SHAP:
            logger.info(f"Fitting SHAP explainer for {model_key}...")

            # Use TreeExplainer for tree-based models (XGBoost, LightGBM)
            # Use KernelExplainer for other models
            try:
                explainer = shap.TreeExplainer(model, X_background)
            except:
                # Fallback to KernelExplainer
                logger.info("Using KernelExplainer (slower)...")

                def model_predict(X):
                    if hasattr(model, "predict_proba"):
                        return model.predict_proba(X)[:, 1]
                    else:
                        return model.predict(X)

                explainer = shap.KernelExplainer(model_predict, shap.sample(X_background, 100))

            self.explainers[model_key] = explainer
            self.baseline_values[model_key] = X_background.mean(axis=0)
            logger.info(f"SHAP explainer fitted for {model_key}")
        else:
            logger.warning("SHAP not available. Explanations will be approximate.")
            self.baseline_values[model_key] = X_background.mean(axis=0)

    def explain(self, X: np.ndarray, model_key: str = "default", top_n: int = 5) -> Dict:
        """
        Explain predictions for input samples.

        Args:
            X: Feature matrix to explain (n_samples, n_features)
            model_key: Which model to use for explanation
            top_n: Number of top features to return

        Returns:
            Dict with SHAP values and top feature contributions
        """
        if model_key not in self.explainers and not HAS_SHAP:
            return self._fallback_explanation(X, top_n)

        if model_key not in self.explainers:
            logger.warning(f"No explainer for {model_key}. Using fallback.")
            return self._fallback_explanation(X, top_n)

        # Get SHAP values
        explainer = self.explainers[model_key]
        shap_values = explainer.shap_values(X)

        # If binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        # Ensure 2D
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Get top features by absolute SHAP value
        explanations = []
        for i in range(shap_values.shape[0]):
            sample_shap = shap_values[i]
            sample_features = X[i]

            # Get indices of top features
            abs_shap = np.abs(sample_shap)
            top_indices = np.argsort(abs_shap)[-top_n:][::-1]

            top_features = []
            for idx in top_indices:
                feature_name = (
                    self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                )
                contribution = float(sample_shap[idx])
                feature_value = float(sample_features[idx])
                direction = "positive" if contribution > 0 else "negative"

                top_features.append(
                    {
                        "feature_name": feature_name,
                        "contribution": abs(contribution),
                        "direction": direction,
                        "value": feature_value,
                        "description": self._get_feature_description(
                            feature_name, feature_value, contribution
                        ),
                    }
                )

            explanations.append(
                {
                    "top_features": top_features,
                    "base_value": (
                        float(explainer.expected_value)
                        if hasattr(explainer, "expected_value")
                        else 0.5
                    ),
                    "shap_values": sample_shap.tolist(),
                }
            )

        return explanations[0] if len(explanations) == 1 else explanations

    def get_global_importance(self, model_key: str = "default") -> List[Dict]:
        """
        Get global feature importance across all predictions.

        Args:
            model_key: Which model to analyze

        Returns:
            List of features sorted by importance
        """
        # This would typically be computed from a validation set
        # For now, return placeholder
        logger.info("Global importance requires validation set SHAP values")
        return []

    def _fallback_explanation(self, X: np.ndarray, top_n: int = 5) -> Dict:
        """
        Fallback explanation when SHAP is not available.
        Uses simple heuristics based on feature magnitudes.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Simple heuristic: features with highest absolute values
        explanations = []
        for i in range(X.shape[0]):
            sample = X[i]

            # Get top features by absolute value
            abs_values = np.abs(sample)
            top_indices = np.argsort(abs_values)[-top_n:][::-1]

            top_features = []
            for idx in top_indices:
                feature_name = (
                    self.feature_names[idx] if idx < len(self.feature_names) else f"feature_{idx}"
                )
                value = float(sample[idx])

                # Approximate contribution (normalized)
                contribution = abs_values[idx] / (abs_values.sum() + 1e-10)
                direction = "positive" if value > 0 else "negative"

                top_features.append(
                    {
                        "feature_name": feature_name,
                        "contribution": float(contribution),
                        "direction": direction,
                        "value": value,
                        "description": self._get_feature_description(
                            feature_name, value, contribution
                        ),
                    }
                )

            explanations.append(
                {"top_features": top_features, "base_value": 0.5, "shap_values": None}
            )

        return explanations[0] if len(explanations) == 1 else explanations

    def _get_feature_description(self, feature_name: str, value: float, contribution: float) -> str:
        """
        Generate human-readable description of feature contribution.

        Args:
            feature_name: Name of the feature
            value: Feature value
            contribution: SHAP contribution value

        Returns:
            Human-readable description
        """
        direction = "increases" if contribution > 0 else "decreases"

        # Feature-specific descriptions
        if "volume" in feature_name.lower():
            return f"Trading volume of ${value:,.0f} {direction} probability"
        elif "liquidity" in feature_name.lower():
            return f"Liquidity of ${value:,.0f} {direction} probability"
        elif "gini" in feature_name.lower():
            return f"Holder concentration (Gini={value:.2f}) {direction} probability"
        elif "smart_money" in feature_name.lower():
            return f"Smart money activity ({value:.1%}) {direction} probability"
        elif "sentiment" in feature_name.lower():
            return f"Social sentiment score ({value:.2f}) {direction} probability"
        elif "mentions" in feature_name.lower():
            return f"{value:.0f} social mentions {direction} probability"
        elif "kol" in feature_name.lower():
            return f"KOL activity ({value:.0f}) {direction} probability"
        elif "bot_score" in feature_name.lower():
            return f"Bot activity score ({value:.1%}) {direction} probability"
        elif "price_change" in feature_name.lower():
            return f"Price change ({value:+.1%}) {direction} probability"
        elif "holder" in feature_name.lower():
            return f"{value:.0f} token holders {direction} probability"
        else:
            return f"{feature_name}={value:.3f} {direction} probability"
