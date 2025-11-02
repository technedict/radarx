"""
Risk Scorer

Multi-component risk assessment model for token security analysis.
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


class RiskScorer:
    """
    Multi-component risk scoring model.
    
    Assesses five risk dimensions:
    1. Rug Risk - Contract vulnerabilities, LP lock status
    2. Dev Risk - Dev holdings, selling patterns, wallet flags
    3. Distribution Risk - Holder concentration, bot activity
    4. Social Manipulation Risk - Fake engagement, coordinated pumps
    5. Liquidity Risk - Depth, volatility, slippage
    
    Combines individual scorers into a composite risk score (0-100).
    """
    
    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize risk scorer.
        
        Args:
            model_dir: Directory to load/save models
        """
        self.model_dir = model_dir or Path("models/risk")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Component scorers
        self.rug_scorer = None
        self.dev_scorer = None
        self.distribution_scorer = None
        self.social_scorer = None
        self.liquidity_scorer = None
        
        # Composite weights
        self.component_weights = {
            'rug_risk': 0.30,
            'dev_risk': 0.25,
            'distribution_risk': 0.20,
            'social_risk': 0.15,
            'liquidity_risk': 0.10
        }
        
        self.is_trained = False
        self.version = "1.0.0"
        
    def train(self, X_features: np.ndarray, 
              y_labels: Dict[str, np.ndarray],
              feature_names: List[str]):
        """
        Train risk scoring models.
        
        Args:
            X_features: Feature matrix
            y_labels: Dict with labels for each risk component
                     {'rug_risk': labels, 'dev_risk': labels, ...}
            feature_names: Feature names
        """
        logger.info("Training risk scoring models...")
        
        # Train individual component scorers
        for component in ['rug_risk', 'dev_risk', 'distribution_risk', 
                         'social_risk', 'liquidity_risk']:
            if component not in y_labels:
                logger.warning(f"No labels for {component}, skipping")
                continue
            
            y = y_labels[component]
            
            if HAS_XGB:
                logger.info(f"Training XGBoost for {component}...")
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='reg:squarederror',
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_features, y)
            else:
                # Fallback to simple linear regression
                from sklearn.linear_model import Ridge
                model = Ridge(alpha=1.0, random_state=42)
                model.fit(X_features, y)
            
            # Assign to appropriate attribute
            if component == 'rug_risk':
                self.rug_scorer = model
            elif component == 'dev_risk':
                self.dev_scorer = model
            elif component == 'distribution_risk':
                self.distribution_scorer = model
            elif component == 'social_risk':
                self.social_scorer = model
            elif component == 'liquidity_risk':
                self.liquidity_scorer = model
        
        self.is_trained = True
        logger.info("Risk scoring models trained!")
        
    def score(self, X_features: np.ndarray) -> Dict[str, float]:
        """
        Score risk components.
        
        Args:
            X_features: Feature matrix (single sample or batch)
            
        Returns:
            Dict with component scores and composite score
        """
        if not self.is_trained:
            logger.warning("Models not trained. Using heuristic scores.")
            return self._heuristic_score(X_features)
        
        # Ensure 2D array
        if X_features.ndim == 1:
            X_features = X_features.reshape(1, -1)
        
        scores = {}
        
        # Score each component (0-100 scale)
        if self.rug_scorer is not None:
            rug_score = self.rug_scorer.predict(X_features)[0]
            scores['rug_risk'] = float(np.clip(rug_score, 0, 100))
        else:
            scores['rug_risk'] = 50.0
        
        if self.dev_scorer is not None:
            dev_score = self.dev_scorer.predict(X_features)[0]
            scores['dev_risk'] = float(np.clip(dev_score, 0, 100))
        else:
            scores['dev_risk'] = 50.0
        
        if self.distribution_scorer is not None:
            dist_score = self.distribution_scorer.predict(X_features)[0]
            scores['distribution_risk'] = float(np.clip(dist_score, 0, 100))
        else:
            scores['distribution_risk'] = 50.0
        
        if self.social_scorer is not None:
            social_score = self.social_scorer.predict(X_features)[0]
            scores['social_risk'] = float(np.clip(social_score, 0, 100))
        else:
            scores['social_risk'] = 50.0
        
        if self.liquidity_scorer is not None:
            liq_score = self.liquidity_scorer.predict(X_features)[0]
            scores['liquidity_risk'] = float(np.clip(liq_score, 0, 100))
        else:
            scores['liquidity_risk'] = 50.0
        
        # Compute composite score (weighted average)
        composite = sum(
            scores[component] * self.component_weights.get(component, 0.2)
            for component in scores
        )
        scores['composite_score'] = float(composite)
        
        return scores
    
    def _heuristic_score(self, X_features: np.ndarray) -> Dict[str, float]:
        """
        Generate heuristic risk scores based on feature analysis.
        Used when models are not trained.
        """
        # Simple heuristic scoring for demonstration
        # In production, this would analyze feature values
        
        scores = {
            'rug_risk': 30.0,
            'dev_risk': 50.0,
            'distribution_risk': 45.0,
            'social_risk': 35.0,
            'liquidity_risk': 40.0
        }
        
        composite = sum(
            scores[component] * self.component_weights.get(component, 0.2)
            for component in scores
        )
        scores['composite_score'] = float(composite)
        
        return scores
    
    def get_risk_flags(self, scores: Dict[str, float], 
                      features: Optional[Dict[str, float]] = None) -> List[str]:
        """
        Generate human-readable risk flags based on scores and features.
        
        Args:
            scores: Risk component scores
            features: Optional feature dict for detailed analysis
            
        Returns:
            List of risk flag descriptions
        """
        flags = []
        
        # Rug risk flags
        if scores.get('rug_risk', 0) > 70:
            flags.append("High Contract Risk")
            if features:
                if features.get('liquidity_locked', False) is False:
                    flags.append("LP Not Locked")
                if features.get('ownership_renounced', False) is False:
                    flags.append("Ownership Not Renounced")
        
        # Dev risk flags
        if scores.get('dev_risk', 0) > 60:
            if features and features.get('dev_holding_pct', 0) > 0.20:
                flags.append("High Dev Holding")
            if features and features.get('dev_recent_sell', False):
                flags.append("Recent Dev Sell")
        
        # Distribution risk flags
        if scores.get('distribution_risk', 0) > 70:
            if features and features.get('holder_gini', 0) > 0.80:
                flags.append("Highly Concentrated Holdings")
            if features and features.get('top10_concentration', 0) > 0.70:
                flags.append("Top 10 Hold >70%")
        
        # Social manipulation risk flags
        if scores.get('social_risk', 0) > 65:
            if features and features.get('bot_score', 0) > 0.50:
                flags.append("High Bot Activity")
            if features and features.get('duplicate_ratio', 0) > 0.40:
                flags.append("Suspicious Social Patterns")
        
        # Liquidity risk flags
        if scores.get('liquidity_risk', 0) > 60:
            if features and features.get('liquidity_usd', 0) < 10000:
                flags.append("Low Liquidity (<$10k)")
            if features and features.get('liquidity_depth_5pct', 0) < 5000:
                flags.append("Poor Liquidity Depth")
        
        return flags
    
    def save(self, path: Optional[Path] = None):
        """Save risk models to disk."""
        save_path = path or self.model_dir
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save each component scorer
        if self.rug_scorer:
            joblib.dump(self.rug_scorer, save_path / "rug_scorer.pkl")
        if self.dev_scorer:
            joblib.dump(self.dev_scorer, save_path / "dev_scorer.pkl")
        if self.distribution_scorer:
            joblib.dump(self.distribution_scorer, save_path / "distribution_scorer.pkl")
        if self.social_scorer:
            joblib.dump(self.social_scorer, save_path / "social_scorer.pkl")
        if self.liquidity_scorer:
            joblib.dump(self.liquidity_scorer, save_path / "liquidity_scorer.pkl")
        
        # Save metadata
        metadata = {
            'version': self.version,
            'component_weights': self.component_weights,
            'is_trained': self.is_trained
        }
        joblib.dump(metadata, save_path / "metadata.pkl")
        
        logger.info(f"Risk models saved to {save_path}")
    
    def load(self, path: Optional[Path] = None):
        """Load risk models from disk."""
        load_path = path or self.model_dir
        
        if not load_path.exists():
            logger.warning(f"Model path {load_path} does not exist")
            return
        
        # Load metadata
        metadata_path = load_path / "metadata.pkl"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.version = metadata.get('version', '1.0.0')
            self.component_weights = metadata.get('component_weights', {})
            self.is_trained = metadata.get('is_trained', False)
        
        # Load component scorers
        for component, attr in [
            ('rug_scorer.pkl', 'rug_scorer'),
            ('dev_scorer.pkl', 'dev_scorer'),
            ('distribution_scorer.pkl', 'distribution_scorer'),
            ('social_scorer.pkl', 'social_scorer'),
            ('liquidity_scorer.pkl', 'liquidity_scorer')
        ]:
            model_path = load_path / component
            if model_path.exists():
                setattr(self, attr, joblib.load(model_path))
        
        logger.info(f"Risk models loaded from {load_path}")
