"""
Unit tests for ML models module.
"""

import pytest
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile

from radarx.models import (
    ProbabilityPredictor,
    RiskScorer,
    ExplainerModel,
    CalibrationPipeline,
    OnlineLearner,
    DriftDetector
)


class TestProbabilityPredictor:
    """Test probability prediction model."""
    
    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ProbabilityPredictor()
        assert predictor.version == "1.0.0"
        assert not predictor.is_trained
        assert predictor.feature_names == []
    
    def test_dummy_predictions(self):
        """Test dummy predictions before training."""
        predictor = ProbabilityPredictor()
        
        X = np.random.rand(10, 20)
        predictions = predictor.predict_proba(
            X,
            horizons=["24h", "7d"],
            multipliers=["2x", "5x", "10x"]
        )
        
        # Check all combinations present
        assert ("24h", "2x") in predictions
        assert ("7d", "10x") in predictions
        
        # Check probabilities are reasonable (decay with multiplier)
        assert predictions[("24h", "2x")] > predictions[("24h", "10x")]
        assert predictions[("24h", "5x")] > predictions[("7d", "5x")]
    
    def test_save_load(self):
        """Test model save/load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            predictor = ProbabilityPredictor(model_dir=Path(tmpdir))
            predictor.feature_names = ['feature1', 'feature2']
            predictor.is_trained = True
            predictor.trained_at = datetime.utcnow()
            
            # Save
            predictor.save()
            
            # Load into new instance
            new_predictor = ProbabilityPredictor(model_dir=Path(tmpdir))
            new_predictor.load()
            
            assert new_predictor.feature_names == ['feature1', 'feature2']
            assert new_predictor.is_trained


class TestRiskScorer:
    """Test risk scoring model."""
    
    def test_initialization(self):
        """Test scorer initialization."""
        scorer = RiskScorer()
        assert scorer.version == "1.0.0"
        assert not scorer.is_trained
        assert 'rug_risk' in scorer.component_weights
        assert 'dev_risk' in scorer.component_weights
    
    def test_heuristic_score(self):
        """Test heuristic scoring before training."""
        scorer = RiskScorer()
        
        X = np.random.rand(10, 20)
        scores = scorer.score(X[0])
        
        # Check all components present
        assert 'rug_risk' in scores
        assert 'dev_risk' in scores
        assert 'distribution_risk' in scores
        assert 'social_risk' in scores
        assert 'liquidity_risk' in scores
        assert 'composite_score' in scores
        
        # Check scores are in valid range
        for component, score in scores.items():
            assert 0 <= score <= 100
    
    def test_risk_flags(self):
        """Test risk flag generation."""
        scorer = RiskScorer()
        
        scores = {
            'rug_risk': 80.0,
            'dev_risk': 70.0,
            'distribution_risk': 75.0,
            'social_risk': 40.0,
            'liquidity_risk': 50.0
        }
        
        features = {
            'dev_holding_pct': 0.30,
            'holder_gini': 0.85,
            'liquidity_usd': 5000
        }
        
        flags = scorer.get_risk_flags(scores, features)
        
        # Should have multiple flags
        assert len(flags) > 0
        # Check for high dev or low liquidity flags (which should be present given the scores)
        assert any('dev' in flag.lower() or 'hold' in flag.lower() or 'liquidity' in flag.lower() for flag in flags)


class TestExplainerModel:
    """Test explainability model."""
    
    def test_initialization(self):
        """Test explainer initialization."""
        explainer = ExplainerModel()
        assert explainer.explainers == {}
        assert explainer.feature_names == []
    
    def test_fallback_explanation(self):
        """Test fallback explanation when SHAP unavailable."""
        explainer = ExplainerModel()
        explainer.feature_names = ['vol_24h', 'liquidity', 'gini', 'sentiment', 'mentions']
        
        X = np.array([[1000, 50000, 0.75, 0.6, 100]])
        
        explanation = explainer.explain(X, top_n=3)
        
        assert 'top_features' in explanation
        assert len(explanation['top_features']) == 3
        
        # Check feature structure
        for feature in explanation['top_features']:
            assert 'feature_name' in feature
            assert 'contribution' in feature
            assert 'direction' in feature
            assert 'value' in feature
            assert 'description' in feature
    
    def test_feature_descriptions(self):
        """Test feature description generation."""
        explainer = ExplainerModel()
        
        # Test various feature types
        desc = explainer._get_feature_description('volume_24h', 100000, 0.15)
        assert 'volume' in desc.lower()
        
        desc = explainer._get_feature_description('holder_gini', 0.85, -0.10)
        assert 'gini' in desc.lower()
        
        desc = explainer._get_feature_description('sentiment_score', 0.75, 0.08)
        assert 'sentiment' in desc.lower()


class TestCalibrationPipeline:
    """Test calibration pipeline."""
    
    def test_initialization(self):
        """Test calibration initialization."""
        calibrator = CalibrationPipeline(method='isotonic')
        assert calibrator.method == 'isotonic'
        assert not calibrator.is_fitted
    
    def test_ece_computation(self):
        """Test Expected Calibration Error calculation."""
        calibrator = CalibrationPipeline()
        
        # Perfect calibration
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
        y_pred = np.array([0.2, 0.3, 0.7, 0.8, 0.1, 0.9, 0.85, 0.25, 0.75, 0.8])
        
        ece = calibrator._compute_ece(y_true, y_pred, n_bins=5)
        
        # ECE should be relatively low for well-calibrated predictions
        assert 0 <= ece <= 0.5


class TestOnlineLearner:
    """Test online learning system."""
    
    def test_initialization(self):
        """Test online learner initialization."""
        learner = OnlineLearner(update_frequency='daily')
        assert learner.update_frequency == 'daily'
        assert learner.update_count == 0
        assert learner.last_update is None
    
    def test_should_update(self):
        """Test update scheduling."""
        learner = OnlineLearner(update_frequency='daily')
        
        # Should update initially
        assert learner.should_update()
        
        # After marking update, might not need to update immediately
        learner.last_update = datetime.utcnow()
        # Depends on time elapsed
    
    def test_performance_tracking(self):
        """Test performance logging and trends."""
        learner = OnlineLearner()
        
        # Log improving performance
        for i in range(10):
            learner.log_performance(0.6 + i * 0.02)
        
        trend = learner.get_performance_trend(window=5)
        
        assert 'current' in trend
        assert 'mean' in trend
        assert 'trend' in trend
        assert trend['trend'] == 'improving'
    
    def test_retrain_triggers(self):
        """Test retraining trigger logic."""
        learner = OnlineLearner()
        
        # Build baseline
        for _ in range(10):
            learner.log_performance(0.80)
        
        # No retrain needed for stable performance
        assert not learner.should_retrain(0.79)
        
        # Retrain needed for significant drop
        assert learner.should_retrain(0.65)  # >10% drop


class TestDriftDetector:
    """Test drift detection."""
    
    def test_initialization(self):
        """Test drift detector initialization."""
        detector = DriftDetector(window_size=1000, drift_threshold=0.05)
        assert detector.window_size == 1000
        assert detector.drift_threshold == 0.05
    
    def test_insufficient_data(self):
        """Test detection with insufficient data."""
        detector = DriftDetector()
        
        # Add few samples
        for _ in range(10):
            features = np.random.rand(5)
            detector.update(features, 0.5, 0.8, is_reference=True)
        
        report = detector.detect_drift()
        
        assert not report.drift_detected
        assert report.details['status'] == 'insufficient_data'
    
    def test_no_drift(self):
        """Test detection when no drift present."""
        from radarx.models.drift_detector import DriftType
        
        detector = DriftDetector()
        
        # Add reference data
        for _ in range(200):
            features = np.random.randn(5) * 1.0  # Mean 0, std 1
            detector.update(features, 0.5, 0.8, is_reference=True)
        
        # Add current data from same distribution
        for _ in range(100):
            features = np.random.randn(5) * 1.0
            detector.update(features, 0.5, 0.8, is_reference=False)
        
        report = detector.detect_drift(feature_names=['f1', 'f2', 'f3', 'f4', 'f5'])
        
        # Should not detect drift for same distribution
        # (though statistical tests might occasionally flag due to randomness)
        assert report.drift_type in [DriftType.NONE, DriftType.GRADUAL]
    
    def test_drift_summary(self):
        """Test drift summary generation."""
        detector = DriftDetector()
        
        # Need some history
        for _ in range(5):
            # Add minimal data to get reports
            for _ in range(150):
                features = np.random.rand(3)
                detector.update(features, 0.5, 0.8, is_reference=True)
            
            for _ in range(75):
                features = np.random.rand(3)
                detector.update(features, 0.5, 0.8, is_reference=False)
            
            detector.detect_drift()
        
        summary = detector.get_drift_summary()
        
        assert 'total_reports' in summary
        assert 'drift_count' in summary
        assert 'drift_rate' in summary


@pytest.fixture
def sample_features():
    """Generate sample features for testing."""
    return np.random.rand(100, 20)


@pytest.fixture
def sample_labels():
    """Generate sample labels for testing."""
    return {
        ("24h", "2x"): np.random.randint(0, 2, 100),
        ("24h", "5x"): np.random.randint(0, 2, 100),
        ("7d", "10x"): np.random.randint(0, 2, 100)
    }


def test_full_pipeline_integration(sample_features, sample_labels):
    """Test integration of multiple components."""
    # This tests that components can work together
    
    # 1. Train predictor (would normally train, but using dummy for test)
    predictor = ProbabilityPredictor()
    predictions = predictor.predict_proba(
        sample_features,
        horizons=["24h", "7d"],
        multipliers=["2x", "5x", "10x"]
    )
    
    # 2. Get risk scores
    scorer = RiskScorer()
    risk_scores = scorer.score(sample_features[0])
    
    # 3. Explain predictions
    explainer = ExplainerModel()
    explainer.feature_names = [f'feature_{i}' for i in range(20)]
    explanation = explainer.explain(sample_features[0:1], top_n=5)
    
    # 4. Drift detection
    detector = DriftDetector()
    for i in range(len(sample_features)):
        detector.update(sample_features[i], list(predictions.values())[0], 
                       is_reference=(i < 50))
    
    # All components should work without errors
    assert len(predictions) > 0
    assert risk_scores['composite_score'] >= 0
    assert len(explanation['top_features']) == 5
