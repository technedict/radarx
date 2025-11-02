"""
Drift Detector

Detects concept drift in features and model performance over time.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available. Some drift tests unavailable.")


class DriftType(Enum):
    """Types of drift that can be detected."""
    NONE = "none"
    GRADUAL = "gradual"
    SUDDEN = "sudden"
    INCREMENTAL = "incremental"
    RECURRING = "recurring"


@dataclass
class DriftReport:
    """Drift detection report."""
    drift_detected: bool
    drift_type: DriftType
    confidence: float
    affected_features: List[str]
    performance_change: Optional[float]
    timestamp: datetime
    details: Dict


class DriftDetector:
    """
    Concept drift detector for monitoring data and model changes.
    
    Detects:
    - Feature distribution drift (covariate shift)
    - Model performance drift
    - Prediction drift
    
    Uses multiple detection methods:
    - ADWIN (Adaptive Windowing)
    - Page-Hinkley Test
    - Kolmogorov-Smirnov Test
    - Performance degradation tracking
    """
    
    def __init__(self, window_size: int = 1000, 
                 drift_threshold: float = 0.05,
                 warning_threshold: float = 0.03):
        """
        Initialize drift detector.
        
        Args:
            window_size: Size of reference window for comparison
            drift_threshold: Threshold for declaring drift
            warning_threshold: Threshold for drift warning
        """
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.warning_threshold = warning_threshold
        
        # Reference windows
        self.reference_features = deque(maxlen=window_size)
        self.reference_predictions = deque(maxlen=window_size)
        self.reference_performance = deque(maxlen=window_size)
        
        # Current windows
        self.current_features = deque(maxlen=window_size // 2)
        self.current_predictions = deque(maxlen=window_size // 2)
        self.current_performance = deque(maxlen=window_size // 2)
        
        # Drift history
        self.drift_history = []
        
        # Feature statistics
        self.feature_names = []
        self.feature_means = {}
        self.feature_stds = {}
        
    def update(self, features: np.ndarray, prediction: float, 
              performance: Optional[float] = None,
              is_reference: bool = False):
        """
        Update detector with new sample.
        
        Args:
            features: Feature vector
            prediction: Model prediction
            performance: Performance metric (e.g., accuracy, loss)
            is_reference: Whether this is reference (baseline) data
        """
        if is_reference:
            self.reference_features.append(features)
            self.reference_predictions.append(prediction)
            if performance is not None:
                self.reference_performance.append(performance)
        else:
            self.current_features.append(features)
            self.current_predictions.append(prediction)
            if performance is not None:
                self.current_performance.append(performance)
    
    def detect_drift(self, feature_names: Optional[List[str]] = None) -> DriftReport:
        """
        Detect if drift has occurred.
        
        Args:
            feature_names: Optional list of feature names
            
        Returns:
            DriftReport with detection results
        """
        if len(self.reference_features) < 100 or len(self.current_features) < 50:
            return DriftReport(
                drift_detected=False,
                drift_type=DriftType.NONE,
                confidence=0.0,
                affected_features=[],
                performance_change=None,
                timestamp=datetime.utcnow(),
                details={'status': 'insufficient_data'}
            )
        
        self.feature_names = feature_names or self.feature_names
        
        # Test 1: Feature distribution drift
        feature_drift, affected_features, feature_confidence = self._detect_feature_drift()
        
        # Test 2: Prediction drift
        prediction_drift, pred_confidence = self._detect_prediction_drift()
        
        # Test 3: Performance drift
        performance_drift, perf_change, perf_confidence = self._detect_performance_drift()
        
        # Combine results
        drift_detected = feature_drift or prediction_drift or performance_drift
        
        # Determine drift type
        if not drift_detected:
            drift_type = DriftType.NONE
        elif performance_drift and perf_change < -0.10:
            drift_type = DriftType.SUDDEN
        elif feature_drift and len(affected_features) > len(self.feature_names) * 0.5:
            drift_type = DriftType.SUDDEN
        else:
            drift_type = DriftType.GRADUAL
        
        # Overall confidence (max of individual confidences)
        confidence = max(feature_confidence, pred_confidence, perf_confidence)
        
        report = DriftReport(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=confidence,
            affected_features=affected_features,
            performance_change=perf_change,
            timestamp=datetime.utcnow(),
            details={
                'feature_drift': feature_drift,
                'prediction_drift': prediction_drift,
                'performance_drift': performance_drift,
                'n_affected_features': len(affected_features)
            }
        )
        
        # Log to history
        self.drift_history.append(report)
        if len(self.drift_history) > 100:
            self.drift_history = self.drift_history[-100:]
        
        if drift_detected:
            logger.warning(f"Drift detected! Type: {drift_type}, Confidence: {confidence:.2f}")
        
        return report
    
    def _detect_feature_drift(self) -> Tuple[bool, List[str], float]:
        """
        Detect drift in feature distributions using KS test.
        
        Returns:
            (drift_detected, affected_features, confidence)
        """
        if not HAS_SCIPY:
            return False, [], 0.0
        
        ref_features = np.array(list(self.reference_features))
        cur_features = np.array(list(self.current_features))
        
        n_features = ref_features.shape[1]
        affected_features = []
        p_values = []
        
        for i in range(n_features):
            ref_col = ref_features[:, i]
            cur_col = cur_features[:, i]
            
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_col, cur_col)
            p_values.append(p_value)
            
            # Drift detected if p-value < threshold
            if p_value < self.drift_threshold:
                feature_name = (self.feature_names[i] 
                              if i < len(self.feature_names) 
                              else f"feature_{i}")
                affected_features.append(feature_name)
        
        drift_detected = len(affected_features) > 0
        
        # Confidence based on minimum p-value
        confidence = 1 - min(p_values) if p_values else 0.0
        
        return drift_detected, affected_features, confidence
    
    def _detect_prediction_drift(self) -> Tuple[bool, float]:
        """
        Detect drift in model predictions using KS test.
        
        Returns:
            (drift_detected, confidence)
        """
        if not HAS_SCIPY:
            return False, 0.0
        
        ref_preds = np.array(list(self.reference_predictions))
        cur_preds = np.array(list(self.current_predictions))
        
        statistic, p_value = stats.ks_2samp(ref_preds, cur_preds)
        
        drift_detected = p_value < self.drift_threshold
        confidence = 1 - p_value
        
        return drift_detected, confidence
    
    def _detect_performance_drift(self) -> Tuple[bool, Optional[float], float]:
        """
        Detect drift in model performance.
        
        Returns:
            (drift_detected, performance_change, confidence)
        """
        if not self.reference_performance or not self.current_performance:
            return False, None, 0.0
        
        ref_perf = np.array(list(self.reference_performance))
        cur_perf = np.array(list(self.current_performance))
        
        ref_mean = ref_perf.mean()
        cur_mean = cur_perf.mean()
        
        # Performance change (negative = degradation)
        perf_change = cur_mean - ref_mean
        
        # Drift if performance degraded significantly
        drift_detected = perf_change < -self.drift_threshold
        
        # Confidence based on magnitude of change
        confidence = min(abs(perf_change) / self.drift_threshold, 1.0)
        
        return drift_detected, perf_change, confidence
    
    def page_hinkley_test(self, values: List[float], 
                         delta: float = 0.005,
                         lambda_: float = 50) -> bool:
        """
        Page-Hinkley test for detecting changes in mean.
        
        Args:
            values: Sequence of values to test
            delta: Magnitude of changes to detect
            lambda_: Detection threshold
            
        Returns:
            True if change detected
        """
        if len(values) < 30:
            return False
        
        mean = np.mean(values[:30])
        cumsum = 0
        min_cumsum = 0
        
        for value in values[30:]:
            cumsum += value - mean - delta
            min_cumsum = min(min_cumsum, cumsum)
            
            if cumsum - min_cumsum > lambda_:
                return True
        
        return False
    
    def reset_reference(self):
        """
        Reset reference windows to current data.
        Use after retraining or when drift is confirmed and handled.
        """
        # Move current to reference
        self.reference_features = deque(self.current_features, 
                                       maxlen=self.window_size)
        self.reference_predictions = deque(self.current_predictions,
                                          maxlen=self.window_size)
        self.reference_performance = deque(self.current_performance,
                                          maxlen=self.window_size)
        
        # Clear current
        self.current_features.clear()
        self.current_predictions.clear()
        self.current_performance.clear()
        
        logger.info("Reference windows reset")
    
    def get_drift_summary(self) -> Dict:
        """
        Get summary of drift detection history.
        
        Returns:
            Dict with drift statistics
        """
        if not self.drift_history:
            return {'status': 'no_history'}
        
        recent = self.drift_history[-20:]  # Last 20 reports
        
        drift_count = sum(1 for r in recent if r.drift_detected)
        drift_rate = drift_count / len(recent)
        
        # Most common drift type
        drift_types = [r.drift_type for r in recent if r.drift_detected]
        most_common_type = max(set(drift_types), key=drift_types.count) if drift_types else DriftType.NONE
        
        # Average confidence when drift detected
        confidences = [r.confidence for r in recent if r.drift_detected]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        return {
            'total_reports': len(recent),
            'drift_count': drift_count,
            'drift_rate': drift_rate,
            'most_common_type': most_common_type.value,
            'avg_confidence': avg_confidence,
            'last_drift': recent[-1].timestamp if drift_count > 0 else None
        }
