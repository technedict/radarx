"""
Wallet Scorer

Scores wallets by combining multiple signals into a calibrated probability
that the wallet is "smart money" for the given token.
"""

import logging
from typing import Any, Dict

import numpy as np

logger = logging.getLogger(__name__)


class WalletScorer:
    """
    Scores and ranks wallets using ensemble of signals.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
    ):
        """
        Initialize wallet scorer.

        Args:
            weights: Optional custom weights for signal categories
        """
        # Default weights for signal categories
        self.weights = weights or {
            "timing": 0.30,
            "profitability": 0.35,
            "graph": 0.15,
            "behavioral": 0.20,
        }

        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def score_wallet(
        self,
        signals: Dict[str, Any],
    ) -> float:
        """
        Compute smart-money probability score for wallet.

        Args:
            signals: Dictionary of computed signals

        Returns:
            Probability score between 0 and 1
        """
        # Extract and normalize each signal category
        timing_score = self._score_timing(signals.get("timing", {}))
        profitability_score = self._score_profitability(signals.get("profitability", {}))
        graph_score = self._score_graph(signals.get("graph", {}))
        behavioral_score = self._score_behavioral(signals.get("behavioral", {}))

        # Weighted combination
        combined_score = (
            self.weights["timing"] * timing_score
            + self.weights["profitability"] * profitability_score
            + self.weights["graph"] * graph_score
            + self.weights["behavioral"] * behavioral_score
        )

        # Apply calibration (simplified sigmoid)
        calibrated_score = self._calibrate(combined_score)

        return calibrated_score

    def _score_timing(self, timing_signals: Dict[str, Any]) -> float:
        """Score timing signals (0-1)."""
        if not timing_signals:
            return 0.0

        # Pre-pump entry rate (higher is better)
        pre_pump_rate = timing_signals.get("pre_pump_entry_rate", 0.0)

        # Pre-dump exit rate (higher is better)
        pre_dump_rate = timing_signals.get("pre_dump_exit_rate", 0.0)

        # Average entry timing (negative minutes before peak is better)
        avg_entry_timing = timing_signals.get("avg_entry_timing_minutes", 0.0)
        entry_timing_score = self._normalize_timing(avg_entry_timing)

        # Combine sub-scores
        score = 0.4 * pre_pump_rate + 0.4 * pre_dump_rate + 0.2 * entry_timing_score

        return min(1.0, max(0.0, score))

    def _score_profitability(self, profit_signals: Dict[str, Any]) -> float:
        """Score profitability signals (0-1)."""
        if not profit_signals:
            return 0.0

        # Win rate (already 0-1)
        win_rate = profit_signals.get("win_rate", 0.0)

        # Average ROI (normalize to 0-1)
        avg_roi = profit_signals.get("avg_roi", 0.0)
        roi_score = self._normalize_roi(avg_roi)

        # Sharpe ratio (normalize)
        sharpe = profit_signals.get("sharpe_ratio", 0.0)
        sharpe_score = self._normalize_sharpe(sharpe)

        # Combine sub-scores
        score = 0.4 * win_rate + 0.4 * roi_score + 0.2 * sharpe_score

        return min(1.0, max(0.0, score))

    def _score_graph(self, graph_signals: Dict[str, Any]) -> float:
        """Score graph signals (0-1)."""
        if not graph_signals:
            return 0.0

        # Centrality score (already 0-1)
        centrality = graph_signals.get("centrality_score", 0.0)

        # Clustering coefficient (already 0-1)
        clustering = graph_signals.get("clustering_coefficient", 0.0)

        # Connected smart wallets (normalize)
        smart_connections = graph_signals.get("connected_smart_wallets", 0)
        connection_score = min(1.0, smart_connections / 10.0)  # Cap at 10

        # Combine sub-scores
        score = 0.3 * centrality + 0.3 * clustering + 0.4 * connection_score

        return min(1.0, max(0.0, score))

    def _score_behavioral(self, behavioral_signals: Dict[str, Any]) -> float:
        """Score behavioral signals (0-1)."""
        if not behavioral_signals:
            return 0.0

        # DEX diversity (normalize)
        dex_diversity = behavioral_signals.get("dex_diversity", 0)
        diversity_score = min(1.0, dex_diversity / 5.0)  # Cap at 5

        # Gas pattern consistency (already 0-1)
        gas_consistency = behavioral_signals.get("gas_pattern_consistency", 0.0)

        # Pattern tags (score based on smart money indicators)
        pattern_tags = behavioral_signals.get("pattern_tags", [])
        pattern_score = self._score_patterns(pattern_tags)

        # Combine sub-scores
        score = 0.3 * diversity_score + 0.3 * gas_consistency + 0.4 * pattern_score

        return min(1.0, max(0.0, score))

    def _score_patterns(self, patterns: list) -> float:
        """Score behavioral patterns."""
        # Positive patterns
        positive = {"early_adopter", "diamond_hands"}
        # Neutral patterns
        neutral = {"swing_trader"}
        # Negative patterns
        negative = {"high_frequency"}

        score = 0.5  # Start neutral

        for pattern in patterns:
            if pattern in positive:
                score += 0.2
            elif pattern in negative:
                score -= 0.1

        return min(1.0, max(0.0, score))

    def _normalize_timing(self, minutes: float) -> float:
        """Normalize timing to 0-1 (earlier is better)."""
        # Negative minutes (before peak) are good
        # Convert to score where -60 minutes = 1.0, 0 minutes = 0.5, +60 = 0.0
        if minutes <= -60:
            return 1.0
        elif minutes >= 60:
            return 0.0
        else:
            return 0.5 - (minutes / 120.0)

    def _normalize_roi(self, roi: float) -> float:
        """Normalize ROI to 0-1."""
        # ROI of 1.0 (100%) = score 0.8
        # ROI of 5.0 (500%) = score 1.0
        if roi <= 0:
            return 0.0
        elif roi >= 5.0:
            return 1.0
        else:
            return roi / 6.25  # Scale so 5.0 -> 0.8

    def _normalize_sharpe(self, sharpe: float) -> float:
        """Normalize Sharpe ratio to 0-1."""
        # Sharpe of 2.0 = good, 4.0 = excellent
        if sharpe <= 0:
            return 0.0
        elif sharpe >= 4.0:
            return 1.0
        else:
            return sharpe / 4.0

    def _calibrate(self, score: float) -> float:
        """
        Apply calibration to convert combined score to probability.

        Uses a simple sigmoid-like function for calibration.
        In production, would use isotonic regression on validation data.
        """
        # Simple sigmoid: 1 / (1 + exp(-k * (x - 0.5)))
        # This maps [0, 1] -> [0, 1] with steeper response in middle
        k = 6  # Steepness parameter

        # Shift and scale
        x = (score - 0.5) * k

        # Sigmoid
        calibrated = 1.0 / (1.0 + np.exp(-x))

        return calibrated
