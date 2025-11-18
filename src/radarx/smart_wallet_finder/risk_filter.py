"""
Risk Filter

Filters out wallets that show signs of wash trading, bot activity,
or other suspicious patterns.
"""

import logging
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class RiskFilter:
    """
    Filters risky wallets using wash trading and bot detection.
    """

    def __init__(
        self,
        max_risk_threshold: float = 0.7,
    ):
        """
        Initialize risk filter.

        Args:
            max_risk_threshold: Maximum acceptable risk score (0-1)
        """
        self.max_risk_threshold = max_risk_threshold

    def compute_risk_score(
        self,
        signals: Dict[str, Any],
    ) -> float:
        """
        Compute composite risk score for a wallet.

        Args:
            signals: Wallet signals

        Returns:
            Risk score between 0 (safe) and 1 (high risk)
        """
        risk_scores = []

        # Wash trading risk
        wash_risk = self._detect_wash_trading(signals)
        risk_scores.append(wash_risk)

        # Bot-like behavior risk
        bot_risk = self._detect_bot_behavior(signals)
        risk_scores.append(bot_risk)

        # Circular trade risk
        circular_risk = self._detect_circular_trades(signals)
        risk_scores.append(circular_risk)

        # Rapid trade risk
        rapid_risk = self._detect_rapid_trades(signals)
        risk_scores.append(rapid_risk)

        # Combine risks (maximum risk)
        composite_risk = max(risk_scores) if risk_scores else 0.0

        return composite_risk

    def _detect_wash_trading(
        self,
        signals: Dict[str, Any],
    ) -> float:
        """
        Detect wash trading patterns.

        Indicators:
        - High trade frequency with no net position change
        - Simultaneous buy/sell patterns
        - Self-trading indicators
        """
        profitability = signals.get("profitability", {})
        behavioral = signals.get("behavioral", {})

        # High frequency with zero ROI is suspicious
        trade_frequency = behavioral.get("trade_frequency_per_day", 0)
        avg_roi = profitability.get("avg_roi", 0)

        if trade_frequency > 20 and abs(avg_roi) < 0.01:
            return 0.8  # High wash trading risk

        # Check for balanced buy/sell volumes
        timing = signals.get("timing", {})
        buys = timing.get("total_buys", 0)
        sells = timing.get("total_sells", 0)

        if buys > 0 and sells > 0:
            balance_ratio = min(buys, sells) / max(buys, sells)
            if balance_ratio > 0.95 and trade_frequency > 10:
                return 0.7  # Suspicious balance

        return 0.0

    def _detect_bot_behavior(
        self,
        signals: Dict[str, Any],
    ) -> float:
        """
        Detect bot-like trading behavior.

        Indicators:
        - Very regular trading intervals
        - Identical gas prices
        - No variation in trade sizes
        """
        behavioral = signals.get("behavioral", {})

        # Very high gas consistency suggests bot
        gas_consistency = behavioral.get("gas_pattern_consistency", 0)
        if gas_consistency > 0.95:
            return 0.6

        # Extremely high trade frequency
        trade_frequency = behavioral.get("trade_frequency_per_day", 0)
        if trade_frequency > 50:
            return 0.7

        return 0.0

    def _detect_circular_trades(
        self,
        signals: Dict[str, Any],
    ) -> float:
        """
        Detect circular trade patterns.

        Indicators:
        - High clustering coefficient with low centrality
        - Trades with same counterparties repeatedly
        """
        graph = signals.get("graph", {})

        clustering = graph.get("clustering_coefficient", 0)
        centrality = graph.get("centrality_score", 0)

        # High clustering with low centrality suggests closed loop
        if clustering > 0.8 and centrality < 0.2:
            return 0.7

        return 0.0

    def _detect_rapid_trades(
        self,
        signals: Dict[str, Any],
    ) -> float:
        """
        Detect rapid-fire trading (potential MEV bot).

        Indicators:
        - Very short holding durations
        - High frequency
        """
        profitability = signals.get("profitability", {})
        behavioral = signals.get("behavioral", {})

        avg_duration = profitability.get("avg_holding_duration_hours", 0)
        trade_frequency = behavioral.get("trade_frequency_per_day", 0)

        # Average holding less than 1 hour with high frequency
        if avg_duration < 1.0 and trade_frequency > 20:
            return 0.6

        return 0.0
