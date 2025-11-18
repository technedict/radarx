"""
Wallet Explainer

Generates human-readable explanations for wallet smart-money scores
with top contributing signals and time-aligned events.
"""

import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class WalletExplainer:
    """
    Generates explainability output for wallet scores.
    """

    def __init__(self):
        """Initialize wallet explainer."""
        pass

    def explain_wallet(
        self,
        wallet_address: str,
        signals: Dict[str, Any],
        score: float,
    ) -> Dict[str, Any]:
        """
        Generate explanation for wallet score.

        Args:
            wallet_address: Wallet address
            signals: Computed signals
            score: Smart-money probability score

        Returns:
            Dictionary with explanation including:
                - summary: High-level summary
                - top_signals: Top contributing signals
                - timeline: Time-aligned events
                - interpretation: Human-readable interpretation
        """
        # Extract top contributing signals
        top_signals = self._extract_top_signals(signals, score)

        # Generate summary
        summary = self._generate_summary(wallet_address, score, top_signals)

        # Create interpretation
        interpretation = self._interpret_score(score, signals)

        # Build timeline of key events
        timeline = self._build_timeline(signals)

        return {
            "summary": summary,
            "top_signals": top_signals,
            "interpretation": interpretation,
            "timeline": timeline,
            "confidence_level": self._get_confidence_level(score),
        }

    def _extract_top_signals(
        self,
        signals: Dict[str, Any],
        score: float,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Extract top contributing signals."""
        signal_contributions = []

        # Timing signals
        timing = signals.get("timing", {})
        if timing:
            signal_contributions.extend(
                [
                    {
                        "category": "timing",
                        "name": "Pre-pump entry rate",
                        "value": timing.get("pre_pump_entry_rate", 0),
                        "contribution": timing.get("pre_pump_entry_rate", 0) * 0.3,
                        "direction": "positive",
                        "description": f"{timing.get('pre_pump_buys', 0)} buys before price pumps",
                    },
                    {
                        "category": "timing",
                        "name": "Pre-dump exit rate",
                        "value": timing.get("pre_dump_exit_rate", 0),
                        "contribution": timing.get("pre_dump_exit_rate", 0) * 0.3,
                        "direction": "positive",
                        "description": f"{timing.get('pre_dump_sells', 0)} sells before price dumps",
                    },
                ]
            )

        # Profitability signals
        profitability = signals.get("profitability", {})
        if profitability:
            signal_contributions.extend(
                [
                    {
                        "category": "profitability",
                        "name": "Win rate",
                        "value": profitability.get("win_rate", 0),
                        "contribution": profitability.get("win_rate", 0) * 0.35,
                        "direction": "positive",
                        "description": f"{profitability.get('win_rate', 0):.1%} of trades profitable",
                    },
                    {
                        "category": "profitability",
                        "name": "Average ROI",
                        "value": profitability.get("avg_roi", 0),
                        "contribution": min(1.0, profitability.get("avg_roi", 0) / 5.0) * 0.35,
                        "direction": "positive",
                        "description": f"{profitability.get('avg_roi', 0):.1%} average return",
                    },
                ]
            )

        # Graph signals
        graph = signals.get("graph", {})
        if graph:
            signal_contributions.extend(
                [
                    {
                        "category": "graph",
                        "name": "Connected to smart wallets",
                        "value": graph.get("connected_smart_wallets", 0),
                        "contribution": min(1.0, graph.get("connected_smart_wallets", 0) / 10.0)
                        * 0.15,
                        "direction": "positive",
                        "description": f"Connected to {graph.get('connected_smart_wallets', 0)} known smart wallets",
                    },
                    {
                        "category": "graph",
                        "name": "Network centrality",
                        "value": graph.get("centrality_score", 0),
                        "contribution": graph.get("centrality_score", 0) * 0.15,
                        "direction": "positive",
                        "description": f"Centrality score {graph.get('centrality_score', 0):.2f}",
                    },
                ]
            )

        # Behavioral signals
        behavioral = signals.get("behavioral", {})
        if behavioral:
            patterns = behavioral.get("pattern_tags", [])
            if patterns:
                signal_contributions.append(
                    {
                        "category": "behavioral",
                        "name": "Trading patterns",
                        "value": len(patterns),
                        "contribution": 0.2,
                        "direction": "positive",
                        "description": f"Patterns: {', '.join(patterns)}",
                    }
                )

        # Sort by contribution and take top K
        signal_contributions.sort(key=lambda x: x["contribution"], reverse=True)
        return signal_contributions[:top_k]

    def _generate_summary(
        self,
        wallet_address: str,
        score: float,
        top_signals: List[Dict[str, Any]],
    ) -> str:
        """Generate high-level summary."""
        confidence = "high" if score > 0.7 else "medium" if score > 0.5 else "low"

        summary = (
            f"Wallet {wallet_address[:10]}... has a {score:.1%} probability "
            f"of being smart money ({confidence} confidence). "
        )

        if top_signals:
            top_signal = top_signals[0]
            summary += f"Primary indicator: {top_signal['name']} - {top_signal['description']}."

        return summary

    def _interpret_score(
        self,
        score: float,
        signals: Dict[str, Any],
    ) -> str:
        """Generate interpretation of the score."""
        if score >= 0.8:
            return (
                "Very high confidence this is a smart-money wallet. "
                "Strong evidence of superior timing, profitability, and strategic positioning."
            )
        elif score >= 0.6:
            return (
                "High confidence this is a smart-money wallet. "
                "Shows multiple positive indicators of sophisticated trading."
            )
        elif score >= 0.4:
            return (
                "Moderate confidence. Some positive indicators but mixed signals. "
                "May be skilled trader or beneficiary of favorable market conditions."
            )
        else:
            return (
                "Low confidence. Limited evidence of smart-money characteristics. "
                "Could be retail trader or opportunistic participant."
            )

    def _build_timeline(
        self,
        signals: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build timeline of key events."""
        timeline = []

        timing = signals.get("timing", {})
        profitability = signals.get("profitability", {})

        # Add key events
        if timing.get("pre_pump_buys", 0) > 0:
            timeline.append(
                {
                    "event_type": "early_entry",
                    "description": f"Entered position before {timing.get('pumps_detected', 0)} price pumps",
                    "impact": "positive",
                }
            )

        if timing.get("pre_dump_sells", 0) > 0:
            timeline.append(
                {
                    "event_type": "early_exit",
                    "description": f"Exited position before {timing.get('dumps_detected', 0)} price dumps",
                    "impact": "positive",
                }
            )

        if profitability.get("best_roi", 0) > 2.0:
            timeline.append(
                {
                    "event_type": "high_roi_trade",
                    "description": f"Best trade achieved {profitability.get('best_roi', 0):.1%} ROI",
                    "impact": "positive",
                }
            )

        return timeline

    def _get_confidence_level(self, score: float) -> str:
        """Get confidence level label."""
        if score >= 0.8:
            return "very_high"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        else:
            return "low"
