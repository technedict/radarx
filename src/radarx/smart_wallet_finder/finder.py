"""
Core Smart Wallet Finder

Main orchestrator for discovering smart-money wallets given a token address.
Coordinates data fetching, signal computation, scoring, and ranking.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SmartWalletFinder:
    """
    Discovers probable smart-money wallets that traded a given token.

    Combines multiple detection signals:
    - Event timing (pre-pump/pre-dump detection)
    - Profitability metrics
    - Transaction graph analysis
    - Behavioral patterns
    - Smart-money markers
    - Risk filters
    """

    def __init__(
        self,
        data_fetcher=None,
        timing_detector=None,
        profitability_analyzer=None,
        graph_analyzer=None,
        behavioral_analyzer=None,
        risk_filter=None,
        scorer=None,
        explainer=None,
    ):
        """
        Initialize Smart Wallet Finder.

        Args:
            data_fetcher: Fetches on-chain and DEX data
            timing_detector: Detects event timing signals
            profitability_analyzer: Computes profitability metrics
            graph_analyzer: Analyzes transaction graph
            behavioral_analyzer: Detects behavioral patterns
            risk_filter: Filters wash trading and bots
            scorer: Scores and ranks wallets
            explainer: Generates explanations
        """
        from radarx.smart_wallet_finder.data_fetcher import DataFetcher
        from radarx.smart_wallet_finder.explainer import WalletExplainer
        from radarx.smart_wallet_finder.risk_filter import RiskFilter
        from radarx.smart_wallet_finder.scorer import WalletScorer
        from radarx.smart_wallet_finder.signals import (
            BehavioralAnalyzer,
            GraphAnalyzer,
            ProfitabilityAnalyzer,
            TimingSignalDetector,
        )

        self.data_fetcher = data_fetcher or DataFetcher()
        self.timing_detector = timing_detector or TimingSignalDetector()
        self.profitability_analyzer = profitability_analyzer or ProfitabilityAnalyzer()
        self.graph_analyzer = graph_analyzer or GraphAnalyzer()
        self.behavioral_analyzer = behavioral_analyzer or BehavioralAnalyzer()
        self.risk_filter = risk_filter or RiskFilter()
        self.scorer = scorer or WalletScorer()
        self.explainer = explainer or WalletExplainer()

    def find_smart_wallets(
        self,
        token_address: str,
        chain: str = "ethereum",
        window_days: int = 30,
        min_trade_size_usd: Optional[float] = None,
        min_holdings_usd: Optional[float] = None,
        include_internal_transfers: bool = False,
        top_k: int = 100,
        min_confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Find smart wallets that traded the given token.

        Args:
            token_address: Token contract address
            chain: Blockchain network
            window_days: Time window to analyze (default 30 days)
            min_trade_size_usd: Minimum trade size filter (USD)
            min_holdings_usd: Minimum holdings filter (USD)
            include_internal_transfers: Include internal transfers
            top_k: Number of top wallets to return
            min_confidence: Minimum confidence score threshold

        Returns:
            Dictionary containing:
                - ranked_wallets: List of wallet rankings with scores
                - metadata: Analysis metadata (token, chain, window)
                - summary_stats: Aggregate statistics
        """
        logger.info(
            f"Finding smart wallets for token {token_address} on {chain}, "
            f"window={window_days}d, top_k={top_k}"
        )

        # Create chain-specific data fetcher if not already set
        if self.data_fetcher.__class__.__name__ == "DataFetcher":
            from radarx.smart_wallet_finder.data_fetcher import DataFetcher

            self.data_fetcher = DataFetcher.create_for_chain(chain)

        # Step 1: Fetch data
        logger.info("Step 1: Fetching on-chain and price data")
        data = self._fetch_data(
            token_address=token_address,
            chain=chain,
            window_days=window_days,
            include_internal_transfers=include_internal_transfers,
        )

        # Step 2: Extract wallet addresses and trades
        logger.info("Step 2: Extracting wallets and matching trades")
        wallets_data = self._extract_wallets_and_trades(
            data=data,
            min_trade_size_usd=min_trade_size_usd,
            min_holdings_usd=min_holdings_usd,
        )

        # Step 3: Compute signals for each wallet
        logger.info(f"Step 3: Computing signals for {len(wallets_data)} wallets")
        wallet_signals = self._compute_signals(
            wallets_data=wallets_data,
            price_timeline=data.get("price_timeline", []),
            graph_data=data.get("graph_data", {}),
        )

        # Step 4: Apply risk filters
        logger.info("Step 4: Applying risk filters")
        filtered_signals = self._apply_risk_filters(wallet_signals)

        # Step 5: Score and rank wallets
        logger.info("Step 5: Scoring and ranking wallets")
        ranked_wallets = self._score_and_rank(
            filtered_signals,
            top_k=top_k,
            min_confidence=min_confidence,
        )

        # Step 6: Generate explanations
        logger.info("Step 6: Generating explanations")
        explained_wallets = self._generate_explanations(
            ranked_wallets,
            wallet_signals=wallet_signals,
        )

        # Step 7: Compile results
        result = {
            "token_address": token_address,
            "chain": chain,
            "analysis_window_days": window_days,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ranked_wallets": explained_wallets,
            "summary_stats": self._compute_summary_stats(explained_wallets),
            "metadata": {
                "total_wallets_analyzed": len(wallets_data),
                "wallets_passing_filters": len(filtered_signals),
                "wallets_returned": len(explained_wallets),
                "confidence_threshold": min_confidence,
            },
        }

        logger.info(f"Analysis complete. Returning {len(explained_wallets)} smart wallets")
        return result

    def get_wallet_profile(
        self,
        wallet_address: str,
        token_address: str,
        chain: str = "ethereum",
        window_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get detailed profile for a specific wallet and token.

        Args:
            wallet_address: Wallet address to profile
            token_address: Token address
            chain: Blockchain network
            window_days: Analysis window

        Returns:
            Detailed wallet profile with trades, metrics, and explanations
        """
        logger.info(f"Getting profile for wallet {wallet_address} on token {token_address}")

        # Fetch data for this specific wallet
        data = self._fetch_wallet_data(
            wallet_address=wallet_address,
            token_address=token_address,
            chain=chain,
            window_days=window_days,
        )

        # Compute all signals
        signals = self._compute_wallet_signals(
            wallet_data=data,
            price_timeline=data.get("price_timeline", []),
        )

        # Score wallet
        score = self.scorer.score_wallet(signals)

        # Generate explanation
        explanation = self.explainer.explain_wallet(
            wallet_address=wallet_address,
            signals=signals,
            score=score,
        )

        return {
            "wallet_address": wallet_address,
            "token_address": token_address,
            "chain": chain,
            "score": score,
            "trades": data.get("trades", []),
            "realized_roi": signals.get("profitability", {}).get("avg_roi", 0.0),
            "win_rate": signals.get("profitability", {}).get("win_rate", 0.0),
            "graph_neighbors": data.get("graph_neighbors", []),
            "explanation": explanation,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _fetch_data(
        self,
        token_address: str,
        chain: str,
        window_days: int,
        include_internal_transfers: bool,
    ) -> Dict[str, Any]:
        """Fetch all required data for analysis."""
        return self.data_fetcher.fetch_token_data(
            token_address=token_address,
            chain=chain,
            window_days=window_days,
            include_internal_transfers=include_internal_transfers,
        )

    def _fetch_wallet_data(
        self,
        wallet_address: str,
        token_address: str,
        chain: str,
        window_days: int,
    ) -> Dict[str, Any]:
        """Fetch data for a specific wallet."""
        return self.data_fetcher.fetch_wallet_token_data(
            wallet_address=wallet_address,
            token_address=token_address,
            chain=chain,
            window_days=window_days,
        )

    def _extract_wallets_and_trades(
        self,
        data: Dict[str, Any],
        min_trade_size_usd: Optional[float],
        min_holdings_usd: Optional[float],
    ) -> Dict[str, Dict[str, Any]]:
        """Extract and filter wallets from data."""
        from radarx.smart_wallet_finder.trade_matcher import TradeMatcher

        matcher = TradeMatcher()

        # Match buys and sells for each wallet
        wallets_data = matcher.match_trades(
            trades=data.get("trades", []),
            min_trade_size_usd=min_trade_size_usd,
            min_holdings_usd=min_holdings_usd,
        )

        return wallets_data

    def _compute_signals(
        self,
        wallets_data: Dict[str, Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
        graph_data: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Compute all signals for each wallet."""
        wallet_signals = {}

        for wallet_address, wallet_data in wallets_data.items():
            signals = {}

            # Timing signals
            signals["timing"] = self.timing_detector.detect(
                trades=wallet_data.get("trades", []),
                price_timeline=price_timeline,
            )

            # Profitability signals
            signals["profitability"] = self.profitability_analyzer.analyze(
                trades=wallet_data.get("trades", []),
            )

            # Graph signals
            signals["graph"] = self.graph_analyzer.analyze(
                wallet_address=wallet_address,
                graph_data=graph_data,
            )

            # Behavioral signals
            signals["behavioral"] = self.behavioral_analyzer.analyze(
                wallet_address=wallet_address,
                trades=wallet_data.get("trades", []),
            )

            wallet_signals[wallet_address] = signals

        return wallet_signals

    def _compute_wallet_signals(
        self,
        wallet_data: Dict[str, Any],
        price_timeline: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute signals for a single wallet."""
        signals = {}

        signals["timing"] = self.timing_detector.detect(
            trades=wallet_data.get("trades", []),
            price_timeline=price_timeline,
        )

        signals["profitability"] = self.profitability_analyzer.analyze(
            trades=wallet_data.get("trades", []),
        )

        signals["behavioral"] = self.behavioral_analyzer.analyze(
            wallet_address=wallet_data.get("wallet_address"),
            trades=wallet_data.get("trades", []),
        )

        return signals

    def _apply_risk_filters(
        self,
        wallet_signals: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Filter out risky wallets (wash trading, bots, etc.)."""
        filtered = {}

        for wallet_address, signals in wallet_signals.items():
            risk_score = self.risk_filter.compute_risk_score(signals)

            # Keep wallet if risk score is acceptable
            if risk_score < self.risk_filter.max_risk_threshold:
                filtered[wallet_address] = signals
                filtered[wallet_address]["risk_score"] = risk_score

        return filtered

    def _score_and_rank(
        self,
        wallet_signals: Dict[str, Dict[str, Any]],
        top_k: int,
        min_confidence: float,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Score and rank wallets by smart-money probability."""
        scored_wallets = []

        for wallet_address, signals in wallet_signals.items():
            score = self.scorer.score_wallet(signals)

            if score >= min_confidence:
                scored_wallets.append((wallet_address, score, signals))

        # Sort by score descending
        scored_wallets.sort(key=lambda x: x[1], reverse=True)

        return scored_wallets[:top_k]

    def find_smart_wallets_with_advanced_ml(
        self,
        token_address: str,
        chain: str = "ethereum",
        window_days: int = 30,
        top_k: int = 100,
        enable_granger: bool = True,
        enable_embeddings: bool = True,
        enable_counterfactual: bool = True,
    ) -> Dict[str, Any]:
        """
        Find smart wallets with advanced ML features enabled.

        Args:
            token_address: Token contract address
            chain: Blockchain network
            window_days: Analysis window in days
            top_k: Number of top wallets to return
            enable_granger: Enable Granger causality analysis
            enable_embeddings: Enable behavior embeddings
            enable_counterfactual: Enable counterfactual impact analysis

        Returns:
            Enhanced results with ML insights
        """
        from radarx.smart_wallet_finder.advanced_ml import (
            CounterfactualAnalyzer,
            GrangerCausalityAnalyzer,
            WalletBehaviorEmbedder,
        )

        # Get base results
        base_results = self.find_smart_wallets(
            token_address=token_address,
            chain=chain,
            window_days=window_days,
            top_k=top_k,
        )

        # Initialize advanced analyzers
        granger_analyzer = GrangerCausalityAnalyzer() if enable_granger else None
        embedder = WalletBehaviorEmbedder() if enable_embeddings else None
        counterfactual = CounterfactualAnalyzer() if enable_counterfactual else None

        # Enhance each wallet with advanced features
        enhanced_wallets = []

        for wallet in base_results["ranked_wallets"]:
            wallet_address = wallet["wallet_address"]

            # Get wallet trades from metadata
            wallet_trades = wallet.get("metadata", {}).get("trades", [])

            enhanced = wallet.copy()
            enhanced["advanced_features"] = {}

            # Granger causality
            if granger_analyzer and wallet_trades:
                price_timeline = base_results.get("metadata", {}).get("price_timeline", [])
                causality = granger_analyzer.analyze_wallet_price_causality(
                    wallet_trades, price_timeline
                )
                enhanced["advanced_features"]["granger_causality"] = causality

            # Behavior embeddings
            if embedder and wallet_trades:
                embedding = embedder.embed_wallet_sequence(wallet_trades)
                enhanced["advanced_features"]["behavior_embedding"] = embedding.tolist()

            # Counterfactual impact
            if counterfactual and wallet_trades:
                all_trades = base_results.get("metadata", {}).get("all_trades", [])
                price_timeline = base_results.get("metadata", {}).get("price_timeline", [])
                impact = counterfactual.estimate_wallet_impact(
                    wallet_trades, all_trades, price_timeline
                )
                enhanced["advanced_features"]["counterfactual_impact"] = impact

            enhanced_wallets.append(enhanced)

        return {
            **base_results,
            "ranked_wallets": enhanced_wallets,
            "ml_features_enabled": {
                "granger_causality": enable_granger,
                "behavior_embeddings": enable_embeddings,
                "counterfactual_analysis": enable_counterfactual,
            },
        }

    def _generate_explanations(
        self,
        ranked_wallets: List[Tuple[str, float, Dict[str, Any]]],
        wallet_signals: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate explanations for each wallet."""
        explained = []

        for rank, (wallet_address, score, signals) in enumerate(ranked_wallets, 1):
            explanation = self.explainer.explain_wallet(
                wallet_address=wallet_address,
                signals=signals,
                score=score,
            )

            explained.append(
                {
                    "rank": rank,
                    "wallet_address": wallet_address,
                    "smart_money_score": score,
                    "key_metrics": self._extract_key_metrics(signals),
                    "explanation": explanation,
                    "risk_score": signals.get("risk_score", 0.0),
                }
            )

        return explained

    def _extract_key_metrics(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from signals for display."""
        return {
            "win_rate": signals.get("profitability", {}).get("win_rate", 0.0),
            "realized_roi": signals.get("profitability", {}).get("avg_roi", 0.0),
            "trades_count": signals.get("profitability", {}).get("total_trades", 0),
            "early_entry_rate": signals.get("timing", {}).get("pre_pump_entry_rate", 0.0),
            "graph_centrality": signals.get("graph", {}).get("centrality_score", 0.0),
        }

    def _compute_summary_stats(
        self,
        explained_wallets: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute summary statistics across all wallets."""
        if not explained_wallets:
            return {}

        scores = [w["smart_money_score"] for w in explained_wallets]
        win_rates = [w["key_metrics"]["win_rate"] for w in explained_wallets]

        return {
            "avg_smart_money_score": sum(scores) / len(scores) if scores else 0.0,
            "median_smart_money_score": sorted(scores)[len(scores) // 2] if scores else 0.0,
            "avg_win_rate": sum(win_rates) / len(win_rates) if win_rates else 0.0,
            "total_smart_wallets": len(explained_wallets),
        }
