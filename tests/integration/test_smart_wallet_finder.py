"""
Integration tests for Smart Wallet Finder.

Tests the end-to-end pipeline with mock data.
"""

import pytest
from datetime import datetime, timedelta
from radarx.smart_wallet_finder.finder import SmartWalletFinder
from radarx.smart_wallet_finder.data_fetcher import DataFetcher
from radarx.smart_wallet_finder.signals import (
    TimingSignalDetector,
    ProfitabilityAnalyzer,
    GraphAnalyzer,
    BehavioralAnalyzer,
)
from radarx.smart_wallet_finder.scorer import WalletScorer
from radarx.smart_wallet_finder.risk_filter import RiskFilter
from radarx.smart_wallet_finder.explainer import WalletExplainer


class MockDataFetcher:
    """Mock data fetcher for testing."""

    def fetch_token_data(self, token_address, chain, window_days, include_internal_transfers):
        """Return mock token data."""
        now = datetime.utcnow()

        # Create mock trades
        trades = [
            {
                "side": "buy",
                "buyer": "0xwallet1",
                "seller": "0xother1",
                "timestamp": (now - timedelta(hours=48)).isoformat(),
                "amount_tokens": 100,
                "amount_usd": 1000,
                "dex": "uniswap",
                "gas_price": 50,
            },
            {
                "side": "sell",
                "seller": "0xwallet1",
                "buyer": "0xother2",
                "timestamp": (now - timedelta(hours=12)).isoformat(),
                "amount_tokens": 100,
                "amount_usd": 1500,
                "dex": "uniswap",
                "gas_price": 52,
            },
            {
                "side": "buy",
                "buyer": "0xwallet2",
                "seller": "0xother3",
                "timestamp": (now - timedelta(hours=24)).isoformat(),
                "amount_tokens": 50,
                "amount_usd": 500,
                "dex": "sushiswap",
                "gas_price": 48,
            },
        ]

        # Create price timeline with pump
        price_timeline = [
            {
                "timestamp": (now - timedelta(hours=60)).isoformat(),
                "price": 10.0,
            },
            {
                "timestamp": (now - timedelta(hours=36)).isoformat(),
                "price": 12.5,  # 25% pump
            },
            {
                "timestamp": (now - timedelta(hours=24)).isoformat(),
                "price": 11.0,
            },
            {
                "timestamp": (now - timedelta(hours=12)).isoformat(),
                "price": 15.0,  # Another pump
            },
            {
                "timestamp": now.isoformat(),
                "price": 14.0,
            },
        ]

        # Create graph data
        graph_data = {
            "nodes": {
                "0xwallet1": {"address": "0xwallet1"},
                "0xwallet2": {"address": "0xwallet2"},
                "0xother1": {"address": "0xother1"},
                "0xother2": {"address": "0xother2"},
            },
            "edges": [
                {"from": "0xother1", "to": "0xwallet1", "type": "trade"},
                {"from": "0xwallet1", "to": "0xother2", "type": "trade"},
                {"from": "0xother3", "to": "0xwallet2", "type": "trade"},
            ],
            "clusters": {"0xwallet1": 1, "0xwallet2": 1},
            "smart_wallets": set(),
        }

        return {
            "trades": trades,
            "price_timeline": price_timeline,
            "graph_data": graph_data,
            "token_metadata": {
                "address": token_address,
                "chain": chain,
                "name": "TestToken",
                "symbol": "TEST",
            },
            "chain": chain,
            "window_start": (now - timedelta(days=window_days)).isoformat(),
            "window_end": now.isoformat(),
        }

    def fetch_wallet_token_data(self, wallet_address, token_address, chain, window_days):
        """Return mock wallet-token data."""
        now = datetime.utcnow()

        trades = [
            {
                "side": "buy",
                "timestamp": (now - timedelta(hours=48)).isoformat(),
                "amount_tokens": 100,
                "amount_usd": 1000,
                "dex": "uniswap",
            },
            {
                "side": "sell",
                "timestamp": (now - timedelta(hours=12)).isoformat(),
                "amount_tokens": 100,
                "amount_usd": 1500,
                "dex": "uniswap",
            },
        ]

        price_timeline = [
            {
                "timestamp": (now - timedelta(hours=60)).isoformat(),
                "price": 10.0,
            },
            {
                "timestamp": now.isoformat(),
                "price": 15.0,
            },
        ]

        return {
            "wallet_address": wallet_address,
            "token_address": token_address,
            "trades": trades,
            "price_timeline": price_timeline,
            "graph_neighbors": ["0xother1", "0xother2"],
        }


class TestSmartWalletFinderIntegration:
    """Integration tests for Smart Wallet Finder."""

    def test_find_smart_wallets_end_to_end(self):
        """Test complete pipeline from data fetching to results."""
        # Initialize finder with mock data fetcher
        finder = SmartWalletFinder(data_fetcher=MockDataFetcher())

        # Find smart wallets
        result = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            top_k=10,
            min_confidence=0.0,  # Include all for testing
        )

        # Verify result structure
        assert "token_address" in result
        assert result["token_address"] == "0xtoken123"
        assert result["chain"] == "ethereum"
        assert result["analysis_window_days"] == 30

        assert "ranked_wallets" in result
        assert "summary_stats" in result
        assert "metadata" in result

        # Verify metadata
        assert result["metadata"]["total_wallets_analyzed"] >= 0
        assert result["metadata"]["wallets_passing_filters"] >= 0
        assert result["metadata"]["wallets_returned"] >= 0

        # If wallets found, verify structure
        if result["ranked_wallets"]:
            wallet = result["ranked_wallets"][0]

            assert "rank" in wallet
            assert "wallet_address" in wallet
            assert "smart_money_score" in wallet
            assert 0 <= wallet["smart_money_score"] <= 1

            assert "key_metrics" in wallet
            assert "win_rate" in wallet["key_metrics"]
            assert "realized_roi" in wallet["key_metrics"]
            assert "trades_count" in wallet["key_metrics"]

            assert "explanation" in wallet
            assert "summary" in wallet["explanation"]
            assert "top_signals" in wallet["explanation"]
            assert "interpretation" in wallet["explanation"]

            assert "risk_score" in wallet
            assert 0 <= wallet["risk_score"] <= 1

    def test_get_wallet_profile_end_to_end(self):
        """Test wallet profile retrieval."""
        finder = SmartWalletFinder(data_fetcher=MockDataFetcher())

        # Get wallet profile
        profile = finder.get_wallet_profile(
            wallet_address="0xwallet1",
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
        )

        # Verify profile structure
        assert "wallet_address" in profile
        assert profile["wallet_address"] == "0xwallet1"

        assert "token_address" in profile
        assert profile["token_address"] == "0xtoken123"

        assert "score" in profile
        assert 0 <= profile["score"] <= 1

        assert "trades" in profile
        assert "realized_roi" in profile
        assert "win_rate" in profile
        assert "graph_neighbors" in profile
        assert "explanation" in profile

    def test_pipeline_with_filters(self):
        """Test pipeline with various filters applied."""
        finder = SmartWalletFinder(data_fetcher=MockDataFetcher())

        # Test with minimum trade size filter
        result = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            min_trade_size_usd=100,
            top_k=10,
            min_confidence=0.0,
        )

        assert "ranked_wallets" in result

        # Test with minimum holdings filter
        result2 = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            min_holdings_usd=100,
            top_k=10,
            min_confidence=0.0,
        )

        assert "ranked_wallets" in result2

        # Test with confidence threshold
        result3 = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            top_k=10,
            min_confidence=0.8,  # High threshold
        )

        assert "ranked_wallets" in result3
        # May have fewer or no results due to high threshold
        assert len(result3["ranked_wallets"]) <= len(result["ranked_wallets"])

    def test_risk_filtering(self):
        """Test that risk filtering works."""
        finder = SmartWalletFinder(data_fetcher=MockDataFetcher())

        result = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            top_k=10,
            min_confidence=0.0,
        )

        # All returned wallets should have risk scores
        for wallet in result["ranked_wallets"]:
            assert "risk_score" in wallet
            # Risk score should be below threshold (0.7 default)
            assert wallet["risk_score"] < 0.7

    def test_explanation_generation(self):
        """Test that explanations are generated properly."""
        finder = SmartWalletFinder(data_fetcher=MockDataFetcher())

        result = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            top_k=10,
            min_confidence=0.0,
        )

        if result["ranked_wallets"]:
            wallet = result["ranked_wallets"][0]
            explanation = wallet["explanation"]

            # Verify explanation completeness
            assert isinstance(explanation["summary"], str)
            assert len(explanation["summary"]) > 0

            assert isinstance(explanation["top_signals"], list)

            for signal in explanation["top_signals"]:
                assert "category" in signal
                assert "name" in signal
                assert "value" in signal
                assert "contribution" in signal
                assert "direction" in signal
                assert "description" in signal

            assert isinstance(explanation["interpretation"], str)
            assert len(explanation["interpretation"]) > 0

            assert explanation["confidence_level"] in ["very_high", "high", "medium", "low"]

    def test_empty_data_handling(self):
        """Test handling of empty or minimal data."""

        class EmptyDataFetcher:
            def fetch_token_data(self, *args, **kwargs):
                return {
                    "trades": [],
                    "price_timeline": [],
                    "graph_data": {
                        "nodes": {},
                        "edges": [],
                        "clusters": {},
                        "smart_wallets": set(),
                    },
                    "token_metadata": {},
                    "chain": "ethereum",
                    "window_start": datetime.utcnow().isoformat(),
                    "window_end": datetime.utcnow().isoformat(),
                }

        finder = SmartWalletFinder(data_fetcher=EmptyDataFetcher())

        result = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            top_k=10,
            min_confidence=0.0,
        )

        # Should handle empty data gracefully
        assert "ranked_wallets" in result
        assert len(result["ranked_wallets"]) == 0
        assert result["metadata"]["total_wallets_analyzed"] == 0


class TestAPIIntegration:
    """Test API endpoint integration (mock)."""

    def test_api_schema_compatibility(self):
        """Test that finder output matches API schema expectations."""
        from radarx.smart_wallet_finder.schemas import SmartWalletFindResponse

        finder = SmartWalletFinder(data_fetcher=MockDataFetcher())

        result = finder.find_smart_wallets(
            token_address="0xtoken123",
            chain="ethereum",
            window_days=30,
            top_k=10,
            min_confidence=0.0,
        )

        # Validate against schema
        try:
            validated = SmartWalletFindResponse(**result)
            assert validated.token_address == "0xtoken123"
            assert validated.chain == "ethereum"
        except Exception as e:
            pytest.fail(f"Result does not match API schema: {e}")
