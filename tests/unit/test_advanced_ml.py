"""
Tests for Advanced ML Features
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from radarx.smart_wallet_finder.advanced_ml import (
    GrangerCausalityAnalyzer,
    WalletBehaviorEmbedder,
    CounterfactualAnalyzer,
)


class TestGrangerCausalityAnalyzer:
    """Test Granger causality analysis."""

    @pytest.fixture
    def analyzer(self):
        return GrangerCausalityAnalyzer(max_lag=5)

    @pytest.fixture
    def sample_trades(self):
        """Generate sample wallet trades."""
        trades = []
        base_time = datetime.utcnow()

        for i in range(20):
            trades.append(
                {
                    "timestamp": (base_time + timedelta(minutes=i * 15)).isoformat(),
                    "side": "buy" if i % 2 == 0 else "sell",
                    "amount_usd": 1000 + i * 100,
                    "amount_tokens": 10 + i,
                }
            )

        return trades

    @pytest.fixture
    def sample_price_timeline(self):
        """Generate sample price timeline."""
        timeline = []
        base_time = datetime.utcnow()
        base_price = 10.0

        for i in range(20):
            price = base_price + np.sin(i / 3) * 2  # Oscillating price
            timeline.append(
                {
                    "timestamp": (base_time + timedelta(minutes=i * 15)).isoformat(),
                    "price": price,
                }
            )

        return timeline

    def test_analyze_wallet_price_causality(self, analyzer, sample_trades, sample_price_timeline):
        """Test causality analysis."""
        result = analyzer.analyze_wallet_price_causality(sample_trades, sample_price_timeline)

        assert "has_causality" in result
        assert "optimal_lag" in result
        assert "p_values" in result
        assert "lead_score" in result
        assert isinstance(result["has_causality"], bool)
        assert isinstance(result["lead_score"], float)
        assert 0 <= result["lead_score"] <= 1

    def test_insufficient_data(self, analyzer):
        """Test with insufficient data."""
        trades = [{"timestamp": datetime.utcnow().isoformat(), "amount_usd": 100}]
        timeline = [{"timestamp": datetime.utcnow().isoformat(), "price": 10}]

        result = analyzer.analyze_wallet_price_causality(trades, timeline)

        assert result["has_causality"] is False
        assert result["lead_score"] == 0.0


class TestWalletBehaviorEmbedder:
    """Test wallet behavior embeddings."""

    @pytest.fixture
    def embedder(self):
        return WalletBehaviorEmbedder(embedding_dim=64)

    @pytest.fixture
    def sample_trades(self):
        """Generate sample trades."""
        trades = []
        for i in range(10):
            trades.append(
                {
                    "side": "buy" if i % 2 == 0 else "sell",
                    "amount_usd": 1000 + i * 100,
                    "price": 10 + i * 0.5,
                }
            )
        return trades

    def test_embed_wallet_sequence(self, embedder, sample_trades):
        """Test sequence embedding."""
        embedding = embedder.embed_wallet_sequence(sample_trades)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (64,)
        assert not np.isnan(embedding).any()

    def test_empty_trades(self, embedder):
        """Test with empty trades."""
        embedding = embedder.embed_wallet_sequence([])

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (64,)
        assert np.allclose(embedding, 0.0)

    def test_compute_similarity(self, embedder, sample_trades):
        """Test similarity computation."""
        embedding1 = embedder.embed_wallet_sequence(sample_trades[:5])
        embedding2 = embedder.embed_wallet_sequence(sample_trades[5:])

        similarity = embedder.compute_similarity(embedding1, embedding2)

        assert isinstance(similarity, float)
        assert 0 <= similarity <= 1


class TestCounterfactualAnalyzer:
    """Test counterfactual analysis."""

    @pytest.fixture
    def analyzer(self):
        return CounterfactualAnalyzer()

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for counterfactual analysis."""
        base_time = datetime.utcnow()

        wallet_trades = []
        all_trades = []

        # Wallet trades
        for i in range(5):
            trade = {
                "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                "amount_usd": 5000,
                "side": "buy" if i < 3 else "sell",
            }
            wallet_trades.append(trade)
            all_trades.append(trade)

        # Other trades
        for i in range(20):
            all_trades.append(
                {
                    "timestamp": (base_time + timedelta(hours=i, minutes=30)).isoformat(),
                    "amount_usd": 1000,
                    "side": "buy" if i % 2 == 0 else "sell",
                }
            )

        # Price timeline
        price_timeline = []
        for i in range(24):
            price_timeline.append(
                {
                    "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                    "price": 10 + i * 0.5,
                }
            )

        return wallet_trades, all_trades, price_timeline

    def test_estimate_wallet_impact(self, analyzer, sample_data):
        """Test wallet impact estimation."""
        wallet_trades, all_trades, price_timeline = sample_data

        result = analyzer.estimate_wallet_impact(wallet_trades, all_trades, price_timeline)

        assert "impact_score" in result
        assert "volume_contribution" in result
        assert "price_influence" in result
        assert "counterfactual_returns" in result
        assert "interpretation" in result

        assert isinstance(result["impact_score"], float)
        assert 0 <= result["impact_score"] <= 1
        assert 0 <= result["volume_contribution"] <= 1

    def test_empty_trades(self, analyzer):
        """Test with empty trades."""
        result = analyzer.estimate_wallet_impact([], [], [])

        assert result["impact_score"] == 0.0
        assert result["volume_contribution"] == 0.0
