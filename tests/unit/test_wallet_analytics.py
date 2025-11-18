"""
Tests for Wallet Analytics Engine
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from radarx.wallet import WalletAnalyzer, BehaviorDetector, WalletRanker, RelatedWalletFinder


class TestWalletAnalyzer:
    """Tests for WalletAnalyzer."""

    def test_calculate_win_rate(self):
        """Test win rate calculation."""
        analyzer = WalletAnalyzer()

        trades = [
            {"timestamp": datetime.utcnow().isoformat() + "Z", "pnl": 100},
            {"timestamp": datetime.utcnow().isoformat() + "Z", "pnl": -50},
            {"timestamp": datetime.utcnow().isoformat() + "Z", "pnl": 200},
        ]

        result = analyzer.calculate_win_rate(trades, timeframes=["all"])
        assert "all" in result
        assert result["all"]["win_rate"] == pytest.approx(2 / 3, 0.01)
        assert result["all"]["profitable_trades"] == 2
        assert result["all"]["total_trades"] == 3

    def test_calculate_realized_pnl(self):
        """Test realized PnL calculation."""
        analyzer = WalletAnalyzer()

        trades = [
            {"pnl": 100, "buy_amount": 1000, "sell_amount": 1100},
            {"pnl": -50, "buy_amount": 500, "sell_amount": 450},
            {"pnl": 200, "buy_amount": 2000, "sell_amount": 2200},
        ]

        result = analyzer.calculate_realized_pnl(trades)
        assert result["total"] == 250
        assert result["average"] == pytest.approx(83.33, 0.01)
        assert result["best"] == 200
        assert result["worst"] == -50

    def test_calculate_performance_metrics(self):
        """Test performance metrics calculation."""
        analyzer = WalletAnalyzer()

        now = datetime.utcnow()
        trades = [
            {
                "timestamp": now.isoformat() + "Z",
                "pnl": 100,
                "buy_timestamp": (now - timedelta(hours=2)).isoformat() + "Z",
                "sell_timestamp": now.isoformat() + "Z",
            },
            {
                "timestamp": (now + timedelta(days=1)).isoformat() + "Z",
                "pnl": -50,
                "buy_timestamp": now.isoformat() + "Z",
                "sell_timestamp": (now + timedelta(hours=1)).isoformat() + "Z",
            },
        ]

        result = analyzer.calculate_performance_metrics(trades)
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result
        assert "avg_hold_time_hours" in result
        assert "trade_frequency_per_day" in result

    def test_get_token_breakdown(self):
        """Test token-level breakdown."""
        analyzer = WalletAnalyzer()

        trades = [
            {"token": "0xAAA", "pnl": 100},
            {"token": "0xAAA", "pnl": -20},
            {"token": "0xBBB", "pnl": 50},
        ]

        result = analyzer.get_token_breakdown(trades)
        assert len(result) == 2
        assert result[0]["token"] == "0xAAA"  # Highest PnL first
        assert result[0]["total_pnl"] == 80
        assert result[0]["trades"] == 2
        assert result[0]["wins"] == 1


class TestBehaviorDetector:
    """Tests for BehaviorDetector."""

    def test_detect_patterns(self):
        """Test pattern detection."""
        detector = BehaviorDetector()

        trades = [
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "pnl": 100,
                "action": "buy",
                "buy_timestamp": (datetime.utcnow() - timedelta(days=5)).isoformat() + "Z",
                "sell_timestamp": datetime.utcnow().isoformat() + "Z",
            }
        ]

        patterns = detector.detect_patterns(trades, "0x123")
        assert isinstance(patterns, list)

    def test_check_smart_money(self):
        """Test smart money detection."""
        detector = BehaviorDetector()

        # Very high win rate trades (80%+)
        trades = [
            {"pnl": 100},
            {"pnl": 50},
            {"pnl": 75},
            {"pnl": 80},
            {"pnl": -10},  # Only 1 loss out of 5
        ]

        is_smart = detector.is_smart_money(
            trades, threshold=0.4
        )  # Lower threshold for scaled confidence
        assert is_smart  # 80% win rate should qualify

    def test_wash_trading_score(self):
        """Test wash trading detection."""
        detector = BehaviorDetector()

        # Alternating buy/sell pattern (wash trading)
        trades = [
            {"token": "0xAAA", "action": "buy"},
            {"token": "0xAAA", "action": "sell"},
            {"token": "0xAAA", "action": "buy"},
            {"token": "0xAAA", "action": "sell"},
            {"token": "0xAAA", "action": "buy"},
            {"token": "0xAAA", "action": "sell"},
        ]

        score = detector.calculate_wash_trading_score(trades)
        assert score > 0.5  # High alternation = wash trading


class TestWalletRanker:
    """Tests for WalletRanker."""

    def test_add_and_rank_wallets(self):
        """Test adding and ranking wallets."""
        ranker = WalletRanker()

        ranker.add_wallet("0x111", win_rate=0.7, pnl=10000, trades=50)
        ranker.add_wallet("0x222", win_rate=0.8, pnl=15000, trades=60)
        ranker.add_wallet("0x333", win_rate=0.6, pnl=8000, trades=40)

        rankings = ranker.get_rankings(metric="win_rate")
        assert len(rankings) == 3
        assert rankings[0]["address"] == "0x222"  # Highest win rate
        assert rankings[0]["rank"] == 1

    def test_get_wallet_rank(self):
        """Test getting specific wallet rank."""
        ranker = WalletRanker()

        ranker.add_wallet("0x111", win_rate=0.7, pnl=10000, trades=50)
        ranker.add_wallet("0x222", win_rate=0.8, pnl=15000, trades=60)

        rank_info = ranker.get_wallet_rank("0x111", metric="win_rate")
        assert rank_info is not None
        assert rank_info["rank"] == 2
        assert 0 <= rank_info["percentile"] <= 100

    def test_chain_specific_ranking(self):
        """Test chain-specific rankings."""
        ranker = WalletRanker()

        ranker.add_wallet("0x111", win_rate=0.7, pnl=10000, trades=50, chain="ethereum")
        ranker.add_wallet("0x222", win_rate=0.8, pnl=15000, trades=60, chain="ethereum")
        ranker.add_wallet("0x333", win_rate=0.9, pnl=20000, trades=70, chain="bsc")

        eth_rankings = ranker.get_rankings(chain="ethereum")
        assert len(eth_rankings) == 2

        bsc_rankings = ranker.get_rankings(chain="bsc")
        assert len(bsc_rankings) == 1


class TestRelatedWalletFinder:
    """Tests for RelatedWalletFinder."""

    def test_find_by_fund_flow(self):
        """Test fund flow analysis."""
        finder = RelatedWalletFinder()

        transfers = [
            {
                "from": "0x111",
                "to": "0x222",
                "amount": 1000,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
            {
                "from": "0x222",
                "to": "0x333",
                "amount": 500,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            },
        ]

        related = finder.find_by_fund_flow("0x111", transfers, max_depth=2)
        assert len(related) > 0
        assert any(r["wallet"] == "0x222" for r in related)

    def test_find_by_pattern_similarity(self):
        """Test pattern similarity detection."""
        finder = RelatedWalletFinder()

        features = {
            "0x111": np.array([1.0, 0.5, 0.8]),
            "0x222": np.array([0.9, 0.6, 0.7]),  # Similar
            "0x333": np.array([0.1, 0.1, 0.1]),  # Different
        }

        related = finder.find_by_pattern_similarity("0x111", features, threshold=0.8)
        assert len(related) > 0
        assert related[0]["wallet"] == "0x222"

    def test_find_by_token_overlap(self):
        """Test token overlap detection."""
        finder = RelatedWalletFinder()

        wallet_tokens = {
            "0x111": {"tokenA", "tokenB", "tokenC", "tokenD"},
            "0x222": {"tokenA", "tokenB", "tokenC"},  # 3 overlap
            "0x333": {"tokenX", "tokenY"},  # No overlap
        }

        related = finder.find_by_token_overlap("0x111", wallet_tokens, min_overlap=2)
        assert len(related) == 1
        assert related[0]["wallet"] == "0x222"
        assert related[0]["overlap_count"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
