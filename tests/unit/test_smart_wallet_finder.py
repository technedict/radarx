"""
Unit tests for Smart Wallet Finder components.
"""

import pytest
from datetime import datetime, timedelta
from radarx.smart_wallet_finder.signals import (
    TimingSignalDetector,
    ProfitabilityAnalyzer,
    GraphAnalyzer,
    BehavioralAnalyzer,
)
from radarx.smart_wallet_finder.scorer import WalletScorer
from radarx.smart_wallet_finder.risk_filter import RiskFilter
from radarx.smart_wallet_finder.trade_matcher import TradeMatcher
from radarx.smart_wallet_finder.explainer import WalletExplainer


class TestTimingSignalDetector:
    """Test timing signal detection."""
    
    def test_detect_pre_pump_buys(self):
        """Test detection of buys before price pumps."""
        detector = TimingSignalDetector(
            pre_pump_window_minutes=60,
            pump_threshold_pct=20.0,
        )
        
        # Create mock trades
        now = datetime.utcnow()
        trades = [
            {
                "side": "buy",
                "timestamp": (now - timedelta(minutes=30)).isoformat(),
                "amount_usd": 1000,
            }
        ]
        
        # Create price timeline with pump
        price_timeline = [
            {
                "timestamp": (now - timedelta(minutes=60)).isoformat(),
                "price": 1.0,
            },
            {
                "timestamp": now.isoformat(),
                "price": 1.25,  # 25% pump
            }
        ]
        
        signals = detector.detect(trades, price_timeline)
        
        assert signals["pre_pump_buys"] >= 0
        assert signals["total_buys"] == 1
        assert signals["pumps_detected"] >= 0
    
    def test_empty_trades(self):
        """Test with empty trades."""
        detector = TimingSignalDetector()
        signals = detector.detect([], [])
        
        assert signals["pre_pump_buys"] == 0
        assert signals["total_buys"] == 0


class TestProfitabilityAnalyzer:
    """Test profitability analysis."""
    
    def test_analyze_profitable_trades(self):
        """Test analysis of profitable trades."""
        analyzer = ProfitabilityAnalyzer()
        
        trades = [
            {"pnl": 100, "roi": 0.5},
            {"pnl": 200, "roi": 1.0},
            {"pnl": -50, "roi": -0.25},
        ]
        
        metrics = analyzer.analyze(trades)
        
        assert metrics["total_trades"] == 3
        assert metrics["profitable_trades"] == 2
        assert metrics["win_rate"] == 2/3
        assert metrics["avg_roi"] > 0
    
    def test_empty_trades(self):
        """Test with empty trades."""
        analyzer = ProfitabilityAnalyzer()
        metrics = analyzer.analyze([])
        
        assert metrics["total_trades"] == 0
        assert metrics["win_rate"] == 0.0


class TestGraphAnalyzer:
    """Test graph analysis."""
    
    def test_calculate_centrality(self):
        """Test centrality calculation."""
        analyzer = GraphAnalyzer()
        
        graph_data = {
            "nodes": {"w1": {}, "w2": {}, "w3": {}},
            "edges": [
                {"from": "w1", "to": "w2"},
                {"from": "w1", "to": "w3"},
            ],
            "clusters": {"w1": 1},
            "smart_wallets": set(),
        }
        
        metrics = analyzer.analyze("w1", graph_data)
        
        assert metrics["centrality_score"] >= 0
        assert metrics["total_connections"] == 2
    
    def test_empty_graph(self):
        """Test with empty graph."""
        analyzer = GraphAnalyzer()
        metrics = analyzer.analyze("w1", {})
        
        assert metrics["centrality_score"] == 0.0
        assert metrics["total_connections"] == 0


class TestBehavioralAnalyzer:
    """Test behavioral pattern detection."""
    
    def test_detect_early_adopter(self):
        """Test early adopter pattern detection."""
        analyzer = BehavioralAnalyzer()
        
        trades = [
            {
                "timestamp": datetime.utcnow().isoformat(),
                "token_age_hours": 12,  # Within 24h of launch
                "dex": "uniswap",
            },
            {
                "timestamp": datetime.utcnow().isoformat(),
                "token_age_hours": 6,
                "dex": "uniswap",
            },
        ]
        
        metrics = analyzer.analyze("wallet1", trades)
        
        assert metrics["dex_diversity"] >= 0
        assert "pattern_tags" in metrics
    
    def test_empty_trades(self):
        """Test with empty trades."""
        analyzer = BehavioralAnalyzer()
        metrics = analyzer.analyze("wallet1", [])
        
        assert metrics["trade_frequency_per_day"] == 0.0
        assert metrics["pattern_tags"] == []


class TestWalletScorer:
    """Test wallet scoring."""
    
    def test_score_wallet(self):
        """Test wallet scoring."""
        scorer = WalletScorer()
        
        signals = {
            "timing": {
                "pre_pump_entry_rate": 0.8,
                "pre_dump_exit_rate": 0.7,
                "avg_entry_timing_minutes": -30,
            },
            "profitability": {
                "win_rate": 0.75,
                "avg_roi": 1.5,
                "sharpe_ratio": 2.0,
            },
            "graph": {
                "centrality_score": 0.5,
                "clustering_coefficient": 0.3,
                "connected_smart_wallets": 5,
            },
            "behavioral": {
                "dex_diversity": 3,
                "gas_pattern_consistency": 0.6,
                "pattern_tags": ["early_adopter"],
            },
        }
        
        score = scorer.score_wallet(signals)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should be high for good signals
    
    def test_empty_signals(self):
        """Test with empty signals."""
        scorer = WalletScorer()
        score = scorer.score_wallet({})
        
        assert 0 <= score <= 1


class TestRiskFilter:
    """Test risk filtering."""
    
    def test_detect_wash_trading(self):
        """Test wash trading detection."""
        filter = RiskFilter()
        
        # High frequency with zero ROI
        signals = {
            "profitability": {"avg_roi": 0.001, "total_trades": 100},
            "behavioral": {"trade_frequency_per_day": 25},
            "timing": {"total_buys": 50, "total_sells": 50},
            "graph": {"centrality_score": 0.5, "clustering_coefficient": 0.5},
        }
        
        risk = filter.compute_risk_score(signals)
        
        assert 0 <= risk <= 1
        assert risk > 0.5  # Should flag as risky
    
    def test_low_risk_wallet(self):
        """Test low risk wallet."""
        filter = RiskFilter()
        
        signals = {
            "profitability": {"avg_roi": 1.5, "total_trades": 10},
            "behavioral": {"trade_frequency_per_day": 2, "gas_pattern_consistency": 0.5},
            "timing": {"total_buys": 10, "total_sells": 5},
            "graph": {"centrality_score": 0.5, "clustering_coefficient": 0.3},
        }
        
        risk = filter.compute_risk_score(signals)
        
        assert 0 <= risk <= 1
        assert risk < 0.5  # Should be low risk


class TestTradeMatcher:
    """Test trade matching."""
    
    def test_match_buys_and_sells(self):
        """Test FIFO trade matching."""
        matcher = TradeMatcher()
        
        now = datetime.utcnow()
        trades = [
            {
                "side": "buy",
                "buyer": "wallet1",
                "timestamp": (now - timedelta(hours=2)).isoformat(),
                "amount_tokens": 100,
                "amount_usd": 1000,
            },
            {
                "side": "sell",
                "seller": "wallet1",
                "timestamp": now.isoformat(),
                "amount_tokens": 50,
                "amount_usd": 750,
            },
        ]
        
        wallets = matcher.match_trades(trades)
        
        assert "wallet1" in wallets
        assert len(wallets["wallet1"]["trades"]) >= 0
    
    def test_empty_trades(self):
        """Test with empty trades."""
        matcher = TradeMatcher()
        wallets = matcher.match_trades([])
        
        assert wallets == {}


class TestWalletExplainer:
    """Test wallet explainer."""
    
    def test_explain_wallet(self):
        """Test explanation generation."""
        explainer = WalletExplainer()
        
        signals = {
            "timing": {
                "pre_pump_buys": 5,
                "pre_pump_entry_rate": 0.8,
                "pumps_detected": 3,
            },
            "profitability": {
                "win_rate": 0.75,
                "avg_roi": 1.5,
                "best_roi": 3.0,
            },
            "graph": {
                "connected_smart_wallets": 5,
                "centrality_score": 0.6,
            },
            "behavioral": {
                "pattern_tags": ["early_adopter", "swing_trader"],
            },
        }
        
        explanation = explainer.explain_wallet(
            wallet_address="0x1234",
            signals=signals,
            score=0.75,
        )
        
        assert "summary" in explanation
        assert "top_signals" in explanation
        assert "interpretation" in explanation
        assert "timeline" in explanation
        assert explanation["confidence_level"] in ["very_high", "high", "medium", "low"]
    
    def test_empty_signals(self):
        """Test with empty signals."""
        explainer = WalletExplainer()
        
        explanation = explainer.explain_wallet(
            wallet_address="0x1234",
            signals={},
            score=0.3,
        )
        
        assert "summary" in explanation
        assert explanation["confidence_level"] == "low"
