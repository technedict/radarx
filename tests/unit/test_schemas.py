"""Unit tests for schema validation."""

import pytest
from datetime import datetime
from radarx.schemas.token import (
    TokenScore, ProbabilityHeatmap, HorizonProbabilities,
    ProbabilityWithConfidence, RiskScore, RiskComponents
)
from radarx.schemas.wallet import (
    WalletReport, WinRate, WinRateByTimeframe,
    PnLSummary, RealizedPnL, TotalVolume
)


def test_probability_with_confidence():
    """Test ProbabilityWithConfidence model."""
    prob = ProbabilityWithConfidence(
        probability=0.5,
        confidence_interval={
            "lower": 0.4,
            "upper": 0.6,
            "confidence_level": 0.95
        }
    )
    assert prob.probability == 0.5
    assert prob.confidence_interval["lower"] == 0.4


def test_probability_validation():
    """Test probability value validation."""
    # Valid probability
    prob = ProbabilityWithConfidence(
        probability=0.5,
        confidence_interval={"lower": 0.4, "upper": 0.6}
    )
    assert prob.probability == 0.5
    
    # Invalid probability (> 1)
    with pytest.raises(ValueError):
        ProbabilityWithConfidence(
            probability=1.5,
            confidence_interval={"lower": 0.4, "upper": 0.6}
        )


def test_risk_score():
    """Test RiskScore model."""
    risk = RiskScore(
        composite_score=45.5,
        components=RiskComponents(
            rug_risk=30,
            dev_risk=50,
            distribution_risk=40,
            social_manipulation_risk=55,
            liquidity_risk=35
        ),
        risk_flags=["high_dev_holding"]
    )
    assert risk.composite_score == 45.5
    assert risk.components.dev_risk == 50
    assert len(risk.risk_flags) == 1


def test_win_rate():
    """Test WinRate model."""
    win_rate = WinRate(
        overall=0.68,
        by_timeframe=WinRateByTimeframe(
            one_day=0.75,
            seven_day=0.70,
            thirty_day=0.68,
            all_time=0.68
        ),
        profitable_trades=68,
        total_trades=100
    )
    assert win_rate.overall == 0.68
    assert win_rate.profitable_trades == 68


def test_pnl_summary():
    """Test PnLSummary model."""
    pnl = PnLSummary(
        realized_pnl=RealizedPnL(
            total_usd=10000,
            average_per_trade_usd=100,
            best_trade_usd=5000,
            worst_trade_usd=-500
        ),
        total_volume=TotalVolume(
            buy_volume_usd=50000,
            sell_volume_usd=60000,
            total_volume_usd=110000
        )
    )
    assert pnl.realized_pnl.total_usd == 10000
    assert pnl.total_volume.total_volume_usd == 110000


def test_token_score_serialization():
    """Test TokenScore can be serialized to JSON."""
    from examples.sample_responses import SAMPLE_TOKEN_SCORE
    
    token_score = TokenScore(**SAMPLE_TOKEN_SCORE)
    json_data = token_score.model_dump(mode='json')
    
    assert json_data["token_address"] == SAMPLE_TOKEN_SCORE["token_address"]
    assert "probability_heatmap" in json_data
    assert "risk_score" in json_data


def test_wallet_report_serialization():
    """Test WalletReport can be serialized to JSON."""
    from examples.sample_responses import SAMPLE_WALLET_REPORT
    
    wallet_report = WalletReport(**SAMPLE_WALLET_REPORT)
    json_data = wallet_report.model_dump(mode='json')
    
    assert json_data["wallet_address"] == SAMPLE_WALLET_REPORT["wallet_address"]
    assert "win_rate" in json_data
    assert "pnl_summary" in json_data
