"""Integration tests for complete workflows."""

import pytest
from radarx.api.services import TokenScoringService, WalletAnalyticsService


@pytest.mark.asyncio
async def test_token_scoring_workflow():
    """Test complete token scoring workflow."""
    service = TokenScoringService()
    
    result = await service.score_token(
        address="0x1234567890abcdef1234567890abcdef12345678",
        chain="ethereum",
        horizons=["24h", "7d"],
        include_features=True,
        include_timelines=True
    )
    
    # Verify all components are present
    assert result.token_address
    assert result.chain == "ethereum"
    assert result.probability_heatmap
    assert result.risk_score
    assert result.explanations
    assert result.features
    assert result.timelines


@pytest.mark.asyncio
async def test_wallet_analytics_workflow():
    """Test complete wallet analytics workflow."""
    service = WalletAnalyticsService()
    
    result = await service.get_wallet_report(
        address="0xabcdef1234567890abcdef1234567890abcdef12",
        chain="ethereum",
        period="30d",
        include_trades=True,
        max_trades=50
    )
    
    # Verify all components are present
    assert result.wallet_address
    assert result.win_rate
    assert result.pnl_summary
    assert result.timeframe
    
    # Verify win rate calculations
    assert 0 <= result.win_rate.overall <= 1
    assert result.win_rate.total_trades >= result.win_rate.profitable_trades


@pytest.mark.asyncio
async def test_wallet_search_workflow():
    """Test wallet search workflow."""
    service = WalletAnalyticsService()
    
    results = await service.search_wallets(
        min_win_rate=0.6,
        min_trades=10,
        sort_by="win_rate",
        order="desc",
        limit=10
    )
    
    assert "total" in results
    assert "wallets" in results
    assert len(results["wallets"]) <= 10
