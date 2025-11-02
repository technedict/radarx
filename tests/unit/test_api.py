"""Unit tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from radarx.api.server import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns API info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data
    assert "endpoints" in data


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_score_token():
    """Test token scoring endpoint."""
    response = client.get(
        "/score/token",
        params={
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "chain": "ethereum"
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "token_address" in data
    assert "chain" in data
    assert "probability_heatmap" in data
    assert "risk_score" in data
    assert "explanations" in data
    
    # Verify probability heatmap
    heatmap = data["probability_heatmap"]
    assert "horizons" in heatmap
    
    # Verify risk score
    risk = data["risk_score"]
    assert "composite_score" in risk
    assert "components" in risk
    assert 0 <= risk["composite_score"] <= 100


def test_score_token_with_options():
    """Test token scoring with optional parameters."""
    response = client.get(
        "/score/token",
        params={
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "chain": "ethereum",
            "horizons": "24h,7d",
            "include_features": True,
            "include_timelines": True
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Should have features and timelines
    assert "features" in data
    assert "timelines" in data


def test_wallet_report():
    """Test wallet report endpoint."""
    response = client.get(
        "/wallet/report",
        params={
            "address": "0xabcdef1234567890abcdef1234567890abcdef12",
            "period": "30d"
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "wallet_address" in data
    assert "win_rate" in data
    assert "pnl_summary" in data
    
    # Verify win rate
    win_rate = data["win_rate"]
    assert "overall" in win_rate
    assert 0 <= win_rate["overall"] <= 1
    assert "profitable_trades" in win_rate
    assert "total_trades" in win_rate
    
    # Verify PnL
    pnl = data["pnl_summary"]
    assert "realized_pnl" in pnl
    assert "total_volume" in pnl


def test_search_wallets():
    """Test wallet search endpoint."""
    response = client.get(
        "/search/wallets",
        params={
            "min_win_rate": 0.6,
            "min_trades": 10,
            "limit": 50
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    assert "total" in data
    assert "wallets" in data
    assert "limit" in data
    assert "offset" in data
    assert isinstance(data["wallets"], list)


def test_alerts_subscribe():
    """Test alert subscription endpoint."""
    response = client.post(
        "/alerts/subscribe",
        params={
            "webhook_url": "https://example.com/webhook",
            "min_probability_10x": 0.5
        }
    )
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "subscribed"
    assert "subscription_id" in data
    assert "webhook_url" in data


def test_missing_required_params():
    """Test endpoints reject missing required parameters."""
    response = client.get("/score/token")
    assert response.status_code == 422  # Validation error
    
    response = client.get("/wallet/report")
    assert response.status_code == 422  # Validation error
