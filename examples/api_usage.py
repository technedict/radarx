"""Example notebook demonstrating RadarX API usage."""

# This would be a Jupyter notebook in production
# For now, providing Python script equivalent

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"


def example_token_scoring():
    """Example: Score a token."""
    print("=== Token Scoring Example ===\n")
    
    response = requests.get(
        f"{BASE_URL}/score/token",
        params={
            "address": "0x1234567890abcdef1234567890abcdef12345678",
            "chain": "ethereum",
            "horizons": "24h,7d,30d",
            "include_features": True,
            "include_timelines": True
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Token: {data['token_address']}")
        print(f"Chain: {data['chain']}")
        print(f"Risk Score: {data['risk_score']['composite_score']:.2f}/100")
        print("\nProbability Heatmap (24h):")
        for mult, prob_data in data['probability_heatmap']['horizons']['24h'].items():
            prob = prob_data['probability']
            print(f"  {mult}: {prob:.1%}")
        print("\nTop Risk Factors:")
        for flag in data['risk_score']['risk_flags'][:3]:
            print(f"  - {flag}")
    else:
        print(f"Error: {response.status_code}")


def example_wallet_analytics():
    """Example: Get wallet analytics."""
    print("\n=== Wallet Analytics Example ===\n")
    
    response = requests.get(
        f"{BASE_URL}/wallet/report",
        params={
            "address": "0xabcdef1234567890abcdef1234567890abcdef12",
            "period": "30d",
            "include_trades": True,
            "max_trades": 10
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Wallet: {data['wallet_address']}")
        print(f"Win Rate: {data['win_rate']['overall']:.1%}")
        print(f"Profitable Trades: {data['win_rate']['profitable_trades']}/{data['win_rate']['total_trades']}")
        print(f"Total PnL: ${data['pnl_summary']['realized_pnl']['total_usd']:,.2f}")
        print(f"Best Trade: ${data['pnl_summary']['realized_pnl']['best_trade_usd']:,.2f}")
        print("\nBehavioral Patterns:")
        for pattern in data.get('behavioral_patterns', {}).get('pattern_tags', [])[:3]:
            print(f"  - {pattern}")
    else:
        print(f"Error: {response.status_code}")


def example_wallet_search():
    """Example: Search for top wallets."""
    print("\n=== Wallet Search Example ===\n")
    
    response = requests.get(
        f"{BASE_URL}/search/wallets",
        params={
            "min_win_rate": 0.6,
            "min_trades": 10,
            "sort_by": "win_rate",
            "order": "desc",
            "limit": 5
        }
    )
    
    if response.status_code == 200:
        data = response.json()
        print(f"Found {data['total']} wallets")
        print(f"\nTop {len(data['wallets'])} wallets:")
        for i, wallet in enumerate(data['wallets'], 1):
            print(f"{i}. {wallet['wallet_address'][:10]}...")
            print(f"   Win Rate: {wallet['win_rate']:.1%}")
            print(f"   Total PnL: ${wallet['realized_pnl_usd']:,.2f}")
            print(f"   Rank: #{wallet['ranking']['global_rank']}")
    else:
        print(f"Error: {response.status_code}")


if __name__ == "__main__":
    print("RadarX API Examples")
    print("=" * 50)
    print("Make sure the API server is running at http://localhost:8000\n")
    
    try:
        # Test connection
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✓ API server is running\n")
        else:
            print("✗ API server is not responding correctly")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to API server")
        print("Please start the server with: python -m radarx.api.server")
        exit(1)
    
    # Run examples
    example_token_scoring()
    example_wallet_analytics()
    example_wallet_search()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("Visit http://localhost:8000/docs for interactive API documentation")
