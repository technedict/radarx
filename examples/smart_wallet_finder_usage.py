"""
Smart Wallet Finder - Example Usage

Demonstrates how to use the Smart Wallet Finder API endpoints.
"""

import requests
import json
from typing import Dict, Any


class SmartWalletFinderClient:
    """Client for Smart Wallet Finder API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize client.
        
        Args:
            base_url: Base URL for the API
        """
        self.base_url = base_url
    
    def find_smart_wallets(
        self,
        token_address: str,
        chain: str = "ethereum",
        window_days: int = 30,
        min_confidence: float = 0.5,
        top_k: int = 100,
    ) -> Dict[str, Any]:
        """
        Find smart wallets for a token.
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            window_days: Analysis window in days
            min_confidence: Minimum confidence threshold
            top_k: Number of top wallets to return
            
        Returns:
            API response with ranked wallets
        """
        url = f"{self.base_url}/smart-wallets/find"
        
        payload = {
            "token_address": token_address,
            "chain": chain,
            "window_days": window_days,
            "min_confidence": min_confidence,
            "top_k": top_k,
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def get_wallet_profile(
        self,
        wallet_address: str,
        token_address: str,
        chain: str = "ethereum",
        window_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Get detailed wallet profile.
        
        Args:
            wallet_address: Wallet address
            token_address: Token address
            chain: Blockchain network
            window_days: Analysis window
            
        Returns:
            API response with wallet profile
        """
        url = f"{self.base_url}/smart-wallets/profile"
        
        payload = {
            "wallet_address": wallet_address,
            "token_address": token_address,
            "chain": chain,
            "window_days": window_days,
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()
    
    def bulk_scan(
        self,
        token_addresses: list,
        chain: str = "ethereum",
        window_days: int = 30,
        top_k_per_token: int = 10,
        min_confidence: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Scan multiple tokens for smart wallets.
        
        Args:
            token_addresses: List of token addresses
            chain: Blockchain network
            window_days: Analysis window
            top_k_per_token: Top wallets per token
            min_confidence: Minimum confidence threshold
            
        Returns:
            API response with results and leaderboard
        """
        url = f"{self.base_url}/smart-wallets/bulk-scan"
        
        payload = {
            "token_addresses": token_addresses,
            "chain": chain,
            "window_days": window_days,
            "top_k_per_token": top_k_per_token,
            "min_confidence": min_confidence,
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        return response.json()


def example_find_smart_wallets():
    """Example: Find smart wallets for a token."""
    print("=" * 80)
    print("Example 1: Find Smart Wallets for a Token")
    print("=" * 80)
    
    client = SmartWalletFinderClient()
    
    # Find smart wallets
    result = client.find_smart_wallets(
        token_address="0x1234567890abcdef1234567890abcdef12345678",
        chain="ethereum",
        window_days=30,
        min_confidence=0.5,
        top_k=50,
    )
    
    print(f"\nToken: {result['token_address']}")
    print(f"Chain: {result['chain']}")
    print(f"Analysis Window: {result['analysis_window_days']} days")
    print(f"\nFound {len(result['ranked_wallets'])} smart wallets")
    
    print("\nTop 10 Smart Wallets:")
    print("-" * 80)
    
    for wallet in result['ranked_wallets'][:10]:
        print(f"\n#{wallet['rank']}: {wallet['wallet_address']}")
        print(f"  Smart Money Score: {wallet['smart_money_score']:.2%}")
        print(f"  Win Rate: {wallet['key_metrics']['win_rate']:.2%}")
        print(f"  Realized ROI: {wallet['key_metrics']['realized_roi']:.2f}x")
        print(f"  Trades: {wallet['key_metrics']['trades_count']}")
        print(f"  Early Entry Rate: {wallet['key_metrics']['early_entry_rate']:.2%}")
        print(f"  Risk Score: {wallet['risk_score']:.2%}")
        
        # Print explanation summary
        print(f"\n  {wallet['explanation']['summary']}")
        
        # Print top contributing signals
        if wallet['explanation']['top_signals']:
            print("\n  Top Signals:")
            for signal in wallet['explanation']['top_signals'][:3]:
                print(f"    - {signal['name']}: {signal['description']}")
    
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    
    stats = result['summary_stats']
    print(f"Average Smart Money Score: {stats['avg_smart_money_score']:.2%}")
    print(f"Median Smart Money Score: {stats['median_smart_money_score']:.2%}")
    print(f"Average Win Rate: {stats['avg_win_rate']:.2%}")
    print(f"Total Smart Wallets: {stats['total_smart_wallets']}")
    
    print("\n" + "=" * 80)
    print("Analysis Metadata")
    print("=" * 80)
    
    metadata = result['metadata']
    print(f"Total Wallets Analyzed: {metadata['total_wallets_analyzed']}")
    print(f"Wallets Passing Filters: {metadata['wallets_passing_filters']}")
    print(f"Wallets Returned: {metadata['wallets_returned']}")
    print(f"Confidence Threshold: {metadata['confidence_threshold']:.2%}")


def example_wallet_profile():
    """Example: Get detailed wallet profile."""
    print("\n\n")
    print("=" * 80)
    print("Example 2: Get Wallet Profile")
    print("=" * 80)
    
    client = SmartWalletFinderClient()
    
    # Get wallet profile
    profile = client.get_wallet_profile(
        wallet_address="0xabcdef1234567890abcdef1234567890abcdef12",
        token_address="0x1234567890abcdef1234567890abcdef12345678",
        chain="ethereum",
        window_days=30,
    )
    
    print(f"\nWallet: {profile['wallet_address']}")
    print(f"Token: {profile['token_address']}")
    print(f"Chain: {profile['chain']}")
    print(f"\nSmart Money Score: {profile['score']:.2%}")
    print(f"Realized ROI: {profile['realized_roi']:.2f}x")
    print(f"Win Rate: {profile['win_rate']:.2%}")
    
    print(f"\nTotal Trades: {len(profile['trades'])}")
    
    if profile['trades']:
        print("\nRecent Trades:")
        print("-" * 80)
        
        for i, trade in enumerate(profile['trades'][:5], 1):
            print(f"\n{i}. {trade['side'].upper()} - {trade['timestamp']}")
            print(f"   Amount: ${trade['amount_usd']:,.2f}")
            print(f"   Price: ${trade['price']:.6f}")
            
            if trade.get('pnl') is not None:
                print(f"   PnL: ${trade['pnl']:,.2f}")
                print(f"   ROI: {trade['roi']:.2%}")
    
    print(f"\nGraph Neighbors: {len(profile['graph_neighbors'])} wallets")
    
    if profile['graph_neighbors']:
        print("  Connected to:")
        for neighbor in profile['graph_neighbors'][:5]:
            print(f"    - {neighbor}")
    
    print("\n" + "=" * 80)
    print("Explanation")
    print("=" * 80)
    
    explanation = profile['explanation']
    print(f"\n{explanation['summary']}")
    print(f"\nInterpretation: {explanation['interpretation']}")
    print(f"Confidence Level: {explanation['confidence_level']}")
    
    if explanation['timeline']:
        print("\nKey Events:")
        for event in explanation['timeline']:
            print(f"  - {event['event_type']}: {event['description']} ({event['impact']})")


def example_bulk_scan():
    """Example: Bulk scan multiple tokens."""
    print("\n\n")
    print("=" * 80)
    print("Example 3: Bulk Scan Multiple Tokens")
    print("=" * 80)
    
    client = SmartWalletFinderClient()
    
    # Scan multiple tokens
    result = client.bulk_scan(
        token_addresses=[
            "0x1111111111111111111111111111111111111111",
            "0x2222222222222222222222222222222222222222",
            "0x3333333333333333333333333333333333333333",
        ],
        chain="ethereum",
        window_days=30,
        top_k_per_token=10,
        min_confidence=0.6,
    )
    
    print(f"\nChain: {result['chain']}")
    print(f"Tokens Analyzed: {result['tokens_analyzed']}")
    print(f"Timestamp: {result['timestamp']}")
    
    print("\n" + "=" * 80)
    print("Results Per Token")
    print("=" * 80)
    
    for token_result in result['results']:
        print(f"\nToken: {token_result['token_address']}")
        print(f"Average Score: {token_result['avg_score']:.2%}")
        print(f"Top Wallets: {len(token_result['top_wallets'])}")
        
        if token_result['top_wallets']:
            top_wallet = token_result['top_wallets'][0]
            print(f"  Best: {top_wallet['wallet_address']} "
                  f"(score: {top_wallet['smart_money_score']:.2%})")
    
    print("\n" + "=" * 80)
    print("Global Leaderboard (Top 10)")
    print("=" * 80)
    
    for wallet in result['leaderboard'][:10]:
        print(f"\n#{wallet['rank']}: {wallet['wallet_address']}")
        print(f"  Score: {wallet['smart_money_score']:.2%}")
        print(f"  Win Rate: {wallet['key_metrics']['win_rate']:.2%}")
        print(f"  ROI: {wallet['key_metrics']['realized_roi']:.2f}x")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("Smart Wallet Finder - API Examples")
    print("=" * 80)
    
    # Note: These examples use mock data and will not work without
    # a running API server with actual blockchain data
    
    print("\nNote: These are demonstration examples.")
    print("To run them, ensure the API server is running at http://localhost:8000")
    print("and has access to blockchain data sources.")
    
    try:
        # Example 1: Find smart wallets
        example_find_smart_wallets()
        
        # Example 2: Get wallet profile
        example_wallet_profile()
        
        # Example 3: Bulk scan
        example_bulk_scan()
        
    except requests.exceptions.ConnectionError:
        print("\n\nError: Could not connect to API server.")
        print("Please start the server with: python -m radarx.api.server")
    
    except requests.exceptions.HTTPError as e:
        print(f"\n\nHTTP Error: {e}")
    
    except Exception as e:
        print(f"\n\nError: {e}")


if __name__ == "__main__":
    main()
