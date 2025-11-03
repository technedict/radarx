"""
Smart Wallet Finder - Comprehensive Demo

Demonstrates the complete Smart Wallet Finder system with mock data.
This can be run directly to see how the system works without requiring
live blockchain data.
"""

from datetime import datetime, timedelta
from radarx.smart_wallet_finder.finder import SmartWalletFinder
from radarx.smart_wallet_finder.data_fetcher import DataFetcher


class DemoDataFetcher(DataFetcher):
    """Demo data fetcher with realistic mock data."""
    
    def fetch_token_data(self, token_address, chain, window_days, include_internal_transfers):
        """Return realistic mock data for demo."""
        now = datetime.utcnow()
        
        # Create diverse set of wallets with different characteristics
        trades = []
        
        # Smart Money Wallet 1: Early adopter, high win rate
        for i in range(10):
            trades.extend([
                {
                    "side": "buy",
                    "buyer": "0xSmart1",
                    "seller": "0xOther",
                    "timestamp": (now - timedelta(hours=72-i*6)).isoformat(),
                    "amount_tokens": 1000,
                    "amount_usd": 1000,
                    "dex": "uniswap",
                    "gas_price": 50 + i,
                    "token_age_hours": 12,
                },
                {
                    "side": "sell",
                    "seller": "0xSmart1",
                    "buyer": "0xOther",
                    "timestamp": (now - timedelta(hours=48-i*6)).isoformat(),
                    "amount_tokens": 1000,
                    "amount_usd": 2500,  # 150% profit
                    "dex": "uniswap",
                    "gas_price": 51 + i,
                },
            ])
        
        # Smart Money Wallet 2: Pre-pump detector
        for i in range(5):
            trades.extend([
                {
                    "side": "buy",
                    "buyer": "0xSmart2",
                    "seller": "0xOther",
                    "timestamp": (now - timedelta(hours=96-i*12)).isoformat(),
                    "amount_tokens": 500,
                    "amount_usd": 500,
                    "dex": "sushiswap",
                    "gas_price": 48,
                },
                {
                    "side": "sell",
                    "seller": "0xSmart2",
                    "buyer": "0xOther",
                    "timestamp": (now - timedelta(hours=24-i*4)).isoformat(),
                    "amount_tokens": 500,
                    "amount_usd": 1000,  # 100% profit
                    "dex": "sushiswap",
                    "gas_price": 49,
                },
            ])
        
        # Regular Trader: Mixed results
        for i in range(8):
            profit_multiplier = 1.2 if i % 2 == 0 else 0.8
            trades.extend([
                {
                    "side": "buy",
                    "buyer": "0xRegular1",
                    "seller": "0xOther",
                    "timestamp": (now - timedelta(hours=120-i*10)).isoformat(),
                    "amount_tokens": 200,
                    "amount_usd": 200,
                    "dex": "uniswap",
                    "gas_price": 45,
                    "token_age_hours": 72,
                },
                {
                    "side": "sell",
                    "seller": "0xRegular1",
                    "buyer": "0xOther",
                    "timestamp": (now - timedelta(hours=100-i*10)).isoformat(),
                    "amount_tokens": 200,
                    "amount_usd": 200 * profit_multiplier,
                    "dex": "uniswap",
                    "gas_price": 46,
                },
            ])
        
        # Wash Trader: High frequency, no profit
        for i in range(50):
            trades.extend([
                {
                    "side": "buy",
                    "buyer": "0xBot1",
                    "seller": "0xOther",
                    "timestamp": (now - timedelta(hours=48-i*0.5)).isoformat(),
                    "amount_tokens": 10,
                    "amount_usd": 10,
                    "dex": "uniswap",
                    "gas_price": 50,  # Constant gas
                },
                {
                    "side": "sell",
                    "seller": "0xBot1",
                    "buyer": "0xOther",
                    "timestamp": (now - timedelta(hours=47.9-i*0.5)).isoformat(),
                    "amount_tokens": 10,
                    "amount_usd": 10.01,  # Minimal profit
                    "dex": "uniswap",
                    "gas_price": 50,  # Constant gas
                },
            ])
        
        # Price timeline with clear pumps and dumps
        price_timeline = []
        base_price = 1.0
        
        for hour in range(120, -1, -1):
            time = now - timedelta(hours=hour)
            
            # Create pump patterns
            if 90 <= hour <= 96:
                price = base_price * 1.5  # Pump
            elif 40 <= hour <= 48:
                price = base_price * 2.0  # Big pump
            elif 10 <= hour <= 20:
                price = base_price * 0.6  # Dump
            else:
                price = base_price
            
            price_timeline.append({
                "timestamp": time.isoformat(),
                "price": price,
            })
        
        # Build graph showing connections
        wallets = ["0xSmart1", "0xSmart2", "0xRegular1", "0xBot1"]
        graph_data = {
            "nodes": {w: {"address": w} for w in wallets},
            "edges": [
                {"from": "0xSmart1", "to": "0xSmart2", "type": "fund_flow"},
                {"from": "0xSmart1", "to": "0xRegular1", "type": "trade"},
            ],
            "clusters": {
                "0xSmart1": 1,
                "0xSmart2": 1,
                "0xRegular1": 2,
                "0xBot1": 3,
            },
            "smart_wallets": {"0xSmart1"},  # Known smart wallet
        }
        
        return {
            "trades": trades,
            "price_timeline": price_timeline,
            "graph_data": graph_data,
            "token_metadata": {
                "address": token_address,
                "chain": chain,
                "name": "DemoToken",
                "symbol": "DEMO",
            },
            "chain": chain,
            "window_start": (now - timedelta(days=window_days)).isoformat(),
            "window_end": now.isoformat(),
        }


def print_banner(text):
    """Print a formatted banner."""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80 + "\n")


def print_section(text):
    """Print a section header."""
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def demo_find_smart_wallets():
    """Demonstrate finding smart wallets."""
    print_banner("SMART WALLET FINDER DEMO")
    
    print("Initializing Smart Wallet Finder with demo data...")
    finder = SmartWalletFinder(data_fetcher=DemoDataFetcher())
    
    print("\nFinding smart wallets for demo token...")
    print("  Token: 0xDEMO123")
    print("  Chain: ethereum")
    print("  Window: 30 days")
    print("  Confidence threshold: 0.3")
    
    result = finder.find_smart_wallets(
        token_address="0xDEMO123",
        chain="ethereum",
        window_days=30,
        top_k=10,
        min_confidence=0.3,
    )
    
    print_section("ANALYSIS RESULTS")
    
    print(f"Token: {result['token_address']}")
    print(f"Chain: {result['chain']}")
    print(f"Analysis Window: {result['analysis_window_days']} days")
    print(f"Timestamp: {result['timestamp']}")
    
    print_section("METADATA")
    
    metadata = result['metadata']
    print(f"Total Wallets Analyzed: {metadata['total_wallets_analyzed']}")
    print(f"Wallets Passing Filters: {metadata['wallets_passing_filters']}")
    print(f"Wallets Returned: {metadata['wallets_returned']}")
    print(f"Confidence Threshold: {metadata['confidence_threshold']:.0%}")
    
    print_section("SUMMARY STATISTICS")
    
    stats = result['summary_stats']
    if stats:
        print(f"Average Smart Money Score: {stats.get('avg_smart_money_score', 0):.2%}")
        print(f"Median Smart Money Score: {stats.get('median_smart_money_score', 0):.2%}")
        print(f"Average Win Rate: {stats.get('avg_win_rate', 0):.2%}")
        print(f"Total Smart Wallets: {stats.get('total_smart_wallets', 0)}")
    
    print_section("RANKED SMART WALLETS")
    
    for wallet in result['ranked_wallets']:
        print(f"\n#{wallet['rank']}: {wallet['wallet_address']}")
        print(f"  Smart Money Score: {wallet['smart_money_score']:.2%}")
        print(f"  Risk Score: {wallet['risk_score']:.2%}")
        
        metrics = wallet['key_metrics']
        print(f"\n  Key Metrics:")
        print(f"    Win Rate: {metrics['win_rate']:.2%}")
        print(f"    Realized ROI: {metrics['realized_roi']:.2f}x")
        print(f"    Trades Count: {metrics['trades_count']}")
        print(f"    Early Entry Rate: {metrics['early_entry_rate']:.2%}")
        print(f"    Graph Centrality: {metrics['graph_centrality']:.2%}")
        
        explanation = wallet['explanation']
        print(f"\n  Summary:")
        print(f"    {explanation['summary']}")
        
        print(f"\n  Interpretation:")
        print(f"    {explanation['interpretation']}")
        
        print(f"\n  Confidence Level: {explanation['confidence_level']}")
        
        if explanation['top_signals']:
            print(f"\n  Top Contributing Signals:")
            for i, signal in enumerate(explanation['top_signals'][:3], 1):
                print(f"    {i}. {signal['name']} ({signal['category']})")
                print(f"       {signal['description']}")
                print(f"       Contribution: {signal['contribution']:.2%}, Direction: {signal['direction']}")
        
        if explanation['timeline']:
            print(f"\n  Timeline of Key Events:")
            for event in explanation['timeline']:
                print(f"    - {event['event_type']}: {event['description']} ({event['impact']})")
        
        print("\n  " + "-" * 76)


def demo_wallet_profile():
    """Demonstrate wallet profiling."""
    print_banner("WALLET PROFILE DEMO")
    
    finder = SmartWalletFinder(data_fetcher=DemoDataFetcher())
    
    print("Getting profile for wallet 0xSmart1...")
    
    profile = finder.get_wallet_profile(
        wallet_address="0xSmart1",
        token_address="0xDEMO123",
        chain="ethereum",
        window_days=30,
    )
    
    print_section("WALLET PROFILE")
    
    print(f"Wallet: {profile['wallet_address']}")
    print(f"Token: {profile['token_address']}")
    print(f"Chain: {profile['chain']}")
    print(f"\nSmart Money Score: {profile['score']:.2%}")
    print(f"Realized ROI: {profile['realized_roi']:.2f}x")
    print(f"Win Rate: {profile['win_rate']:.2%}")
    
    print_section("TRADES")
    
    print(f"Total Trades: {len(profile['trades'])}")
    
    if profile['trades']:
        print("\nRecent Trades:")
        for i, trade in enumerate(profile['trades'][:5], 1):
            print(f"\n  {i}. {trade.get('side', 'N/A').upper()} - {trade.get('timestamp', 'N/A')}")
            if 'amount_usd' in trade:
                print(f"     Amount: ${trade['amount_usd']:,.2f}")
            if 'price' in trade:
                print(f"     Price: ${trade['price']:.6f}")
            if trade.get('pnl') is not None:
                print(f"     PnL: ${trade['pnl']:,.2f}")
            if trade.get('roi') is not None:
                print(f"     ROI: {trade['roi']:.2%}")
    
    print_section("GRAPH CONNECTIONS")
    
    print(f"Connected to {len(profile['graph_neighbors'])} wallets:")
    for neighbor in profile['graph_neighbors']:
        print(f"  - {neighbor}")
    
    print_section("EXPLANATION")
    
    explanation = profile['explanation']
    print(f"\nSummary:\n  {explanation['summary']}")
    print(f"\nInterpretation:\n  {explanation['interpretation']}")
    print(f"\nConfidence Level: {explanation['confidence_level']}")


def main():
    """Run the demo."""
    print("\n")
    print("*" * 80)
    print("SMART WALLET FINDER - INTERACTIVE DEMO".center(80))
    print("*" * 80)
    print("\nThis demo showcases the Smart Wallet Finder system using mock data.")
    print("It demonstrates how the system identifies smart-money wallets based on")
    print("trading patterns, timing signals, profitability, and graph analysis.")
    
    # Demo 1: Find smart wallets
    demo_find_smart_wallets()
    
    # Demo 2: Wallet profile
    demo_wallet_profile()
    
    print_banner("DEMO COMPLETE")
    print("\nKey Observations:")
    print("  1. Smart wallets (0xSmart1, 0xSmart2) scored highly due to:")
    print("     - High win rates and profitability")
    print("     - Pre-pump entry timing")
    print("     - Graph connections to other smart wallets")
    print("\n  2. Regular traders scored moderately:")
    print("     - Mixed trading results")
    print("     - Decent but not exceptional timing")
    print("\n  3. Bot/wash traders were filtered out:")
    print("     - High risk scores from bot detection")
    print("     - Suspicious trading patterns")
    print("\nFor production use, connect to real blockchain data sources!")


if __name__ == "__main__":
    main()
