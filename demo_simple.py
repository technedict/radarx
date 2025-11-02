#!/usr/bin/env python3
"""
Demo script to showcase RadarX capabilities using raw JSON data.
This script demonstrates the system without requiring installed dependencies.
"""

import json
import sys
import os

# Add parent directory to path to import sample data
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'examples'))

try:
    from sample_responses import SAMPLE_TOKEN_SCORE, SAMPLE_WALLET_REPORT
except ImportError:
    # Fallback: load from JSON files
    with open('examples/sample_token_score.json', 'r') as f:
        SAMPLE_TOKEN_SCORE = json.load(f)
    with open('examples/sample_wallet_report.json', 'r') as f:
        SAMPLE_WALLET_REPORT = json.load(f)


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_token_scoring():
    """Demonstrate token scoring functionality."""
    print_header("TOKEN SCORING DEMO")
    
    token = SAMPLE_TOKEN_SCORE
    metadata = token.get('token_metadata', {})
    
    print(f"📊 Token: {metadata.get('symbol')} ({metadata.get('name')})")
    print(f"🔗 Chain: {token['chain']}")
    print(f"📍 Address: {token['token_address']}")
    print(f"⏰ Analyzed at: {token['timestamp']}")
    
    print("\n🎯 PROBABILITY HEATMAP")
    print("-" * 80)
    
    horizons = token['probability_heatmap']['horizons']
    
    for horizon_name, horizon_data in horizons.items():
        print(f"\n⏱️  {horizon_name.upper()} Horizon:")
        
        for mult_name, mult_data in horizon_data.items():
            prob = mult_data['probability']
            ci_lower = mult_data['confidence_interval']['lower']
            ci_upper = mult_data['confidence_interval']['upper']
            
            # Create a simple bar chart
            bar_length = int(prob * 50)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            
            print(f"  {mult_name:4} [{bar}] {prob:6.1%} (CI: {ci_lower:.1%}-{ci_upper:.1%})")
    
    print("\n⚠️  RISK ASSESSMENT")
    print("-" * 80)
    
    risk = token['risk_score']
    print(f"Composite Risk Score: {risk['composite_score']:.1f}/100")
    
    print("\nRisk Components:")
    components = risk['components']
    
    for name, score in components.items():
        display_name = name.replace('_', ' ').title()
        bar_length = int(score / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {display_name:20} [{bar}] {score:.1f}/100")
    
    if risk.get('risk_flags'):
        print(f"\n🚩 Risk Flags ({len(risk['risk_flags'])}):")
        for flag in risk['risk_flags']:
            print(f"  • {flag.replace('_', ' ').title()}")
    
    print("\n💡 TOP EXPLANATIONS")
    print("-" * 80)
    
    explanations = token.get('explanations', {})
    if "probability_2x_24h" in explanations:
        print("\nFactors affecting 2x probability (24h):")
        for feature in explanations["probability_2x_24h"]["top_features"][:3]:
            symbol = "📈" if feature['direction'] == "positive" else "📉"
            print(f"  {symbol} {feature['feature_name']}: {feature['description']}")


def demo_wallet_analytics():
    """Demonstrate wallet analytics functionality."""
    print_header("WALLET ANALYTICS DEMO")
    
    wallet = SAMPLE_WALLET_REPORT
    
    print(f"👛 Wallet: {wallet['wallet_address']}")
    print(f"🔗 Chain: {wallet['chain']}")
    print(f"📅 Period: {wallet['timeframe']['period']}")
    
    print("\n📊 WIN RATE ANALYSIS")
    print("-" * 80)
    
    wr = wallet['win_rate']
    print(f"Overall Win Rate: {wr['overall']:.1%}")
    print(f"Profitable Trades: {wr['profitable_trades']}/{wr['total_trades']}")
    
    print("\nWin Rate by Timeframe:")
    for period, rate in wr['by_timeframe'].items():
        period_display = period.replace('_', ' ').title()
        print(f"  {period_display:10} {rate:.1%}")
    
    print("\n💰 PROFIT & LOSS")
    print("-" * 80)
    
    pnl = wallet['pnl_summary']
    rpnl = pnl['realized_pnl']
    print(f"Total Realized PnL: ${rpnl['total_usd']:,.2f}")
    print(f"Average Per Trade:  ${rpnl['average_per_trade_usd']:,.2f}")
    print(f"Best Trade:        ${rpnl['best_trade_usd']:,.2f}")
    print(f"Worst Trade:       ${rpnl['worst_trade_usd']:,.2f}")
    
    if 'unrealized_pnl' in pnl:
        print(f"\nUnrealized PnL:    ${pnl['unrealized_pnl']['total_usd']:,.2f}")
    
    print(f"\nTrading Volume:")
    vol = pnl['total_volume']
    print(f"  Buy Volume:      ${vol['buy_volume_usd']:,.2f}")
    print(f"  Sell Volume:     ${vol['sell_volume_usd']:,.2f}")
    print(f"  Total Volume:    ${vol['total_volume_usd']:,.2f}")
    
    if 'performance_metrics' in wallet:
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 80)
        pm = wallet['performance_metrics']
        print(f"Avg Trade Duration: {pm['average_trade_duration_hours']:.1f} hours")
        print(f"Trade Frequency:    {pm['trade_frequency_per_day']:.2f} trades/day")
        if 'sharpe_ratio' in pm and pm['sharpe_ratio']:
            print(f"Sharpe Ratio:       {pm['sharpe_ratio']:.2f}")
        if 'max_drawdown' in pm and pm['max_drawdown']:
            print(f"Max Drawdown:       {pm['max_drawdown']:.1%}")
    
    if 'behavioral_patterns' in wallet:
        print("\n🎭 BEHAVIORAL PATTERNS")
        print("-" * 80)
        bp = wallet['behavioral_patterns']
        
        if bp.get('pattern_tags'):
            print("Detected Patterns:")
            for tag in bp['pattern_tags']:
                print(f"  • {tag.replace('_', ' ').title()}")
        
        if bp.get('is_smart_money'):
            print("\n⭐ Classified as Smart Money")
        
        if bp.get('copies_wallet'):
            print(f"\n📋 Follows wallet: {bp['copies_wallet']}")
        
        print(f"\nWash Trading Score: {bp['wash_trading_score']:.1%}")
    
    if 'ranking' in wallet:
        print("\n🏆 RANKINGS")
        print("-" * 80)
        rank = wallet['ranking']
        if rank.get('global_rank'):
            print(f"Global Rank: #{rank['global_rank']:,}")
        if rank.get('chain_rank'):
            print(f"Chain Rank:  #{rank['chain_rank']:,}")
        print(f"Percentile:  Top {(1 - rank['percentile']) * 100:.1f}%")
    
    if 'breakdown_by_token' in wallet and wallet['breakdown_by_token']:
        print("\n🪙 TOP TOKENS")
        print("-" * 80)
        print(f"{'Token':<10} {'Chain':<10} {'Trades':<8} {'Win Rate':<10} {'PnL (USD)':<15}")
        print("-" * 80)
        
        for token in wallet['breakdown_by_token'][:5]:
            print(f"{token['token_symbol']:<10} {token['chain']:<10} {token['trade_count']:<8} "
                  f"{token['win_rate']:<10.1%} ${token['total_pnl_usd']:>13,.2f}")


def main():
    """Run the demo."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                            ║")
    print("║                           RadarX Demo                                      ║")
    print("║                  Memecoin Analysis & Wallet Intelligence                   ║")
    print("║                                                                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    try:
        demo_token_scoring()
        demo_wallet_analytics()
        
        print_header("DEMO COMPLETE")
        print("✨ This demo showcased RadarX capabilities with sample data.")
        print("🚀 To use with real data, configure API keys and run the server:")
        print("   python -m radarx.api.server")
        print("📚 Visit http://localhost:8000/docs for interactive API documentation")
        print("\n")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        print("Make sure you're running from the radarx directory.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
