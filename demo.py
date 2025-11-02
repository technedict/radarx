#!/usr/bin/env python3
"""
Demo script to showcase RadarX capabilities.
This script demonstrates the API with mock data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from datetime import datetime
from radarx.schemas.token import TokenScore
from radarx.schemas.wallet import WalletReport
from examples.sample_responses import SAMPLE_TOKEN_SCORE, SAMPLE_WALLET_REPORT


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def demo_token_scoring():
    """Demonstrate token scoring functionality."""
    print_header("TOKEN SCORING DEMO")
    
    # Load sample data into Pydantic model
    token_score = TokenScore(**SAMPLE_TOKEN_SCORE)
    
    print(f"📊 Token: {token_score.token_metadata.symbol} ({token_score.token_metadata.name})")
    print(f"🔗 Chain: {token_score.chain}")
    print(f"📍 Address: {token_score.token_address}")
    print(f"⏰ Analyzed at: {token_score.timestamp}")
    
    print("\n🎯 PROBABILITY HEATMAP")
    print("-" * 80)
    
    for horizon_name, horizon_data in token_score.probability_heatmap.horizons.items():
        print(f"\n⏱️  {horizon_name.upper()} Horizon:")
        
        multipliers = [
            ("2x", horizon_data.two_x),
            ("5x", horizon_data.five_x),
            ("10x", horizon_data.ten_x),
            ("20x", horizon_data.twenty_x),
            ("50x", horizon_data.fifty_x),
        ]
        
        for mult_name, mult_data in multipliers:
            if mult_data:
                prob = mult_data.probability
                ci_lower = mult_data.confidence_interval.get("lower", 0)
                ci_upper = mult_data.confidence_interval.get("upper", 0)
                
                # Create a simple bar chart
                bar_length = int(prob * 50)
                bar = "█" * bar_length + "░" * (50 - bar_length)
                
                print(f"  {mult_name:4} [{bar}] {prob:6.1%} (CI: {ci_lower:.1%}-{ci_upper:.1%})")
    
    print("\n⚠️  RISK ASSESSMENT")
    print("-" * 80)
    
    risk = token_score.risk_score
    print(f"Composite Risk Score: {risk.composite_score:.1f}/100")
    
    print("\nRisk Components:")
    components = [
        ("Rug Risk", risk.components.rug_risk),
        ("Dev Risk", risk.components.dev_risk),
        ("Distribution Risk", risk.components.distribution_risk),
        ("Social Manipulation", risk.components.social_manipulation_risk),
        ("Liquidity Risk", risk.components.liquidity_risk),
    ]
    
    for name, score in components:
        bar_length = int(score / 2)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        print(f"  {name:20} [{bar}] {score:.1f}/100")
    
    if risk.risk_flags:
        print(f"\n🚩 Risk Flags ({len(risk.risk_flags)}):")
        for flag in risk.risk_flags:
            print(f"  • {flag.replace('_', ' ').title()}")
    
    print("\n💡 TOP EXPLANATIONS")
    print("-" * 80)
    
    if "probability_2x_24h" in token_score.explanations:
        print("\nFactors affecting 2x probability (24h):")
        for feature in token_score.explanations["probability_2x_24h"].top_features[:3]:
            symbol = "📈" if feature.direction == "positive" else "📉"
            print(f"  {symbol} {feature.feature_name}: {feature.description}")


def demo_wallet_analytics():
    """Demonstrate wallet analytics functionality."""
    print_header("WALLET ANALYTICS DEMO")
    
    # Load sample data into Pydantic model
    wallet = WalletReport(**SAMPLE_WALLET_REPORT)
    
    print(f"👛 Wallet: {wallet.wallet_address}")
    print(f"🔗 Chain: {wallet.chain}")
    print(f"📅 Period: {wallet.timeframe.period}")
    
    print("\n📊 WIN RATE ANALYSIS")
    print("-" * 80)
    
    wr = wallet.win_rate
    print(f"Overall Win Rate: {wr.overall:.1%}")
    print(f"Profitable Trades: {wr.profitable_trades}/{wr.total_trades}")
    
    print("\nWin Rate by Timeframe:")
    if wr.by_timeframe.one_day:
        print(f"  24 Hours:  {wr.by_timeframe.one_day:.1%}")
    if wr.by_timeframe.seven_day:
        print(f"  7 Days:    {wr.by_timeframe.seven_day:.1%}")
    if wr.by_timeframe.thirty_day:
        print(f"  30 Days:   {wr.by_timeframe.thirty_day:.1%}")
    if wr.by_timeframe.all_time:
        print(f"  All-Time:  {wr.by_timeframe.all_time:.1%}")
    
    print("\n💰 PROFIT & LOSS")
    print("-" * 80)
    
    pnl = wallet.pnl_summary
    print(f"Total Realized PnL: ${pnl.realized_pnl.total_usd:,.2f}")
    print(f"Average Per Trade:  ${pnl.realized_pnl.average_per_trade_usd:,.2f}")
    print(f"Best Trade:        ${pnl.realized_pnl.best_trade_usd:,.2f}")
    print(f"Worst Trade:       ${pnl.realized_pnl.worst_trade_usd:,.2f}")
    
    if pnl.unrealized_pnl:
        print(f"\nUnrealized PnL:    ${pnl.unrealized_pnl.total_usd:,.2f}")
    
    print(f"\nTrading Volume:")
    print(f"  Buy Volume:      ${pnl.total_volume.buy_volume_usd:,.2f}")
    print(f"  Sell Volume:     ${pnl.total_volume.sell_volume_usd:,.2f}")
    print(f"  Total Volume:    ${pnl.total_volume.total_volume_usd:,.2f}")
    
    if wallet.performance_metrics:
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 80)
        pm = wallet.performance_metrics
        print(f"Avg Trade Duration: {pm.average_trade_duration_hours:.1f} hours")
        print(f"Trade Frequency:    {pm.trade_frequency_per_day:.2f} trades/day")
        if pm.sharpe_ratio:
            print(f"Sharpe Ratio:       {pm.sharpe_ratio:.2f}")
        if pm.max_drawdown:
            print(f"Max Drawdown:       {pm.max_drawdown:.1%}")
    
    if wallet.behavioral_patterns:
        print("\n🎭 BEHAVIORAL PATTERNS")
        print("-" * 80)
        bp = wallet.behavioral_patterns
        
        if bp.pattern_tags:
            print("Detected Patterns:")
            for tag in bp.pattern_tags:
                print(f"  • {tag.replace('_', ' ').title()}")
        
        if bp.is_smart_money:
            print("\n⭐ Classified as Smart Money")
        
        if bp.copies_wallet:
            print(f"\n📋 Follows wallet: {bp.copies_wallet}")
        
        print(f"\nWash Trading Score: {bp.wash_trading_score:.1%}")
    
    if wallet.ranking:
        print("\n🏆 RANKINGS")
        print("-" * 80)
        rank = wallet.ranking
        if rank.global_rank:
            print(f"Global Rank: #{rank.global_rank:,}")
        if rank.chain_rank:
            print(f"Chain Rank:  #{rank.chain_rank:,}")
        print(f"Percentile:  Top {(1 - rank.percentile) * 100:.1f}%")
    
    if wallet.breakdown_by_token:
        print("\n🪙 TOP TOKENS")
        print("-" * 80)
        print(f"{'Token':<10} {'Chain':<10} {'Trades':<8} {'Win Rate':<10} {'PnL (USD)':<15}")
        print("-" * 80)
        
        for token in wallet.breakdown_by_token[:5]:
            print(f"{token.token_symbol:<10} {token.chain:<10} {token.trade_count:<8} "
                  f"{token.win_rate:<10.1%} ${token.total_pnl_usd:>13,.2f}")


def main():
    """Run the demo."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                                            ║")
    print("║                           RadarX Demo                                      ║")
    print("║                  Memecoin Analysis & Wallet Intelligence                   ║")
    print("║                                                                            ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    demo_token_scoring()
    demo_wallet_analytics()
    
    print_header("DEMO COMPLETE")
    print("✨ This demo showcased RadarX capabilities with sample data.")
    print("🚀 To use with real data, configure API keys and run the server:")
    print("   python -m radarx.api.server")
    print("📚 Visit http://localhost:8000/docs for interactive API documentation")
    print("\n")


if __name__ == "__main__":
    main()
