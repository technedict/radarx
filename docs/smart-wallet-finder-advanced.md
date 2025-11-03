# Smart Wallet Finder - Advanced Features Guide

## Overview

This guide covers the advanced machine learning and backtesting features added to the Smart Wallet Finder system.

## Advanced ML Features

### 1. Granger Causality Analysis

Tests whether wallet trading activity predicts price movements using statistical lead-lag analysis.

#### Usage

```python
from radarx.smart_wallet_finder import SmartWalletFinder

finder = SmartWalletFinder()

# Find wallets with advanced ML features
result = finder.find_smart_wallets_with_advanced_ml(
    token_address="0x1234...",
    chain="ethereum",
    window_days=30,
    enable_granger=True,
    enable_embeddings=True,
    enable_counterfactual=True,
)

# Check Granger causality for top wallet
top_wallet = result['ranked_wallets'][0]
causality = top_wallet['advanced_features']['granger_causality']

print(f"Has causality: {causality['has_causality']}")
print(f"Optimal lag: {causality['optimal_lag']} periods")
print(f"Lead score: {causality['lead_score']:.2f}")
print(f"Interpretation: {causality['interpretation']}")
```

#### Interpretation

- **has_causality**: Boolean indicating statistically significant relationship
- **optimal_lag**: Number of periods wallet leads market
- **lead_score**: 0-1 score for predictive power
- **p_values**: Statistical significance for each lag tested

### 2. Wallet Behavior Embeddings

Creates vector representations of wallet trading patterns for similarity analysis.

#### Usage

```python
# Embeddings are automatically computed
wallet1 = result['ranked_wallets'][0]
wallet2 = result['ranked_wallets'][1]

embedding1 = wallet1['advanced_features']['behavior_embedding']
embedding2 = wallet2['advanced_features']['behavior_embedding']

# Compute similarity
from radarx.smart_wallet_finder.advanced_ml import WalletBehaviorEmbedder

embedder = WalletBehaviorEmbedder()
similarity = embedder.compute_similarity(
    np.array(embedding1),
    np.array(embedding2)
)

print(f"Wallet similarity: {similarity:.2%}")
```

#### Applications

- **Strategy clustering**: Group wallets by similar trading patterns
- **Pattern detection**: Identify repeating multi-token strategies
- **Anomaly detection**: Flag unusual trading behavior

### 3. Counterfactual Impact Analysis

Estimates what would have happened if a wallet hadn't traded.

#### Usage

```python
# Counterfactual analysis in results
impact = top_wallet['advanced_features']['counterfactual_impact']

print(f"Impact score: {impact['impact_score']:.2%}")
print(f"Volume contribution: {impact['volume_contribution']:.2%}")
print(f"Price influence: {impact['price_influence']:.2%}")
print(f"Counterfactual returns: {impact['counterfactual_returns']:.2%}")
print(f"Interpretation: {impact['interpretation']}")
```

#### Metrics

- **impact_score**: Overall market impact (0-1)
- **volume_contribution**: Wallet's % of total volume
- **price_influence**: Estimated price impact magnitude
- **counterfactual_returns**: Simulated returns without wallet

## Backtesting Framework

### Overview

Validates smart wallet predictions through simulated trading.

### Basic Backtest

```python
from radarx.smart_wallet_finder.backtesting import (
    SmartWalletBacktester,
    BacktestConfig,
)

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    position_size=0.1,  # 10% per trade
    top_k=10,  # Follow top 10 wallets
    rebalance_frequency_days=7,
    fee_rate=0.003,  # 0.3%
    slippage_rate=0.001,  # 0.1%
)

backtester = SmartWalletBacktester(config)

# Run backtest
result = backtester.run_backtest(
    smart_wallet_rankings=smart_wallets,
    token_address="0x1234...",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31),
    price_data=price_timeline,
    all_trades_data=trades,
)

# View results
print(f"Total return: {result.total_return:.2%}")
print(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
print(f"Max drawdown: {result.max_drawdown:.2%}")
print(f"Win rate: {result.win_rate:.2%}")
print(f"Number of trades: {result.num_trades}")
```

### Precision@K Metrics

```python
# Precision at different K values
for k, precision in result.precision_at_k.items():
    print(f"Precision@{k}: {precision:.2%}")
```

### Portfolio Value History

```python
import matplotlib.pyplot as plt

# Plot portfolio value over time
dates = [d for d, v in result.portfolio_value_history]
values = [v for d, v in result.portfolio_value_history]

plt.figure(figsize=(12, 6))
plt.plot(dates, values)
plt.title("Portfolio Value Over Time")
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.grid(True)
plt.show()
```

### Walk-Forward Validation

```python
from radarx.smart_wallet_finder.backtesting import WalkForwardValidator

# Set up validator
validator = WalkForwardValidator(
    train_window_days=60,
    test_window_days=30,
    step_days=30,
)

# Run validation
validation_results = validator.validate(
    finder_func=lambda token, data: finder.find_smart_wallets(token, **data),
    token_address="0x1234...",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 1, 1),
    data_provider=data_provider_func,
)

print(f"Number of splits: {validation_results['num_splits']}")
print(f"Average precision: {validation_results['avg_precision']:.2%}")
print(f"Average return: {validation_results['avg_return']:.2%}")
```

## Real-Time Data Streaming

### WebSocket Monitoring

```python
from radarx.smart_wallet_finder.realtime import EventDrivenArchitecture
import asyncio

# Create event-driven system
eda = EventDrivenArchitecture()

# Register handlers
def on_new_trade(trade):
    print(f"New trade detected: {trade}")

def on_signal_update(update):
    wallet = update['wallet']
    signals = update['signals']
    print(f"Wallet {wallet} updated: {signals}")

eda.register_handler("new_trade", on_new_trade)
eda.register_handler("signal_update", on_signal_update)

# Start monitoring
async def monitor():
    await eda.start(
        token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
        chain="solana"
    )

# Run
asyncio.run(monitor())
```

### Live Signal Updates

```python
from radarx.smart_wallet_finder.realtime import LiveSignalUpdater

updater = LiveSignalUpdater()

# Register callback for updates
def on_signal_change(update):
    print(f"Signals updated for {update['wallet']}")
    print(f"New signals: {update['signals']}")

updater.register_callback(on_signal_change)

# Process live trades
async def process_trades(trade_stream):
    async for trade in trade_stream:
        await updater.process_live_trade(trade)
```

## Solana Enhanced Features

### Birdeye Historical Prices

The Solana data fetcher now uses Birdeye API for accurate historical price data.

```python
from radarx.smart_wallet_finder.solana_data_fetcher import SolanaDataFetcher

fetcher = SolanaDataFetcher(
    solscan_api_key="your_key",
    helius_api_key="your_key"
)

# Automatically uses Birdeye for historical prices
timeline = fetcher._fetch_price_timeline(
    token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    chain="solana",
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now(),
)
```

### Enhanced DEX Detection

Now identifies exact DEX programs from instruction data.

```python
# DEX identification is automatic
trades = fetcher._fetch_dex_trades(
    token_address="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    chain="solana",
    start_time=datetime.now() - timedelta(days=1),
    end_time=datetime.now(),
)

for trade in trades:
    print(f"DEX: {trade['dex']}")  # jupiter, raydium, orca, serum
```

## Configuration

### Environment Variables

Add to `.env`:

```bash
# Advanced ML Features
ENABLE_GRANGER_CAUSALITY=true
ENABLE_BEHAVIOR_EMBEDDINGS=true
ENABLE_COUNTERFACTUAL=true

# Solana Enhanced
BIRDEYE_API_KEY=your_birdeye_key

# WebSocket Streaming
ENABLE_REALTIME=true
```

## Performance Considerations

### Computational Cost

- **Granger Causality**: O(n²) where n = time series length. Recommended for top wallets only.
- **Embeddings**: O(n) where n = sequence length. Fast enough for all wallets.
- **Counterfactual**: O(n) where n = number of trades. Moderate cost.

### Recommendations

```python
# For large-scale analysis, enable selectively
if len(wallets) > 100:
    # Only use advanced features for top 20
    enable_ml = True if wallet_rank <= 20 else False
else:
    enable_ml = True
```

## Examples

### Complete Analysis Pipeline

```python
from radarx.smart_wallet_finder import SmartWalletFinder
from radarx.smart_wallet_finder.backtesting import SmartWalletBacktester, BacktestConfig
from datetime import datetime, timedelta

# 1. Find smart wallets with advanced features
finder = SmartWalletFinder()

result = finder.find_smart_wallets_with_advanced_ml(
    token_address="0x1234...",
    chain="ethereum",
    window_days=30,
    top_k=50,
    enable_granger=True,
    enable_embeddings=True,
    enable_counterfactual=True,
)

# 2. Analyze top wallets
for wallet in result['ranked_wallets'][:10]:
    print(f"\nWallet: {wallet['wallet_address']}")
    print(f"Score: {wallet['smart_money_score']:.2%}")
    
    # Granger causality
    if 'granger_causality' in wallet['advanced_features']:
        gc = wallet['advanced_features']['granger_causality']
        print(f"  Leads market: {gc['has_causality']}")
        print(f"  {gc['interpretation']}")
    
    # Impact analysis
    if 'counterfactual_impact' in wallet['advanced_features']:
        impact = wallet['advanced_features']['counterfactual_impact']
        print(f"  Market impact: {impact['impact_score']:.2%}")
        print(f"  {impact['interpretation']}")

# 3. Backtest the strategy
config = BacktestConfig(initial_capital=10000, top_k=10)
backtester = SmartWalletBacktester(config)

backtest_result = backtester.run_backtest(
    smart_wallet_rankings=result['ranked_wallets'],
    token_address="0x1234...",
    start_date=datetime.now() - timedelta(days=90),
    end_date=datetime.now(),
    price_data=result['metadata']['price_timeline'],
    all_trades_data=result['metadata']['all_trades'],
)

print(f"\nBacktest Results:")
print(f"Total Return: {backtest_result.total_return:.2%}")
print(f"Sharpe Ratio: {backtest_result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {backtest_result.max_drawdown:.2%}")
```

## Troubleshooting

### Granger Causality Fails

**Issue**: "Insufficient data for Granger causality test"

**Solution**: Ensure at least 50 data points (trades + price points):
```python
if len(trades) < 50 or len(price_timeline) < 50:
    enable_granger = False
```

### WebSocket Disconnects

**Issue**: WebSocket connection drops

**Solution**: Implement reconnection logic:
```python
async def monitor_with_reconnect():
    while True:
        try:
            await eda.start(token, chain)
        except Exception as e:
            logger.error(f"Connection lost: {e}")
            await asyncio.sleep(5)  # Wait before reconnecting
```

### Birdeye API Rate Limits

**Issue**: "Rate limit exceeded"

**Solution**: Implement caching and request throttling:
```python
# Already built-in with 5-minute cache
# For higher volume, upgrade Birdeye tier
```

## API Reference

See individual module documentation:
- `advanced_ml.py`: ML feature implementations
- `backtesting.py`: Backtesting framework
- `realtime.py`: WebSocket streaming
- `solana_data_fetcher.py`: Enhanced Solana integration
