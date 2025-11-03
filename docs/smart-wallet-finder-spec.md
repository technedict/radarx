# Smart Wallet Finder - Feature Specification

## Overview

The Smart Wallet Finder is an advanced feature that discovers probable "smart money" wallets for a given token by analyzing on-chain trading patterns, timing signals, profitability metrics, transaction graphs, and behavioral fingerprints.

## Architecture

### Components

1. **Data Fetcher** (`data_fetcher.py`)
   - Fetches on-chain data from blockchain indexers (Etherscan, Solscan, etc.)
   - Retrieves DEX trades from DexScreener or swap feeds
   - Constructs price timelines
   - Builds transaction graphs

2. **Trade Matcher** (`trade_matcher.py`)
   - Matches buy and sell trades using FIFO accounting
   - Handles routing, slippage, and internal transfers
   - Computes realized PnL and ROI per trade
   - Calculates holding durations

3. **Signal Detectors** (`signals.py`)
   - **TimingSignalDetector**: Detects pre-pump buys and pre-dump sells
   - **ProfitabilityAnalyzer**: Computes win rate, ROI, Sharpe ratio
   - **GraphAnalyzer**: Analyzes centrality, clustering, fund flows
   - **BehavioralAnalyzer**: Detects trading patterns and fingerprints

4. **Risk Filter** (`risk_filter.py`)
   - Detects wash trading patterns
   - Identifies bot-like behavior
   - Filters circular trades
   - Flags rapid-fire trading

5. **Wallet Scorer** (`scorer.py`)
   - Combines multiple signals with weighted ensemble
   - Applies probability calibration
   - Produces smart-money confidence score (0-1)

6. **Explainer** (`explainer.py`)
   - Generates human-readable explanations
   - Identifies top contributing signals
   - Creates time-aligned event timelines
   - Provides interpretations

7. **Smart Wallet Finder** (`finder.py`)
   - Main orchestrator coordinating all components
   - Manages end-to-end pipeline
   - Produces ranked wallet lists

## Detection Signals

### 1. Event Timing Signals

**Purpose**: Identify wallets that enter before pumps and exit before dumps.

**Metrics**:
- `pre_pump_entry_rate`: Ratio of buys occurring within N minutes before price pumps
- `pre_dump_exit_rate`: Ratio of sells occurring within N minutes before price dumps
- `avg_entry_timing_minutes`: Average time between buy and next price peak
- `avg_exit_timing_minutes`: Average time between sell and next price trough
- `pumps_detected`: Number of significant price increases detected
- `dumps_detected`: Number of significant price decreases detected

**Configuration**:
```python
TimingSignalDetector(
    pre_pump_window_minutes=60,      # Lead window for pump detection
    pre_dump_window_minutes=60,      # Lead window for dump detection
    pump_threshold_pct=20.0,         # Minimum price increase for pump
    dump_threshold_pct=-15.0,        # Maximum price decrease for dump
)
```

### 2. Profitability Signals

**Purpose**: Measure trading performance and consistency.

**Metrics**:
- `win_rate`: Percentage of profitable closed positions
- `avg_roi`: Average return on investment
- `median_roi`: Median ROI (robust to outliers)
- `sharpe_ratio`: Risk-adjusted returns
- `avg_holding_duration_hours`: Average position holding time
- `best_roi`: Best single trade ROI
- `worst_roi`: Worst single trade ROI

**Formula**:
```
Win Rate = Profitable Trades / Total Trades
ROI = (Exit Value - Entry Cost) / Entry Cost
Sharpe Ratio = Mean(ROI) / StdDev(ROI)
```

### 3. Graph Analysis Signals

**Purpose**: Analyze wallet position in transaction network.

**Metrics**:
- `centrality_score`: Degree centrality in transaction graph (0-1)
- `clustering_coefficient`: Local clustering coefficient (0-1)
- `common_funding_sources`: Number of shared funding addresses
- `cluster_id`: Community cluster assignment
- `connected_smart_wallets`: Connections to known smart wallets

**Formula**:
```
Degree Centrality = Neighbors / (Total Nodes - 1)
Clustering Coefficient = Edges Between Neighbors / Max Possible Edges
```

### 4. Behavioral Signals

**Purpose**: Detect trading patterns and behavioral fingerprints.

**Metrics**:
- `trade_frequency_per_day`: Average daily trade count
- `preferred_trading_hours`: Most common trading hours (UTC)
- `dex_diversity`: Number of unique DEX protocols used
- `gas_pattern_consistency`: Gas price consistency score (0-1)
- `pattern_tags`: Detected behavioral patterns

**Pattern Tags**:
- `early_adopter`: Majority of trades within 24h of token launch
- `diamond_hands`: Average holding > 7 days
- `swing_trader`: Holding duration 1h - 7 days
- `high_frequency`: >10 trades per day

### 5. Risk Filters

**Purpose**: Filter out wash trading, bots, and suspicious wallets.

**Risk Indicators**:
- **Wash Trading**: High frequency + near-zero net ROI + balanced buy/sell ratio
- **Bot Behavior**: Extremely consistent gas prices + very high frequency
- **Circular Trading**: High clustering + low centrality
- **Rapid Trading**: Very short holding (<1h) + high frequency

**Risk Score**: Composite score 0 (safe) to 1 (high risk)

**Threshold**: Wallets with risk score > 0.7 are filtered out

## Scoring Model

### Weighted Ensemble

```python
smart_money_score = (
    0.30 * timing_score +
    0.35 * profitability_score +
    0.15 * graph_score +
    0.20 * behavioral_score
)
```

### Sub-Score Calculations

**Timing Score**:
```python
timing_score = (
    0.4 * pre_pump_entry_rate +
    0.4 * pre_dump_exit_rate +
    0.2 * normalize_timing(avg_entry_timing_minutes)
)
```

**Profitability Score**:
```python
profitability_score = (
    0.4 * win_rate +
    0.4 * normalize_roi(avg_roi) +
    0.2 * normalize_sharpe(sharpe_ratio)
)
```

**Graph Score**:
```python
graph_score = (
    0.3 * centrality_score +
    0.3 * clustering_coefficient +
    0.4 * normalize_connections(connected_smart_wallets)
)
```

**Behavioral Score**:
```python
behavioral_score = (
    0.3 * normalize_diversity(dex_diversity) +
    0.3 * gas_pattern_consistency +
    0.4 * score_patterns(pattern_tags)
)
```

### Calibration

Final score is calibrated using sigmoid function:
```python
calibrated_score = 1 / (1 + exp(-6 * (raw_score - 0.5)))
```

This maps [0, 1] → [0, 1] with steeper response in the middle range.

## API Endpoints

### 1. Find Smart Wallets

**Endpoint**: `POST /smart-wallets/find`

**Request**:
```json
{
  "token_address": "0x1234...",
  "chain": "ethereum",
  "window_days": 30,
  "min_trade_size_usd": 100,
  "min_holdings_usd": 500,
  "include_internal_transfers": false,
  "top_k": 100,
  "min_confidence": 0.5
}
```

**Response**:
```json
{
  "token_address": "0x1234...",
  "chain": "ethereum",
  "analysis_window_days": 30,
  "timestamp": "2024-01-15T10:30:00Z",
  "ranked_wallets": [
    {
      "rank": 1,
      "wallet_address": "0xabcd...",
      "smart_money_score": 0.87,
      "key_metrics": {
        "win_rate": 0.78,
        "realized_roi": 2.45,
        "trades_count": 15,
        "early_entry_rate": 0.73,
        "graph_centrality": 0.65
      },
      "explanation": {
        "summary": "Wallet 0xabcd... has an 87% probability...",
        "top_signals": [...],
        "interpretation": "Very high confidence...",
        "timeline": [...]
      },
      "risk_score": 0.15
    }
  ],
  "summary_stats": {
    "avg_smart_money_score": 0.68,
    "median_smart_money_score": 0.71,
    "avg_win_rate": 0.72,
    "total_smart_wallets": 45
  },
  "metadata": {
    "total_wallets_analyzed": 523,
    "wallets_passing_filters": 78,
    "wallets_returned": 45,
    "confidence_threshold": 0.5
  }
}
```

### 2. Get Wallet Profile

**Endpoint**: `POST /smart-wallets/profile`

**Request**:
```json
{
  "wallet_address": "0xabcd...",
  "token_address": "0x1234...",
  "chain": "ethereum",
  "window_days": 30
}
```

**Response**:
```json
{
  "wallet_address": "0xabcd...",
  "token_address": "0x1234...",
  "chain": "ethereum",
  "score": 0.87,
  "trades": [
    {
      "timestamp": "2024-01-10T14:30:00Z",
      "side": "buy",
      "amount_usd": 1500,
      "price": 0.05,
      "pnl": 750,
      "roi": 0.5
    }
  ],
  "realized_roi": 2.45,
  "win_rate": 0.78,
  "graph_neighbors": ["0xdef...", "0x456..."],
  "explanation": {...}
}
```

### 3. Bulk Scan Tokens

**Endpoint**: `POST /smart-wallets/bulk-scan`

**Request**:
```json
{
  "token_addresses": ["0x1234...", "0x5678...", "0x9abc..."],
  "chain": "ethereum",
  "window_days": 30,
  "top_k_per_token": 10,
  "min_confidence": 0.6
}
```

**Response**:
```json
{
  "chain": "ethereum",
  "tokens_analyzed": 3,
  "timestamp": "2024-01-15T10:30:00Z",
  "results": [
    {
      "token_address": "0x1234...",
      "top_wallets": [...],
      "avg_score": 0.72
    }
  ],
  "leaderboard": [
    {
      "rank": 1,
      "wallet_address": "0xabcd...",
      "smart_money_score": 0.92,
      ...
    }
  ]
}
```

## Usage Examples

### Python Client

```python
import requests

# Find smart wallets for a token
response = requests.post(
    "http://localhost:8000/smart-wallets/find",
    json={
        "token_address": "0x1234567890abcdef1234567890abcdef12345678",
        "chain": "ethereum",
        "window_days": 30,
        "top_k": 50,
        "min_confidence": 0.6
    }
)

result = response.json()
print(f"Found {len(result['ranked_wallets'])} smart wallets")

for wallet in result['ranked_wallets'][:5]:
    print(f"#{wallet['rank']}: {wallet['wallet_address']}")
    print(f"  Score: {wallet['smart_money_score']:.2%}")
    print(f"  Win Rate: {wallet['key_metrics']['win_rate']:.2%}")
    print(f"  ROI: {wallet['key_metrics']['realized_roi']:.2f}x")
```

### cURL

```bash
curl -X POST http://localhost:8000/smart-wallets/find \
  -H "Content-Type: application/json" \
  -d '{
    "token_address": "0x1234567890abcdef1234567890abcdef12345678",
    "chain": "ethereum",
    "window_days": 30,
    "top_k": 100,
    "min_confidence": 0.5
  }'
```

## Performance Considerations

### Scalability

- **Data Fetching**: Parallel fetching from multiple sources
- **Signal Computation**: Vectorized operations using NumPy
- **Graph Analysis**: Efficient graph algorithms for large networks
- **Caching**: Cache price timelines and graph data

### Latency

- **Expected Response Time**: 2-10 seconds for single token
- **Bulk Scan**: 5-30 seconds for 10 tokens (parallel processing)
- **Optimization**: Pre-computed graph metrics, indexed data

### Resource Requirements

- **Memory**: ~100MB per 1000 wallets analyzed
- **CPU**: Multi-core for parallel signal computation
- **Storage**: Graph database for transaction networks

## Future Enhancements

### Advanced ML Features

1. **Causal Lead-Lag Modeling**
   - Granger causality tests between wallet activity and price moves
   - Vector autoregression for temporal dependencies

2. **Sequence Embedding**
   - Transformer models for wallet behavior sequences
   - Contrastive learning for similar strategy detection

3. **Temporal Graph Neural Networks**
   - Dynamic GNNs for evolving transaction networks
   - Attention mechanisms for important connections

4. **Counterfactual Analysis**
   - Simulate removing wallets to estimate market impact
   - Causal inference for wallet influence

### Continuous Learning

1. **Active Learning Loop**
   - Surface ambiguous wallets for manual review
   - Incorporate human labels into training data

2. **Model Retraining**
   - Auto-trigger on performance degradation
   - Walk-forward validation on new data

3. **Drift Detection**
   - Monitor feature distributions
   - Alert on concept drift

## Validation & Testing

### Unit Tests

- Trade matching logic
- Signal computation accuracy
- Graph construction
- Score calculation
- Risk filtering

### Integration Tests

- End-to-end pipeline
- API endpoint responses
- Schema validation
- Error handling

### Backtesting

- Historical wallet prediction accuracy
- Precision@K metrics
- Portfolio simulation following top-K wallets
- Comparison against random baseline

## Security & Privacy

### Constraints

- **No Private Keys**: System never accesses or stores private keys
- **No Transactions**: Read-only analysis, no on-chain transactions
- **Pseudonymous**: Wallet addresses are public blockchain data

### Data Privacy

- Cache expiration for user queries
- No PII storage
- Compliance with data provider licenses

## Monitoring

### Key Metrics

- **Prediction Quality**: Track prediction accuracy over time
- **API Performance**: Response times, error rates
- **Data Freshness**: Age of on-chain data
- **User Engagement**: Most queried tokens, usage patterns

### Alerts

- Model drift detection
- API errors or timeouts
- Data source outages
- Unusual activity patterns
