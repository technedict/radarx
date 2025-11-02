# RadarX - Production-Grade Memecoin Analysis System

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![API](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)

RadarX is a comprehensive memecoin analysis and wallet intelligence platform that provides ML-driven probability predictions, risk assessment, and explainable signals for cryptocurrency traders.

## 🎯 Features

### Token Scoring
- **Probability Heatmaps**: Calibrated probabilities for 2x, 5x, 10x, 20x, 50x multipliers across multiple time horizons (24h, 7d, 30d)
- **Risk Assessment**: Composite risk scores with breakdown into rug risk, dev risk, distribution risk, social manipulation risk, and liquidity risk
- **Explainable AI**: Top contributing features for each prediction with direction and impact
- **Real-time Analysis**: Live scoring based on current market conditions and on-chain data

### Wallet Analytics
- **Win Rate Tracking**: Percentage of profitable trades across multiple timeframes
- **PnL Analysis**: Realized and unrealized profit/loss with detailed breakdowns
- **Behavioral Patterns**: Detection of trading patterns (early adopter, diamond hands, swing trader, etc.)
- **Smart Money Detection**: Identify wallets with consistently profitable strategies
- **Related Wallets**: Find correlated wallets through fund flow and pattern analysis
- **Global Rankings**: Performance rankings and percentile calculations

### Data Integration
- DEX price feeds (DexScreener)
- On-chain indexers (Etherscan, BscScan, Solscan)
- Social signals (Twitter, Telegram, Reddit)
- Risk feeds (RugCheck, GoPlus)
- Multi-chain support (Ethereum, BSC, Solana, and more)

### Machine Learning
- Hybrid ML architecture (gradient boosting + temporal neural networks)
- Calibrated probability predictions with confidence intervals
- Online learning with concept drift detection
- Survival analysis for censored targets
- SHAP-based explainability

### Backtesting
- Walk-forward backtesting framework
- Realistic fee and slippage simulation
- Strategy simulation and optimization
- Calibration metrics and performance tracking
- Learning ledger for model versioning

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/technedict/radarx.git
cd radarx

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Configuration

Create a `.env` file in the project root:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./radarx.db
REDIS_URL=redis://localhost:6379/0

# API Keys (optional for demo)
DEXSCREENER_API_KEY=your_key_here
ETHERSCAN_API_KEY=your_key_here
TWITTER_BEARER_TOKEN=your_token_here

# Model Settings
MODEL_VERSION=v1.2.3
FEATURE_VERSION=v2.0.1
ENABLE_ONLINE_LEARNING=true
```

### Running the API Server

```bash
# Start the API server
python -m radarx.api.server

# Or use the CLI command
radarx-server
```

The API will be available at `http://localhost:8000`. Visit `http://localhost:8000/docs` for interactive API documentation.

### Running a Sanity Backtest

```bash
# Run backtest on sample data
python -m radarx.backtesting.runner --start-date 2023-01-01 --end-date 2024-01-01
```

## 📚 API Endpoints

### Token Scoring

**GET** `/score/token`

Score a token with probability heatmaps and risk assessment.

```bash
curl "http://localhost:8000/score/token?address=0x1234...&chain=ethereum&horizons=24h,7d,30d"
```

**Response**: Complete token analysis including probability heatmap, risk score, feature contributions, and optional raw features/timelines.

### Wallet Report

**GET** `/wallet/report`

Get comprehensive wallet analytics.

```bash
curl "http://localhost:8000/wallet/report?address=0xabcd...&period=30d"
```

**Response**: Win rates, PnL summary, trade breakdowns, behavioral patterns, and rankings.

### Wallet Search

**GET** `/search/wallets`

Discover top-performing wallets.

```bash
curl "http://localhost:8000/search/wallets?min_win_rate=0.6&min_trades=10&sort_by=win_rate&limit=100"
```

**Response**: List of wallets matching criteria with performance metrics.

### Alert Subscription

**POST** `/alerts/subscribe`

Subscribe to alerts via webhook.

```bash
curl -X POST "http://localhost:8000/alerts/subscribe?webhook_url=https://example.com/webhook&min_probability_10x=0.5"
```

**Response**: Subscription confirmation with subscription ID.

## 📊 Sample API Responses

### Token Score Response

```json
{
  "token_address": "0x1234567890abcdef1234567890abcdef12345678",
  "chain": "ethereum",
  "timestamp": "2024-01-15T10:30:00Z",
  "probability_heatmap": {
    "horizons": {
      "24h": {
        "2x": {
          "probability": 0.35,
          "confidence_interval": {
            "lower": 0.28,
            "upper": 0.42,
            "confidence_level": 0.95
          }
        },
        "10x": {
          "probability": 0.03,
          "confidence_interval": {
            "lower": 0.01,
            "upper": 0.05,
            "confidence_level": 0.95
          }
        }
      }
    }
  },
  "risk_score": {
    "composite_score": 45.5,
    "components": {
      "rug_risk": 30.0,
      "dev_risk": 50.0,
      "distribution_risk": 40.0,
      "social_manipulation_risk": 55.0,
      "liquidity_risk": 35.0
    },
    "risk_flags": ["high_dev_holding", "recent_dev_sell"]
  },
  "explanations": {
    "probability_2x_24h": {
      "top_features": [
        {
          "feature_name": "volume_momentum_1h",
          "contribution": 0.15,
          "direction": "positive",
          "description": "Strong positive volume momentum in last hour"
        }
      ]
    }
  }
}
```

See `examples/sample_responses.py` for complete sample responses.

### Wallet Report Response

```json
{
  "wallet_address": "0xabcdef1234567890abcdef1234567890abcdef12",
  "chain": "multi-chain",
  "timestamp": "2024-01-15T10:30:00Z",
  "win_rate": {
    "overall": 0.68,
    "by_timeframe": {
      "1d": 0.75,
      "7d": 0.71,
      "30d": 0.69,
      "all_time": 0.68
    },
    "profitable_trades": 68,
    "total_trades": 100
  },
  "pnl_summary": {
    "realized_pnl": {
      "total_usd": 187500.0,
      "average_per_trade_usd": 1875.0,
      "best_trade_usd": 45000.0,
      "worst_trade_usd": -8500.0
    },
    "total_volume": {
      "buy_volume_usd": 425000.0,
      "sell_volume_usd": 612500.0,
      "total_volume_usd": 1037500.0
    }
  },
  "behavioral_patterns": {
    "pattern_tags": ["early_adopter", "swing_trader", "smart_money_follower"],
    "is_smart_money": true,
    "wash_trading_score": 0.05
  }
}
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=radarx --cov-report=html

# Run specific test file
pytest tests/unit/test_api.py

# Run integration tests
pytest tests/integration/
```

## 🏗️ Architecture

### Components

```
radarx/
├── api/              # FastAPI REST API
│   ├── server.py     # API endpoints
│   ├── services.py   # Business logic
│   └── rate_limiter.py
├── models/           # ML models
│   └── __init__.py   # Predictors, scorers, explainers
├── features/         # Feature engineering
│   └── __init__.py   # Feature extractors, feature store
├── data/             # Data ingestion
│   └── __init__.py   # Data source adapters
├── wallet/           # Wallet analytics
│   └── __init__.py   # Win rate, PnL, patterns
├── backtesting/      # Backtesting framework
│   └── __init__.py   # Backtest engine, strategies
├── schemas/          # Pydantic models
│   ├── token.py      # Token schemas
│   └── wallet.py     # Wallet schemas
└── utils/            # Utilities
```

### Data Flow

1. **Ingestion**: Data sources → Adapters → Normalization
2. **Features**: Raw data → Feature extractors → Feature store
3. **Inference**: Features → ML models → Predictions
4. **Explanation**: Predictions → SHAP → Feature contributions
5. **API**: Requests → Services → Response models

### ML Pipeline

1. **Feature Engineering**: Extract token, liquidity, holder, social, and dev features
2. **Model Ensemble**: Gradient boosting + temporal neural network
3. **Calibration**: Isotonic/Platt scaling for probability calibration
4. **Online Learning**: Continual updates with drift detection
5. **Backtesting**: Walk-forward validation with realistic simulation

## 📈 Monitoring in Production

### Key Metrics to Monitor

- **Score Distribution Shifts**: Detect changes in probability distributions
- **Feature Availability**: Track upstream data lag and missing features
- **Backtest vs Live Performance**: Monitor divergence
- **Alert Volume**: Track false positive rates
- **API Latency**: Ensure sub-second response times
- **Model Drift**: Detect feature and concept drift
- **User Feedback**: Collect manual labels for model improvement

### Observability

- Prometheus metrics endpoint at `/metrics` (when enabled)
- Structured logging with correlation IDs
- Sentry error tracking (optional)
- Request/response logging for debugging

## 🛡️ Security & Compliance

- Rate limiting: 60 requests/minute, 1000 requests/hour per client
- CORS configuration for allowed origins
- No real-time PII exposure (wallet addresses are pseudonymous)
- Clear disclaimers: outputs are probabilistic, not financial advice
- Data provider license compliance
- Optional opt-out for wallet analytics

## 🔧 Development

### Code Quality

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. Define feature extractors in `features/`
2. Update ML models in `models/`
3. Add API endpoints in `api/server.py`
4. Update schemas in `schemas/`
5. Write tests in `tests/`
6. Update documentation

## 📋 Project Milestones

### Phase 1: Foundation ✅
- [x] Project structure and configuration
- [x] JSON schemas for API responses
- [x] Pydantic models for validation
- [x] FastAPI application with core endpoints

### Phase 2: Data & Features ✅
- [x] Data source adapters implementation
- [x] Feature extraction pipeline
- [x] Feature store with time-travel
- [x] Data normalization and validation

### Phase 3: ML Models ✅
- [x] Gradient boosting models
- [x] Temporal neural networks
- [x] Model calibration pipeline
- [x] Explainability integration

### Phase 4: Wallet Analytics ✅
- [x] Win rate calculation
- [x] PnL tracking system
- [x] Behavioral pattern detection
- [x] Wallet ranking system

### Phase 5: Backtesting ✅
- [x] Backtest engine implementation
- [x] Strategy simulator
- [x] Outcome labeling
- [x] Learning ledger

### Phase 6: Production ✅
- [x] Streaming pipeline documentation
- [x] Model serving infrastructure
- [x] Monitoring and alerting (Prometheus metrics)
- [x] Documentation and deployment guides

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For questions and support:
- GitHub Issues: [github.com/technedict/radarx/issues](https://github.com/technedict/radarx/issues)
- Documentation: See `/docs` directory

## ⚠️ Disclaimer

RadarX provides probabilistic predictions and analytics for educational and informational purposes only. This is NOT financial advice. Cryptocurrency trading involves significant risk. Always do your own research and consult with financial professionals before making investment decisions. Past performance does not guarantee future results.