# RadarX Architecture

This document describes the architecture and design of the RadarX system.

## System Overview

RadarX is a production-grade memecoin analysis and wallet intelligence platform built with:
- **API Layer**: FastAPI for high-performance REST endpoints
- **ML Pipeline**: Ensemble models with calibration and explainability
- **Data Layer**: Multi-source data ingestion with normalization
- **Analytics Engine**: Wallet performance tracking and behavioral analysis
- **Backtesting Framework**: Walk-forward validation with realistic simulation

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         API Layer (FastAPI)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Token   │  │  Wallet  │  │  Search  │  │  Alerts  │       │
│  │  Score   │  │  Report  │  │ Wallets  │  │Subscribe │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────┬───────────────────────────────────────────────────┘
              │
┌─────────────▼───────────────────────────────────────────────────┐
│                      Service Layer                               │
│  ┌──────────────────────┐  ┌──────────────────────┐            │
│  │ TokenScoringService  │  │WalletAnalyticsService│            │
│  └──────────────────────┘  └──────────────────────┘            │
└─────────────┬───────────────────────┬───────────────────────────┘
              │                       │
    ┌─────────▼─────────┐   ┌────────▼────────┐
    │                   │   │                  │
┌───▼──────────────┐  ┌─▼──────────────┐  ┌──▼─────────────┐
│  Data Ingestion  │  │   ML Models    │  │Wallet Analytics│
│                  │  │                │  │                │
│  • DexScreener   │  │ • Probability  │  │ • Win Rate     │
│  • Blockchain    │  │   Predictor    │  │ • PnL Tracker  │
│  • Social APIs   │  │ • Risk Scorer  │  │ • Behaviors    │
│  • Risk Feeds    │  │ • Explainer    │  │ • Rankings     │
│                  │  │ • Calibrator   │  │ • Related      │
└──────┬───────────┘  └────────┬───────┘  └────────────────┘
       │                       │
┌──────▼───────────────────────▼───────────────┐
│          Feature Engineering                  │
│  • Token Features   • Social Features         │
│  • Wallet Features  • Time Windows            │
│  • Feature Store (with time-travel)           │
└───────────────────────────────────────────────┘
```

## Component Details

### 1. API Layer (`radarx/api/`)

**Responsibilities**:
- HTTP request handling
- Input validation
- Response formatting
- Rate limiting
- CORS management
- Prometheus metrics

**Key Files**:
- `server.py` - FastAPI app and endpoints
- `services.py` - Business logic layer
- `rate_limiter.py` - Request rate limiting

**Technology**: FastAPI, Uvicorn, Pydantic

### 2. Data Ingestion (`radarx/data/`)

**Responsibilities**:
- Fetch data from external sources
- Normalize data formats
- Cache responses
- Handle API rate limits
- Validate data quality

**Key Files**:
- `dexscreener.py` - DEX price feeds
- `blockchain.py` - On-chain indexers
- `social.py` - Social media APIs
- `risk_feeds.py` - Risk assessment feeds
- `normalizer.py` - Data normalization
- `cache.py` - Caching layer

**External APIs**:
- DexScreener (token prices, liquidity)
- Etherscan, BscScan, Solscan (on-chain data)
- Twitter, Reddit APIs (social signals)
- RugCheck, GoPlus (risk assessment)

### 3. Feature Engineering (`radarx/features/`)

**Responsibilities**:
- Extract features from raw data
- Time-windowed aggregations
- Feature versioning
- Point-in-time queries

**Key Files**:
- `token_features.py` - Token-level features
- `social_features.py` - Social signal features
- `wallet_features.py` - Wallet behavioral features
- `feature_store.py` - Feature storage with time-travel
- `time_windows.py` - Windowed aggregations
- `clustering.py` - Wallet clustering

**Feature Categories**:
- **Token**: Market cap, volume, liquidity, holder distribution
- **Social**: Mention velocity, sentiment, KOL activity
- **Wallet**: Win rates, trading patterns, PnL
- **Time**: 1h, 6h, 24h, 7d, 30d windows

### 4. ML Models (`radarx/models/`)

**Responsibilities**:
- Probability predictions
- Risk scoring
- Model calibration
- Explainability
- Online learning
- Drift detection

**Key Files**:
- `probability_predictor.py` - Ensemble predictor
- `risk_scorer.py` - Multi-component risk
- `explainer.py` - SHAP-based explanations
- `calibrator.py` - Probability calibration
- `online_learner.py` - Continual learning
- `drift_detector.py` - Concept drift
- `trainer.py` - Model training CLI

**Model Types**:
- **Ensemble**: XGBoost + LightGBM + Temporal NN
- **Calibration**: Isotonic regression / Platt scaling
- **Explainability**: SHAP values
- **Online**: Incremental updates

### 5. Wallet Analytics (`radarx/wallet/`)

**Responsibilities**:
- Calculate win rates
- Track PnL (realized/unrealized)
- Detect behavioral patterns
- Rank wallets
- Find related wallets

**Key Files**:
- `analyzer.py` - Win rate and PnL
- `behavior.py` - Pattern detection
- `ranker.py` - Performance rankings
- `related.py` - Wallet correlation

**Patterns Detected**:
- Early adopter
- Diamond hands
- Swing trader
- Smart money
- Wash trader
- And 7 more patterns

### 6. Backtesting (`radarx/backtesting/`)

**Responsibilities**:
- Walk-forward backtesting
- Strategy simulation
- Outcome labeling
- Model versioning
- Performance tracking

**Key Files**:
- `engine.py` - Backtest engine
- `strategy.py` - Trading strategies
- `labeler.py` - Outcome labeling
- `ledger.py` - Model versioning
- `runner.py` - CLI interface

**Features**:
- Realistic fee/slippage simulation
- Multiple strategies (threshold, proportional, Kelly, fixed)
- Calibration metrics
- Learning ledger

### 7. Schemas (`radarx/schemas/`)

**Responsibilities**:
- Data validation
- Type safety
- API contracts
- Response formatting

**Key Files**:
- `token.py` - Token-related schemas
- `wallet.py` - Wallet-related schemas
- `responses.py` - API response models

**Technology**: Pydantic v2

## Data Flow

### Token Scoring Flow

```
1. API Request → Validate input
2. Fetch raw data → DexScreener, Blockchain, Social
3. Extract features → Token features, Social features
4. Run models → Probability predictor, Risk scorer
5. Generate explanations → SHAP explainer
6. Calibrate → Probability calibrator
7. Format response → TokenScore schema
8. Return to client
```

### Wallet Analytics Flow

```
1. API Request → Validate wallet address
2. Fetch transactions → Blockchain indexers
3. Calculate metrics → Win rate, PnL
4. Detect patterns → Behavioral analyzer
5. Find related → Related wallet finder
6. Calculate rankings → Wallet ranker
7. Format response → WalletReport schema
8. Return to client
```

## Technology Stack

### Core
- **Language**: Python 3.9+
- **API**: FastAPI 0.104+
- **Server**: Uvicorn
- **Validation**: Pydantic v2

### Machine Learning
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch, Transformers
- **Explainability**: SHAP
- **Time Series**: statsmodels, lifelines
- **Online Learning**: River

### Data & Storage
- **Database**: SQLAlchemy (PostgreSQL/SQLite)
- **Cache**: Redis
- **NoSQL**: MongoDB (optional)
- **Processing**: pandas, numpy

### External APIs
- **HTTP Clients**: httpx, aiohttp, requests
- **Web3**: web3.py, solana-py
- **Social**: tweepy, praw

### Monitoring
- **Metrics**: Prometheus
- **Errors**: Sentry
- **Logging**: Python logging

## Design Patterns

### 1. Service Layer Pattern
- Business logic separated from API endpoints
- Enables testing without HTTP overhead
- Supports multiple API versions

### 2. Repository Pattern
- Data access abstracted
- Easy to swap data sources
- Mockable for testing

### 3. Factory Pattern
- Model creation and initialization
- Feature extractor selection
- Strategy instantiation

### 4. Observer Pattern
- Alert subscriptions
- Event notifications
- Drift detection

## Scalability Considerations

### Horizontal Scaling
- Stateless API servers
- Shared cache (Redis)
- Database connection pooling

### Performance
- Async I/O for external APIs
- Feature caching
- Model caching
- Response caching

### Reliability
- Circuit breakers for external APIs
- Retry logic with exponential backoff
- Graceful degradation
- Health checks

## Security

### API Security
- Rate limiting (60/min, 1000/hour)
- CORS configuration
- Input validation
- No PII in responses

### Data Security
- Wallet addresses are pseudonymous
- No private keys stored
- API keys in environment variables
- Secure communication (HTTPS in production)

## Monitoring

### Metrics Tracked
- Request counts by endpoint
- Request durations
- Prediction counts by chain
- Error counts by type
- Model drift indicators
- Feature availability

### Observability
- Prometheus `/metrics` endpoint
- Structured logging with correlation IDs
- Error tracking (Sentry)
- Request/response logging

## Future Enhancements

### Planned Features
- Real-time streaming with Kafka
- Advanced caching strategies
- Multi-model ensembles
- Automated retraining
- A/B testing framework
- GraphQL API

### Performance Improvements
- Model quantization
- Feature precomputation
- Distributed training
- GPU acceleration

## See Also

- [DEPLOYMENT.md](../DEPLOYMENT.md) - Deployment guide
- [OPERATIONS.md](../OPERATIONS.md) - Operations runbook
- [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md) - Development plan
