# RadarX Quick Start Guide

This guide will help you get RadarX up and running in minutes.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/technedict/radarx.git
cd radarx
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt

# Install RadarX in development mode
pip install -e .
```

### 4. Configure Environment (Optional)

For demo purposes, the system works without configuration. For production use:

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your API keys
nano .env  # or use your preferred editor
```

## Running the API Server

### Start the Server

```bash
# Method 1: Using Python module
python -m radarx.api.server

# Method 2: Using CLI command (if installed)
radarx-server
```

The server will start on `http://localhost:8000`

### Verify Server is Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "0.1.0"
}
```

## Using the API

### Interactive Documentation

Visit `http://localhost:8000/docs` in your browser to access the interactive Swagger UI documentation.

### Example API Calls

#### 1. Score a Token

```bash
curl "http://localhost:8000/score/token?address=0x1234567890abcdef1234567890abcdef12345678&chain=ethereum&horizons=24h,7d,30d"
```

#### 2. Get Wallet Report

```bash
curl "http://localhost:8000/wallet/report?address=0xabcdef1234567890abcdef1234567890abcdef12&period=30d"
```

#### 3. Search for Top Wallets

```bash
curl "http://localhost:8000/search/wallets?min_win_rate=0.6&min_trades=10&sort_by=win_rate&limit=10"
```

### Using the Python Example Script

```bash
# Make sure the server is running, then:
cd examples
python api_usage.py
```

This will demonstrate all major API endpoints with formatted output.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=radarx --cov-report=html

# Run specific test file
pytest tests/unit/test_api.py -v

# Run only integration tests
pytest tests/integration/ -v
```

View coverage report:
```bash
# Coverage report is generated in htmlcov/
# Open htmlcov/index.html in your browser
```

## Understanding the Response

### Token Score Response

When you score a token, you'll receive:

1. **Probability Heatmap**: Chances of reaching 2x, 5x, 10x, 20x, 50x gains
   - Each probability includes confidence intervals
   - Available for multiple time horizons (24h, 7d, 30d)

2. **Risk Score**: Composite risk from 0-100
   - Component breakdown (rug risk, dev risk, etc.)
   - Risk flags highlighting concerns

3. **Explanations**: Top 5 features influencing the prediction
   - Feature name and contribution
   - Direction (positive/negative)
   - Human-readable description

Example response structure:
```json
{
  "probability_heatmap": {
    "horizons": {
      "24h": {
        "2x": {"probability": 0.35, "confidence_interval": {...}},
        "10x": {"probability": 0.03, "confidence_interval": {...}}
      }
    }
  },
  "risk_score": {
    "composite_score": 45.5,
    "components": {...}
  },
  "explanations": {...}
}
```

### Wallet Report Response

When you request a wallet report, you'll receive:

1. **Win Rate**: Percentage of profitable trades
   - Overall and by timeframe (1d, 7d, 30d, all-time)
   - Profitable vs. total trade counts

2. **PnL Summary**: Profit and loss metrics
   - Realized PnL (closed trades)
   - Unrealized PnL (open positions)
   - Total volume traded

3. **Breakdowns**: Performance by token and chain

4. **Behavioral Patterns**: Detected trading patterns
   - Pattern tags (e.g., "early_adopter", "swing_trader")
   - Smart money indicators
   - Bot-like behavior detection

5. **Rankings**: Global and chain-specific ranks

## Development Workflow

### Making Changes

1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes to the code

3. Run tests to ensure nothing broke:
```bash
pytest
```

4. Format your code:
```bash
black src/ tests/
isort src/ tests/
```

5. Commit and push:
```bash
git add .
git commit -m "Description of changes"
git push origin feature/your-feature-name
```

### Adding New Features

1. **New API Endpoint**: Add to `src/radarx/api/server.py`
2. **New Schema**: Add to `src/radarx/schemas/`
3. **New Feature Extractor**: Add to `src/radarx/features/`
4. **New ML Model**: Add to `src/radarx/models/`
5. **Tests**: Always add tests in `tests/`

## Troubleshooting

### Server won't start

**Error**: `ModuleNotFoundError: No module named 'radarx'`

**Solution**: Install the package in development mode:
```bash
pip install -e .
```

### Import errors

**Error**: `ImportError: cannot import name 'X' from 'radarx'`

**Solution**: Make sure you've installed all dependencies:
```bash
pip install -r requirements.txt
```

### Port already in use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**: Either:
- Stop the other process using port 8000
- Change the port in `.env` or via environment variable:
```bash
API_PORT=8001 python -m radarx.api.server
```

### Tests fail

**Error**: Tests fail with import errors

**Solution**: Make sure the test environment is set up:
```bash
pip install pytest pytest-asyncio pytest-cov
export PYTHONPATH=src:$PYTHONPATH
pytest
```

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Read the Documentation**: Check out README.md for detailed information
3. **Review the Architecture**: See IMPLEMENTATION_PLAN.md for system design
4. **Run Examples**: Try the example scripts in `examples/`
5. **Add Real Data**: Implement data source adapters in Phase 6

## Sample Data

The current implementation uses **mock data** for demonstration. To see what real responses look like:

```bash
# View sample token score
cat examples/sample_token_score.json | jq

# View sample wallet report
cat examples/sample_wallet_report.json | jq
```

## Getting Help

- **GitHub Issues**: https://github.com/technedict/radarx/issues
- **Documentation**: See README.md and docs in repository
- **API Docs**: http://localhost:8000/docs (when server is running)

## Important Notes

⚠️ **Current Status**: This is a foundation release with mock data. Real data integration, feature engineering, and ML models are in development.

⚠️ **Not Financial Advice**: RadarX provides probabilistic predictions for educational purposes only. Always do your own research.

⚠️ **Development Mode**: The system is currently in active development. APIs may change.

## What's Next?

Check out `IMPLEMENTATION_PLAN.md` for the complete roadmap and upcoming features.

---

**Last Updated**: 2024-11-02
**Version**: 0.1.0
