# Troubleshooting Guide

This guide helps you diagnose and fix common issues with RadarX.

## Installation Issues

### Dependencies Installation Fails

**Problem**: `pip install -r requirements.txt` fails with package conflicts.

**Solution**:
```bash
# Use a clean virtual environment
python -m venv venv_clean
source venv_clean/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Import Errors After Installation

**Problem**: `ModuleNotFoundError: No module named 'radarx'`

**Solution**:
```bash
# Install in editable mode
pip install -e .

# Verify installation
python -c "import radarx; print(radarx.__version__)"
```

## API Server Issues

### Server Won't Start

**Problem**: `Address already in use` error.

**Solution**:
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
API_PORT=8001 python -m radarx.api.server
```

### 500 Internal Server Errors

**Problem**: API returns 500 errors on all requests.

**Solution**:
1. Check server logs for stack traces
2. Verify configuration in `.env` file
3. Ensure all required services are running (Redis, PostgreSQL if used)

```bash
# Enable verbose logging
LOG_LEVEL=DEBUG python -m radarx.api.server
```

### Slow API Response Times

**Problem**: API responses take >5 seconds.

**Possible Causes**:
- External API rate limits
- Missing cache configuration
- Database connection issues

**Solution**:
```bash
# Check Redis cache is running
redis-cli ping

# Enable request logging to identify slow endpoints
# Add to .env:
LOG_REQUESTS=true
```

## Data Ingestion Issues

### API Rate Limit Errors

**Problem**: `429 Too Many Requests` from external APIs.

**Solution**:
1. Implement backoff strategy (already in code)
2. Enable caching:
```python
# In .env
ENABLE_CACHE=true
CACHE_TTL=300  # 5 minutes
```

### Missing Data from External Sources

**Problem**: Token data returns empty or null values.

**Solution**:
1. Verify API keys are correct
2. Check API key permissions
3. Verify token address format

```bash
# Test API connectivity
curl "https://api.dexscreener.com/latest/dex/tokens/0x..."
```

## Model Issues

### Predictions Always Return 0 or 1

**Problem**: Probabilities are not calibrated.

**Solution**:
```bash
# Retrain model with calibration
radarx-train --model-type probability --output-dir ./models

# Verify calibration
python -c "from radarx.models.calibrator import ProbabilityCalibrator; print('Calibrator available')"
```

### Model Training Fails

**Problem**: `radarx-train` crashes with memory error.

**Solution**:
```bash
# Reduce batch size or use incremental learning
# Check available memory
free -h

# For large datasets, use online learning
python -c "from radarx.models.online_learner import OnlineLearner; learner = OnlineLearner(); print('Online learner ready')"
```

## Database Issues

### SQLite Lock Errors

**Problem**: `database is locked` error.

**Solution**:
```bash
# Switch to PostgreSQL for production
# In .env:
DATABASE_URL=postgresql://user:pass@localhost/radarx
```

### Redis Connection Errors

**Problem**: `Connection refused` to Redis.

**Solution**:
```bash
# Start Redis
redis-server

# Or update Redis URL in .env
REDIS_URL=redis://localhost:6379/0
```

## Backtesting Issues

### Backtest Returns No Results

**Problem**: `radarx-backtest` runs but returns empty results.

**Solution**:
1. Verify date range has data
2. Check data loading function
3. Enable verbose logging

```bash
radarx-backtest --start-date 2023-01-01 --end-date 2024-01-01 --verbose
```

### Unrealistic Backtest Results

**Problem**: Backtest shows 100% win rate or impossible returns.

**Solution**:
1. Verify fee and slippage parameters
2. Check for look-ahead bias
3. Review strategy logic

```bash
# Use realistic fees
radarx-backtest --start-date 2023-01-01 --end-date 2024-01-01 \
                --fee-rate 0.003 --slippage-rate 0.002
```

## Testing Issues

### Tests Fail to Run

**Problem**: `pytest` command not found or tests fail.

**Solution**:
```bash
# Install test dependencies
pip install -e ".[dev]"

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_api.py -v
```

### Tests Hang or Timeout

**Problem**: Tests run indefinitely.

**Solution**:
```bash
# Use timeout
pytest --timeout=30

# Identify slow tests
pytest --durations=10
```

## Performance Issues

### High Memory Usage

**Problem**: Process uses excessive RAM.

**Solution**:
1. Reduce batch sizes
2. Use generators for large datasets
3. Enable garbage collection

```python
import gc
gc.collect()
```

### High CPU Usage

**Problem**: API server pegs CPU at 100%.

**Solution**:
1. Profile the code
2. Check for infinite loops
3. Optimize feature extraction

```bash
# Use profiler
python -m cProfile -o profile.stats -m radarx.api.server
```

## Common Error Messages

### "Model not found"

**Cause**: Model files missing or incorrect path.

**Fix**: Train and save model first:
```bash
radarx-train --model-type probability --output-dir ./models
```

### "Invalid token address"

**Cause**: Token address format incorrect.

**Fix**: Ensure address is:
- 42 characters for EVM chains (0x + 40 hex chars)
- Base58 for Solana
- Lowercase for consistency

### "Feature extraction failed"

**Cause**: Missing or malformed input data.

**Fix**: Validate input data structure:
```python
from radarx.data.normalizer import validate_token_data
validate_token_data(token_data)
```

## Getting Help

If your issue isn't covered here:

1. **Check Logs**: Enable `LOG_LEVEL=DEBUG` for detailed logs
2. **GitHub Issues**: Search [existing issues](https://github.com/technedict/radarx/issues)
3. **Documentation**: Review [full documentation](README.md)
4. **Create Issue**: Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version)
   - Relevant logs

## Debug Mode

Enable comprehensive debugging:

```bash
# In .env
LOG_LEVEL=DEBUG
LOG_REQUESTS=true
LOG_RESPONSES=true
ENABLE_PROFILING=true
```

Then run with logging:

```bash
python -m radarx.api.server 2>&1 | tee debug.log
```
