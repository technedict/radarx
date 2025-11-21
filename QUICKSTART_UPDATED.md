# 🚀 RadarX Quick Start Guide

## Production-Ready Deployment in 5 Minutes

### Prerequisites
- Docker and Docker Compose installed
- Python 3.9+ (for local development)
- kubectl (for Kubernetes deployment)

### Option 1: Automated Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/technedict/radarx.git
cd radarx

# Run automated bootstrap
chmod +x bootstrap.sh
./bootstrap.sh

# Start all services with Docker Compose
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

**Services Started:**
- ✅ API Server (port 8000)
- ✅ PostgreSQL Database (port 5432)
- ✅ Redis Cache (port 6379)
- ✅ Prometheus Monitoring (port 9090)
- ✅ Grafana Dashboards (port 3000)

**Access Points:**
- API Documentation: http://localhost:8000/docs
- Grafana Dashboards: http://localhost:3000 (admin/admin)
- Prometheus Metrics: http://localhost:9090

### Option 2: Kubernetes Deployment

```bash
# Apply all Kubernetes manifests
kubectl apply -f infra/kubernetes/

# Check deployment status
kubectl get pods -n radarx

# Access API (after port-forward or Ingress setup)
kubectl port-forward -n radarx svc/radarx-api 8000:80
```

### Option 3: Helm Deployment

```bash
# Install with Helm
helm install radarx ./helm/radarx --namespace radarx --create-namespace

# Check status
helm status radarx -n radarx

# Upgrade
helm upgrade radarx ./helm/radarx
```

## Quick API Examples

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Score a Token
```bash
curl "http://localhost:8000/score/token?address=0x1234...&chain=ethereum&horizons=24h,7d,30d"
```

### 3. Find Smart Wallets
```bash
curl -X POST http://localhost:8000/smart-wallets/find \
  -H "Content-Type: application/json" \
  -d '{
    "token_address": "0x1234...",
    "chain": "ethereum",
    "window_days": 30,
    "top_k": 100
  }'
```

### 4. Get Wallet Report
```bash
curl "http://localhost:8000/wallet/report?address=0xabcd...&period=30d"
```

## Running the Backtesting Demo

```bash
# Activate virtual environment (if using local setup)
source venv/bin/activate

# Run backtesting demonstration
python notebooks/backtesting_demo.py
```

**Expected Output:**
```
============================================================
RadarX Backtesting Demo
============================================================

Generating synthetic dataset...
  Samples: 2000
  Features: 30
  Positive rate: 50.2%

Running walk-forward backtest (5 folds)...

============================================================
BACKTEST RESULTS
============================================================

Samples tested: 800
Number of folds: 5

Performance Metrics:
  AUC-ROC:           0.7523
  Brier Score:       0.1487
  Calibration Error: 0.0289
  Log Loss:          0.4821

✅ Backtest complete!
Results saved to: models/backtest_demo/
```

## Development Workflow

### 1. Install Development Dependencies
```bash
pip install -e .
pip install -r requirements.txt
```

### 2. Run Tests
```bash
# All tests
pytest tests/

# With coverage
pytest tests/ --cov=radarx --cov-report=html

# Specific test file
pytest tests/unit/test_api.py -v
```

### 3. Code Quality
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
flake8 src/ tests/

# Type check
mypy src/
```

### 4. Run API Server Locally
```bash
python -m radarx.api.server

# Or use uvicorn directly
uvicorn radarx.api.server:app --reload
```

## Monitoring & Observability

### Prometheus Metrics
Access metrics at: http://localhost:9090

**Key Metrics:**
- `radarx_requests_total` - Total API requests
- `radarx_request_duration_seconds` - Request latency
- `radarx_predictions_total` - Model predictions
- `radarx_errors_total` - Error count
- `radarx_feature_drift_score` - Feature drift

### Grafana Dashboards
Access dashboards at: http://localhost:3000

**Default Credentials:** admin / admin

**Dashboards:**
- Model Performance
- API Health
- Cache Performance

### Alerts
Check active alerts at: http://localhost:9090/alerts

**Alert Rules:**
- API Down
- High Error Rate
- Slow Responses
- Feature Drift
- Low Cache Hit Rate

## Troubleshooting

### API Not Starting
```bash
# Check logs
docker-compose logs api

# Restart service
docker-compose restart api
```

### Database Connection Issues
```bash
# Check PostgreSQL
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Recreate database
docker-compose down
docker volume rm radarx_postgres-data
docker-compose up -d
```

### Port Already in Use
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in docker-compose.yml
```

## Configuration

### Environment Variables
Create `.env` file or edit `docker-compose.yml`:

```bash
# API Settings
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=postgresql://radarx:password@postgres:5432/radarx
REDIS_URL=redis://redis:6379/0

# Model Settings
MODEL_VERSION=v1.0.0
FEATURE_VERSION=v1.0.0
ENABLE_ONLINE_LEARNING=false

# API Keys (optional)
DEXSCREENER_API_KEY=
ETHERSCAN_API_KEY=
TWITTER_BEARER_TOKEN=

# Logging
LOG_LEVEL=INFO
```

## Performance Optimization

### 1. Cache Configuration
```python
# Adjust cache TTL in config
CACHE_TTL_SECONDS = 300  # 5 minutes

# Clear cache
redis-cli FLUSHALL
```

### 2. Model Serving
```python
# Batch predictions for better throughput
batch_size = 32

# Enable GPU if available
use_gpu = True
```

### 3. Database Tuning
```sql
-- Add indexes for frequently queried fields
CREATE INDEX idx_predictions_timestamp ON token_predictions(timestamp DESC);

-- Analyze query plans
EXPLAIN ANALYZE SELECT * FROM token_predictions WHERE ...;
```

## Production Checklist

Before deploying to production:

- [ ] Update all API keys in `.env` or Kubernetes Secrets
- [ ] Change default passwords (PostgreSQL, Grafana, etc.)
- [ ] Configure proper CORS origins
- [ ] Set up TLS/HTTPS certificates
- [ ] Configure log aggregation (e.g., ELK stack)
- [ ] Set up alerting notifications (PagerDuty, Slack, etc.)
- [ ] Run load tests
- [ ] Enable backup for database
- [ ] Configure monitoring retention policies
- [ ] Review and adjust resource limits

## Getting Help

### Documentation
- **Technical Report:** `TECHNICAL_REPORT.md`
- **Implementation Summary:** `IMPLEMENTATION_COMPLETE.md`
- **Acceptance Checklist:** `ACCEPTANCE_CHECKLIST.md`
- **Full README:** `README.md`

### Support Channels
- GitHub Issues: [github.com/technedict/radarx/issues](https://github.com/technedict/radarx/issues)
- Documentation: `/docs` directory

## What's Included

✅ **Production-Ready API** - FastAPI with OpenAPI docs  
✅ **Advanced ML Models** - Ensemble with calibration  
✅ **Walk-Forward Validation** - Proper backtesting  
✅ **Docker & Kubernetes** - Container orchestration  
✅ **Monitoring Stack** - Prometheus + Grafana  
✅ **CI/CD Pipeline** - GitHub Actions with model gating  
✅ **Database Schema** - PostgreSQL with migrations  
✅ **Feature Engineering** - Trade matching, bot detection  
✅ **Documentation** - Comprehensive guides

## Next Steps

1. **Explore the API:** Visit http://localhost:8000/docs
2. **Run Backtest:** Execute `python notebooks/backtesting_demo.py`
3. **Check Monitoring:** Open http://localhost:3000
4. **Read Documentation:** Review `TECHNICAL_REPORT.md`
5. **Deploy to Staging:** Use Kubernetes manifests

---

**Status:** ✅ Production-Ready  
**Version:** 1.0.0  
**Last Updated:** 2025-11-20
