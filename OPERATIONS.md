# RadarX Operations Runbook

Quick reference guide for operating and troubleshooting RadarX in production.

## Table of Contents

- [Quick Start](#quick-start)
- [Health Checks](#health-checks)
- [Common Issues](#common-issues)
- [Performance Tuning](#performance-tuning)
- [Incident Response](#incident-response)
- [Maintenance Tasks](#maintenance-tasks)

## Quick Start

### Starting Services

```bash
# Start all services
docker-compose up -d

# Or manually:
# 1. Start PostgreSQL
sudo systemctl start postgresql

# 2. Start Redis
sudo systemctl start redis

# 3. Start Kafka (if using)
cd /opt/kafka && bin/kafka-server-start.sh config/server.properties &

# 4. Start API servers
cd /opt/radarx && uvicorn radarx.api.server:app --host 0.0.0.0 --port 8000 --workers 4

# 5. Start model server
cd /opt/radarx && uvicorn serving.model_server:app --host 0.0.0.0 --port 8001
```

### Stopping Services

```bash
# Stop all
docker-compose down

# Or manually:
sudo systemctl stop postgresql redis
pkill -f kafka
pkill -f uvicorn
```

## Health Checks

### API Health

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","version":"1.0.0"}
```

### Database Connection

```bash
# Test PostgreSQL connection
psql -U radarx_user -d radarx -c "SELECT 1;"

# Check active connections
psql -U radarx_user -d radarx -c "SELECT count(*) FROM pg_stat_activity;"
```

### Redis Connection

```bash
# Test Redis
redis-cli ping
# Expected: PONG

# Check memory usage
redis-cli info memory | grep used_memory_human

# Check hit rate
redis-cli info stats | grep keyspace
```

### Model Server

```bash
# Check model server
curl http://localhost:8001/health

# Test prediction
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"token": {...}, "temporal": [...]}'
```

## Common Issues

### Issue: High API Latency

**Symptoms**: Requests taking >2 seconds

**Diagnosis**:
```bash
# Check API metrics
curl http://localhost:8000/metrics | grep request_duration

# Check database slow queries
psql -U radarx_user -d radarx -c "
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;"

# Check Redis latency
redis-cli --latency-history
```

**Solutions**:
1. Check if cache is warming up (cache misses high)
2. Add database indexes for slow queries
3. Increase connection pool size
4. Scale horizontally (add more API servers)

### Issue: Out of Memory

**Symptoms**: 502/503 errors, services crashing

**Diagnosis**:
```bash
# Check memory usage
free -h
docker stats

# Check which process is using memory
ps aux --sort=-%mem | head -10
```

**Solutions**:
1. Restart services to clear memory leaks
2. Reduce Redis maxmemory
3. Reduce model batch sizes
4. Scale vertically (add more RAM) or horizontally

### Issue: Model Prediction Failures

**Symptoms**: `/score/token` returning errors

**Diagnosis**:
```bash
# Check model server logs
docker logs radarx-model-server

# Check model files exist
ls -lh /models/

# Test model loading
python -c "from radarx.models import ProbabilityPredictor; p = ProbabilityPredictor(); p.load('/models/predictor.pkl')"
```

**Solutions**:
1. Verify model files are not corrupted
2. Check model version compatibility
3. Restart model server
4. Rollback to previous model version

### Issue: Database Connection Pool Exhausted

**Symptoms**: "connection pool exhausted" errors

**Diagnosis**:
```bash
# Check active connections
psql -U radarx_user -d radarx -c "
SELECT count(*), state
FROM pg_stat_activity
GROUP BY state;"
```

**Solutions**:
```python
# Increase pool size in config
engine = create_engine(
    DATABASE_URL,
    pool_size=40,  # Increase from 20
    max_overflow=80  # Increase from 40
)
```

### Issue: Cache Stampede

**Symptoms**: Sudden spike in database load, slow responses

**Diagnosis**:
```bash
# Check cache miss rate
redis-cli info stats | grep keyspace_misses

# Check database load
psql -U radarx_user -d radarx -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"
```

**Solutions**:
1. Implement cache warming on startup
2. Add jitter to cache TTLs
3. Use cache locking (single request computes, others wait)

### Issue: Kafka Consumer Lag

**Symptoms**: Data processing delays

**Diagnosis**:
```bash
# Check consumer lag
bin/kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group radarx-processor
```

**Solutions**:
1. Scale consumer group (add more consumers)
2. Increase batch size for processing
3. Optimize message processing logic
4. Check if downstream services are slow

## Performance Tuning

### Database Optimization

```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_token_scores_lookup 
ON token_scores(token_id, timestamp DESC);

CREATE INDEX CONCURRENTLY idx_wallet_perf 
ON wallet_analytics(wallet_address, chain, timestamp DESC);

-- Update statistics
ANALYZE token_scores;
ANALYZE wallet_analytics;

-- Vacuum
VACUUM ANALYZE;
```

### Redis Optimization

```bash
# Eviction policy for cache
CONFIG SET maxmemory-policy allkeys-lru

# Increase max connections
CONFIG SET maxclients 10000

# Enable persistence (if needed)
CONFIG SET save "900 1 300 10 60 10000"
```

### API Server Tuning

```python
# Increase worker count
uvicorn radarx.api.server:app --workers 8

# Enable uvloop for performance
pip install uvloop
uvicorn radarx.api.server:app --loop uvloop

# Adjust timeouts
app = FastAPI(timeout=30)
```

### Model Serving Optimization

```python
# Batch predictions
predictor.predict_proba_batch(features_list, batch_size=32)

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Model quantization for faster inference
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## Incident Response

### High CPU Usage

1. Identify process:
```bash
top -c
```

2. If API server:
   - Check for infinite loops in request handlers
   - Look for inefficient algorithms
   - Profile with `py-spy`

3. If database:
   - Check for missing indexes
   - Look for long-running queries
   - Consider read replicas

### Memory Leak

1. Monitor memory over time:
```bash
watch -n 5 'free -h && ps aux --sort=-%mem | head -5'
```

2. Profile Python memory:
```bash
pip install memory_profiler
python -m memory_profiler radarx/api/server.py
```

3. Fix common causes:
   - Clear large data structures after use
   - Avoid circular references
   - Use weak references where appropriate

### Data Inconsistency

1. Check data integrity:
```sql
-- Verify foreign keys
SELECT * FROM token_scores WHERE token_id NOT IN (SELECT id FROM tokens);

-- Check for nulls in required fields
SELECT * FROM wallet_analytics WHERE wallet_address IS NULL;
```

2. Reconcile data:
   - Run data validation scripts
   - Compare with source of truth
   - Backfill missing data

3. Prevent recurrence:
   - Add database constraints
   - Improve validation logic
   - Add integration tests

## Maintenance Tasks

### Daily

- [ ] Check health endpoints
- [ ] Monitor error rates
- [ ] Review alert notifications
- [ ] Check disk space usage

### Weekly

- [ ] Review slow query logs
- [ ] Analyze cache hit rates
- [ ] Check model performance metrics
- [ ] Review security logs

### Monthly

- [ ] Update dependencies
- [ ] Rotate logs
- [ ] Review and update alerts
- [ ] Capacity planning review

### Quarterly

- [ ] Database vacuum and analyze
- [ ] Security audit
- [ ] Disaster recovery drill
- [ ] Performance benchmarking

## Monitoring Commands

### System Metrics

```bash
# CPU usage
mpstat 1 5

# Memory usage
vmstat 1 5

# Disk I/O
iostat -x 1 5

# Network
iftop
```

### Application Metrics

```bash
# API request rate
curl http://localhost:8000/metrics | grep radarx_requests_total

# Model prediction latency
curl http://localhost:8000/metrics | grep radarx_model_prediction_seconds

# Cache hit rate
redis-cli info stats | grep keyspace_hits
redis-cli info stats | grep keyspace_misses
```

### Database Metrics

```sql
-- Top queries by execution time
SELECT query, calls, total_time, mean_time, max_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 20;

-- Table sizes
SELECT schemaname, tablename,
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;
```

## Useful Commands

### Log Analysis

```bash
# Find errors in logs
journalctl -u radarx-api -p err --since today

# Count error types
grep ERROR /var/log/radarx/api.log | cut -d' ' -f5- | sort | uniq -c | sort -rn

# Monitor logs in real-time
tail -f /var/log/radarx/api.log | grep -E '(ERROR|WARNING)'
```

### Quick Fixes

```bash
# Restart API server
systemctl restart radarx-api

# Clear Redis cache
redis-cli FLUSHDB

# Kill hung processes
pkill -9 -f "radarx.api.server"

# Reload nginx
nginx -s reload

# Force PostgreSQL connection close
psql -U postgres -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='radarx' AND pid <> pg_backend_pid();"
```

## Emergency Contacts

- **On-Call Engineer**: [Phone/Slack]
- **Database Admin**: [Contact]
- **DevOps Lead**: [Contact]
- **Security Team**: [Contact]

## Escalation Path

1. **Level 1**: On-call engineer
2. **Level 2**: Engineering lead
3. **Level 3**: CTO/VP Engineering

## Rollback Procedures

### API Deployment Rollback

```bash
# Using Docker
docker-compose down
docker-compose -f docker-compose.v1.0.0.yml up -d

# Using Kubernetes
kubectl rollout undo deployment/radarx-api
```

### Model Rollback

```python
from radarx.backtesting import LearningLedger

ledger = LearningLedger()
previous_version = ledger.get_version('v2.0.0')  # Previous stable version

# Load previous model
predictor.load(previous_version.model_path)
```

### Database Migration Rollback

```bash
# Using Alembic
alembic downgrade -1

# Manual rollback
psql -U radarx_user radarx < backup_20240101.sql
```

---

For deployment procedures, see [DEPLOYMENT.md](DEPLOYMENT.md).
