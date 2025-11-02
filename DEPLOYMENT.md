# RadarX Production Deployment Guide

This guide covers deploying RadarX to production with proper infrastructure, monitoring, and operational practices.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Infrastructure Requirements](#infrastructure-requirements)
- [Database Setup](#database-setup)
- [Caching Layer](#caching-layer)
- [Streaming Pipeline](#streaming-pipeline)
- [Model Serving](#model-serving)
- [Monitoring & Observability](#monitoring--observability)
- [Alerting](#alerting)
- [Data Retention](#data-retention)
- [Security](#security)
- [Scaling](#scaling)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer (nginx)                     │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
         ┌──────────▼──────────┐   ┌─────────▼─────────┐
         │   FastAPI App       │   │   FastAPI App     │
         │   (API Server)      │   │   (API Server)    │
         └──────────┬──────────┘   └─────────┬─────────┘
                    │                        │
         ┌──────────┴────────────────────────┴──────────┐
         │                                               │
    ┌────▼─────┐  ┌──────────┐  ┌──────────┐  ┌───────▼──────┐
    │PostgreSQL│  │  Redis   │  │  Kafka   │  │ Model Server │
    │   (DB)   │  │ (Cache)  │  │(Streaming)│  │  (ML Serve)  │
    └──────────┘  └──────────┘  └──────────┘  └──────────────┘
         │              │              │              │
         └──────────────┴──────────────┴──────────────┘
                        │
              ┌─────────▼─────────┐
              │   Prometheus      │
              │   (Monitoring)    │
              └───────────────────┘
```

## Infrastructure Requirements

### Minimum Production Setup

**Application Servers** (2+ instances for HA):
- CPU: 4 cores
- RAM: 8 GB
- Storage: 50 GB SSD
- OS: Ubuntu 22.04 LTS

**Database Server**:
- CPU: 4 cores
- RAM: 16 GB
- Storage: 500 GB SSD (RAID 10)
- PostgreSQL 15+

**Cache Server**:
- CPU: 2 cores
- RAM: 8 GB
- Storage: 20 GB SSD
- Redis 7+

**Streaming Server**:
- CPU: 4 cores
- RAM: 16 GB
- Storage: 200 GB SSD
- Kafka 3.0+ or Redis Streams

**Model Serving**:
- CPU: 8 cores (or GPU)
- RAM: 16 GB
- Storage: 100 GB SSD

## Database Setup

### PostgreSQL Configuration

```bash
# Install PostgreSQL
sudo apt update
sudo apt install postgresql-15 postgresql-contrib

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE radarx;
CREATE USER radarx_user WITH ENCRYPTED PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE radarx TO radarx_user;
ALTER DATABASE radarx OWNER TO radarx_user;
EOF
```

### Database Schema

```sql
-- Token data table
CREATE TABLE tokens (
    id SERIAL PRIMARY KEY,
    address VARCHAR(66) NOT NULL,
    chain VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    first_seen TIMESTAMP DEFAULT NOW(),
    UNIQUE(address, chain)
);

CREATE INDEX idx_tokens_address ON tokens(address);
CREATE INDEX idx_tokens_chain ON tokens(chain);

-- Token scores table
CREATE TABLE token_scores (
    id SERIAL PRIMARY KEY,
    token_id INTEGER REFERENCES tokens(id),
    timestamp TIMESTAMP DEFAULT NOW(),
    probability_data JSONB,
    risk_score JSONB,
    features JSONB,
    model_version VARCHAR(50)
);

CREATE INDEX idx_scores_token ON token_scores(token_id);
CREATE INDEX idx_scores_timestamp ON token_scores(timestamp DESC);
CREATE INDEX idx_scores_model ON token_scores(model_version);

-- Wallet analytics table
CREATE TABLE wallet_analytics (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(66) NOT NULL,
    chain VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    win_rate_data JSONB,
    pnl_data JSONB,
    behavioral_patterns JSONB,
    rankings JSONB,
    UNIQUE(wallet_address, chain, timestamp)
);

CREATE INDEX idx_wallet_address ON wallet_analytics(wallet_address);
CREATE INDEX idx_wallet_chain ON wallet_analytics(chain);

-- Alert subscriptions table
CREATE TABLE alert_subscriptions (
    id SERIAL PRIMARY KEY,
    webhook_url TEXT NOT NULL,
    criteria JSONB NOT NULL,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    last_triggered TIMESTAMP
);

-- Feature store table
CREATE TABLE feature_store (
    id SERIAL PRIMARY KEY,
    entity_id VARCHAR(100) NOT NULL,
    entity_type VARCHAR(20) NOT NULL,  -- 'token' or 'wallet'
    timestamp TIMESTAMP NOT NULL,
    features JSONB NOT NULL,
    metadata JSONB
);

CREATE INDEX idx_features_entity ON feature_store(entity_id, timestamp DESC);
CREATE INDEX idx_features_type ON feature_store(entity_type);

-- Model versions table
CREATE TABLE model_versions (
    id SERIAL PRIMARY KEY,
    version_id VARCHAR(100) UNIQUE NOT NULL,
    version_name VARCHAR(50) NOT NULL,
    model_config JSONB,
    training_data_info JSONB,
    backtest_results JSONB,
    strategy_results JSONB,
    created_at TIMESTAMP DEFAULT NOW(),
    model_path TEXT,
    notes TEXT,
    tags TEXT[]
);

CREATE INDEX idx_model_versions_name ON model_versions(version_name);
CREATE INDEX idx_model_versions_created ON model_versions(created_at DESC);
```

### Connection Pool Configuration

```python
# config/database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

DATABASE_URL = "postgresql://radarx_user:password@localhost:5432/radarx"

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

## Caching Layer

### Redis Configuration

```bash
# Install Redis
sudo apt install redis-server

# Configure Redis
sudo nano /etc/redis/redis.conf
```

Key Redis settings:
```
maxmemory 6gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### Redis Usage Patterns

```python
# config/redis.py
import redis
from typing import Optional, Any
import json

class RedisCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL in seconds."""
        self.client.setex(
            key,
            ttl,
            json.dumps(value)
        )
    
    def delete(self, key: str):
        """Delete key from cache."""
        self.client.delete(key)
    
    def exists(self, key: str) -> bool:
        """Check if key exists."""
        return self.client.exists(key) > 0
```

### Cache Keys Convention

```
# Token scores
token:score:{chain}:{address}:{horizons}

# Wallet reports
wallet:report:{chain}:{address}:{period}

# API rate limits
ratelimit:{ip_address}:{endpoint}

# Feature data
features:{entity_type}:{entity_id}:{timestamp}

# Model predictions (short TTL)
prediction:{model_version}:{token_address}
```

## Streaming Pipeline

### Kafka Setup

```bash
# Install Kafka
wget https://downloads.apache.org/kafka/3.6.0/kafka_2.13-3.6.0.tgz
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0

# Start Zookeeper
bin/zookeeper-server-start.sh config/zookeeper.properties &

# Start Kafka
bin/kafka-server-start.sh config/server.properties &

# Create topics
bin/kafka-topics.sh --create --topic token-data --bootstrap-server localhost:9092 --partitions 10 --replication-factor 1
bin/kafka-topics.sh --create --topic wallet-events --bootstrap-server localhost:9092 --partitions 10 --replication-factor 1
bin/kafka-topics.sh --create --topic model-predictions --bootstrap-server localhost:9092 --partitions 5 --replication-factor 1
```

### Stream Processing

```python
# streams/processor.py
from kafka import KafkaConsumer, KafkaProducer
import json

class StreamProcessor:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'token-data',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id='radarx-processor',
            auto_offset_reset='earliest'
        )
        
        self.producer = KafkaProducer(
            bootstrap_servers=['localhost:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
    
    def process_token_stream(self):
        """Process token data stream."""
        for message in self.consumer:
            token_data = message.value
            
            # Extract features
            features = self.extract_features(token_data)
            
            # Score token
            score = self.score_token(features)
            
            # Publish prediction
            self.producer.send('model-predictions', {
                'token_address': token_data['address'],
                'score': score,
                'timestamp': token_data['timestamp']
            })
```

## Model Serving

### Model Server Setup

```python
# serving/model_server.py
from fastapi import FastAPI
from typing import Dict, List
import torch
from radarx.models import ProbabilityPredictor, RiskScorer

app = FastAPI()

# Load models at startup
predictor = ProbabilityPredictor()
predictor.load('/models/predictor_v2.1.0.pkl')

scorer = RiskScorer()
scorer.load('/models/scorer_v2.1.0.pkl')

@app.post("/predict")
async def predict(features: Dict) -> Dict:
    """Make predictions on features."""
    probabilities = predictor.predict_proba(
        X_features=[features['token']],
        X_temporal=[features['temporal']]
    )
    
    risk_scores = scorer.score([features['token']])
    
    return {
        'probabilities': probabilities,
        'risk_scores': risk_scores
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": True}
```

### Model Deployment with Docker

```dockerfile
# Dockerfile.model-server
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model files
COPY models/ /models/
COPY serving/ /app/serving/
COPY src/ /app/src/

# Set Python path
ENV PYTHONPATH=/app/src

# Expose port
EXPOSE 8001

# Run server
CMD ["uvicorn", "serving.model_server:app", "--host", "0.0.0.0", "--port", "8001"]
```

## Monitoring & Observability

### Prometheus Metrics

```python
# monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
request_count = Counter(
    'radarx_requests_total',
    'Total request count',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'radarx_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

# Model metrics
model_prediction_time = Histogram(
    'radarx_model_prediction_seconds',
    'Model prediction time',
    ['model_type']
)

model_predictions = Counter(
    'radarx_model_predictions_total',
    'Total model predictions',
    ['model_version']
)

# Feature metrics
feature_extraction_time = Histogram(
    'radarx_feature_extraction_seconds',
    'Feature extraction time',
    ['feature_type']
)

# Cache metrics
cache_hits = Counter(
    'radarx_cache_hits_total',
    'Cache hit count',
    ['cache_type']
)

cache_misses = Counter(
    'radarx_cache_misses_total',
    'Cache miss count',
    ['cache_type']
)

# Active connections
active_connections = Gauge(
    'radarx_active_connections',
    'Number of active connections'
)

# Database metrics
db_query_duration = Histogram(
    'radarx_db_query_duration_seconds',
    'Database query duration',
    ['query_type']
)
```

### Structured Logging

```python
# monitoring/logging.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# Configure logger
logger = logging.getLogger('radarx')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
```

### Sentry Integration

```python
# monitoring/sentry.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

def init_sentry():
    sentry_sdk.init(
        dsn="https://your-dsn@sentry.io/project-id",
        integrations=[FastApiIntegration()],
        traces_sample_rate=0.1,
        environment="production",
        release="radarx@1.0.0"
    )
```

## Alerting

### Webhook Delivery System

```python
# alerting/webhook.py
import httpx
import asyncio
from typing import Dict, List

class WebhookDelivery:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=10.0)
    
    async def send_alert(
        self,
        webhook_url: str,
        alert_data: Dict,
        retry_count: int = 3
    ):
        """Send alert to webhook with retries."""
        for attempt in range(retry_count):
            try:
                response = await self.client.post(
                    webhook_url,
                    json=alert_data,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    return True
                
            except Exception as e:
                if attempt == retry_count - 1:
                    # Log failure
                    logger.error(f"Webhook delivery failed: {e}")
                    return False
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)
        
        return False
```

### Alert Criteria Monitoring

```python
# alerting/monitor.py
from typing import Dict, List
import asyncio

class AlertMonitor:
    def __init__(self, webhook_delivery: WebhookDelivery):
        self.webhook_delivery = webhook_delivery
    
    async def check_token_alerts(
        self,
        token_score: Dict,
        subscriptions: List[Dict]
    ):
        """Check if token score triggers any alerts."""
        for sub in subscriptions:
            criteria = sub['criteria']
            
            # Check probability threshold
            if 'probability_threshold' in criteria:
                horizon = criteria.get('horizon', '7d')
                multiplier = criteria.get('multiplier', '10x')
                threshold = criteria['probability_threshold']
                
                prob = token_score['probability_heatmap']['horizons'][horizon][multiplier]['probability']
                
                if prob >= threshold:
                    await self.webhook_delivery.send_alert(
                        sub['webhook_url'],
                        {
                            'type': 'probability_threshold',
                            'token_address': token_score['token_address'],
                            'probability': prob,
                            'threshold': threshold,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    )
            
            # Check risk threshold
            if 'risk_threshold' in criteria:
                risk = token_score['risk_score']['composite_score']
                if risk <= criteria['risk_threshold']:
                    await self.webhook_delivery.send_alert(
                        sub['webhook_url'],
                        {
                            'type': 'risk_threshold',
                            'token_address': token_score['token_address'],
                            'risk_score': risk,
                            'timestamp': datetime.utcnow().isoformat()
                        }
                    )
```

## Data Retention

### Retention Policies

```python
# retention/policy.py
from datetime import datetime, timedelta
from sqlalchemy import delete

class DataRetentionPolicy:
    def __init__(self, db_engine):
        self.engine = db_engine
    
    async def cleanup_old_data(self):
        """Run retention policy cleanup."""
        with self.engine.connect() as conn:
            # Delete token scores older than 90 days
            conn.execute(
                delete(token_scores).where(
                    token_scores.c.timestamp < datetime.now() - timedelta(days=90)
                )
            )
            
            # Delete wallet analytics older than 180 days
            conn.execute(
                delete(wallet_analytics).where(
                    wallet_analytics.c.timestamp < datetime.now() - timedelta(days=180)
                )
            )
            
            # Keep model versions indefinitely, only delete unused artifacts
            # Archive feature store data older than 30 days to cold storage
            
            conn.commit()
```

### Scheduled Cleanup Job

```python
# retention/scheduler.py
import schedule
import time

def run_retention_cleanup():
    policy = DataRetentionPolicy(db_engine)
    asyncio.run(policy.cleanup_old_data())

# Run daily at 2 AM
schedule.every().day.at("02:00").do(run_retention_cleanup)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Security

### API Key Authentication

```python
# security/auth.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key
```

### Rate Limiting

```python
# security/rate_limit.py
from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.get("/score/token")
@limiter.limit("60/minute")
async def score_token(request: Request, ...):
    ...
```

## Scaling

### Horizontal Scaling

Deploy multiple API server instances behind a load balancer:

```nginx
# nginx.conf
upstream radarx_api {
    least_conn;
    server api1.example.com:8000;
    server api2.example.com:8000;
    server api3.example.com:8000;
}

server {
    listen 80;
    server_name api.radarx.example.com;
    
    location / {
        proxy_pass http://radarx_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Auto-Scaling with Kubernetes

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: radarx-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: radarx-api
  template:
    metadata:
      labels:
        app: radarx-api
    spec:
      containers:
      - name: api
        image: radarx/api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: radarx-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: radarx-api
  minReplicas: 3
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Backup & Disaster Recovery

### Database Backups

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/postgres"

# Create backup
pg_dump -U radarx_user radarx | gzip > "$BACKUP_DIR/radarx_$DATE.sql.gz"

# Keep only last 30 days
find $BACKUP_DIR -name "radarx_*.sql.gz" -mtime +30 -delete

# Upload to S3
aws s3 cp "$BACKUP_DIR/radarx_$DATE.sql.gz" s3://radarx-backups/postgres/
```

### Model Backups

```python
# backup/models.py
import shutil
from pathlib import Path

def backup_model(model_path: str, backup_location: str):
    """Backup model to remote storage."""
    # Copy model file
    shutil.copy(model_path, backup_location)
    
    # Upload to S3/GCS
    # ...
```

---

For more detailed configuration and troubleshooting, see the [Operations Runbook](OPERATIONS.md).
