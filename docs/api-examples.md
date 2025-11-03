# API Examples

This guide provides common API usage patterns for RadarX.

## Token Scoring

### Basic Token Score

Get a token score with probability heatmaps:

```bash
curl "http://localhost:8000/score/token?address=0x1234567890abcdef1234567890abcdef12345678&chain=ethereum&horizons=24h,7d,30d"
```

### Token Score with Features

Include raw feature vectors in the response:

```bash
curl "http://localhost:8000/score/token?address=0x1234...&chain=ethereum&horizons=24h&include_features=true"
```

### Token Score with Timeline

Include event timeline in the response:

```bash
curl "http://localhost:8000/score/token?address=0x1234...&chain=ethereum&horizons=24h,7d&include_timelines=true"
```

## Wallet Analytics

### Basic Wallet Report

Get comprehensive wallet analytics:

```bash
curl "http://localhost:8000/wallet/report?address=0xabcdef1234567890abcdef1234567890abcdef12&period=30d"
```

### Wallet Report with Custom Period

Specify a custom date range:

```bash
curl "http://localhost:8000/wallet/report?address=0xabcd...&from_date=2024-01-01&to_date=2024-01-31"
```

### Wallet Report with Trade Details

Include individual trades (up to max_trades):

```bash
curl "http://localhost:8000/wallet/report?address=0xabcd...&period=30d&include_trades=true&max_trades=50"
```

## Wallet Search

### Find Top Performers

Search for high-performing wallets:

```bash
curl "http://localhost:8000/search/wallets?min_win_rate=0.6&min_trades=10&sort_by=win_rate&limit=100"
```

### Filter by Chain

Search within a specific blockchain:

```bash
curl "http://localhost:8000/search/wallets?chain=ethereum&min_win_rate=0.5&sort_by=pnl&limit=50"
```

### Search with PnL Filter

Find profitable wallets:

```bash
curl "http://localhost:8000/search/wallets?min_pnl=10000&min_trades=20&sort_by=pnl&order=desc"
```

## Alert Subscriptions

### Subscribe to Token Alerts

Get notified when tokens meet specific criteria:

```bash
curl -X POST "http://localhost:8000/alerts/subscribe?webhook_url=https://example.com/webhook&min_probability_10x=0.5"
```

### Subscribe with Multiple Criteria

Set multiple alert thresholds:

```bash
curl -X POST "http://localhost:8000/alerts/subscribe" \
  -d "webhook_url=https://example.com/webhook" \
  -d "min_probability_2x=0.6" \
  -d "min_probability_10x=0.3" \
  -d "max_risk_score=50"
```

### Monitor Specific Wallets

Subscribe to alerts for specific wallet activity:

```bash
curl -X POST "http://localhost:8000/alerts/subscribe" \
  -d "webhook_url=https://example.com/webhook" \
  -d "wallet_addresses[]=0xabc..." \
  -d "wallet_addresses[]=0xdef..."
```

## Python Examples

### Using httpx (async)

```python
import httpx
import asyncio

async def get_token_score():
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://localhost:8000/score/token",
            params={
                "address": "0x1234567890abcdef1234567890abcdef12345678",
                "chain": "ethereum",
                "horizons": "24h,7d,30d"
            }
        )
        return response.json()

# Run
result = asyncio.run(get_token_score())
print(result)
```

### Using requests (sync)

```python
import requests

def get_wallet_report(address, period="30d"):
    response = requests.get(
        "http://localhost:8000/wallet/report",
        params={
            "address": address,
            "period": period,
            "include_trades": True
        }
    )
    return response.json()

# Use it
report = get_wallet_report("0xabcdef1234567890abcdef1234567890abcdef12")
print(f"Win rate: {report['win_rate']['overall']:.2%}")
```

## See Also

- [API Reference](api-reference.md) - Complete endpoint documentation
- [User Guide](user-guide.md) - Detailed feature explanations
- Sample responses in `examples/sample_responses.py`
