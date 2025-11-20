"""FastAPI application and REST API endpoints."""

from datetime import datetime, timezone
from typing import List, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from radarx.api.rate_limiter import RateLimiter
from radarx.api.services import TokenScoringService, WalletAnalyticsService
from radarx.config import settings
from radarx.schemas.token import TokenScore
from radarx.schemas.wallet import WalletReport
from radarx.smart_wallet_finder.finder import SmartWalletFinder
from radarx.smart_wallet_finder.schemas import (
    BulkScanRequest,
    BulkScanResponse,
    SmartWalletFindRequest,
    SmartWalletFindResponse,
    WalletProfileRequest,
    WalletProfileResponse,
)

# Optional Prometheus metrics
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    PROMETHEUS_AVAILABLE = True

    # Define metrics
    REQUEST_COUNT = Counter("radarx_requests_total", "Total requests", ["endpoint", "method"])
    REQUEST_DURATION = Histogram(
        "radarx_request_duration_seconds", "Request duration", ["endpoint"]
    )
    PREDICTION_COUNT = Counter("radarx_predictions_total", "Total predictions", ["chain"])
    ERROR_COUNT = Counter("radarx_errors_total", "Total errors", ["endpoint", "error_type"])
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
token_service = TokenScoringService()
wallet_service = WalletAnalyticsService()
rate_limiter = RateLimiter()
smart_wallet_finder = SmartWalletFinder()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "description": settings.api_description,
        "endpoints": {
            "token_score": "/score/token",
            "wallet_report": "/wallet/report",
            "wallet_search": "/search/wallets",
            "alerts": "/alerts/subscribe",
            "health": "/health",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.api_version,
    }


@app.get("/score/token", response_model=TokenScore)
async def score_token(
    address: str = Query(..., description="Token contract address"),
    chain: str = Query(..., description="Blockchain network"),
    horizons: str = Query("24h,7d,30d", description="Comma-separated time horizons"),
    include_features: bool = Query(False, description="Include raw feature vectors"),
    include_timelines: bool = Query(False, description="Include event timelines"),
) -> TokenScore:
    """
    Score a token with probability heatmaps, risk assessment, and explanations.

    This endpoint provides comprehensive token analysis including:
    - Probability heatmaps for reaching 2x, 5x, 10x, 20x, 50x multipliers
    - Composite risk score with component breakdown
    - Feature contribution explanations
    - Optional raw features and event timelines

    Args:
        address: Token contract address
        chain: Blockchain network (ethereum, bsc, solana, etc.)
        horizons: Time horizons for probability predictions
        include_features: Whether to include raw feature vectors
        include_timelines: Whether to include event timelines

    Returns:
        TokenScore: Complete token scoring response
    """
    try:
        horizon_list = [h.strip() for h in horizons.split(",")]

        result = await token_service.score_token(
            address=address,
            chain=chain,
            horizons=horizon_list,
            include_features=include_features,
            include_timelines=include_timelines,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/wallet/report", response_model=WalletReport)
async def get_wallet_report(
    address: str = Query(..., description="Wallet address"),
    chain: str = Query("multi-chain", description="Blockchain network"),
    from_date: Optional[str] = Query(None, description="Start date (ISO 8601)"),
    to_date: Optional[str] = Query(None, description="End date (ISO 8601)"),
    period: str = Query("all-time", description="Analysis period (1d, 7d, 30d, all-time)"),
    include_trades: bool = Query(True, description="Include individual trades"),
    max_trades: int = Query(100, description="Maximum trades to return"),
) -> WalletReport:
    """
    Get comprehensive wallet analytics report.

    This endpoint provides detailed wallet performance analysis including:
    - Win rate statistics across timeframes
    - Realized and unrealized PnL
    - Token and chain-level breakdowns
    - Behavioral pattern detection
    - Related wallets analysis
    - Global and chain-specific rankings

    Args:
        address: Wallet address
        chain: Blockchain network or "multi-chain"
        from_date: Optional start date filter
        to_date: Optional end date filter
        period: Analysis period preset
        include_trades: Whether to include trade list
        max_trades: Maximum number of trades to return

    Returns:
        WalletReport: Complete wallet analytics report
    """
    try:
        result = await wallet_service.get_wallet_report(
            address=address,
            chain=chain,
            from_date=from_date,
            to_date=to_date,
            period=period,
            include_trades=include_trades,
            max_trades=max_trades,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/search/wallets")
async def search_wallets(
    min_win_rate: Optional[float] = Query(None, ge=0, le=1, description="Minimum win rate"),
    min_trades: Optional[int] = Query(None, ge=0, description="Minimum trade count"),
    min_pnl: Optional[float] = Query(None, description="Minimum realized PnL (USD)"),
    chain: Optional[str] = Query(None, description="Filter by chain"),
    sort_by: str = Query("win_rate", description="Sort field"),
    order: str = Query("desc", description="Sort order (asc/desc)"),
    limit: int = Query(100, le=1000, description="Results limit"),
    offset: int = Query(0, description="Results offset"),
):
    """
    Search and discover top performing wallets.

    This endpoint enables wallet discovery based on performance metrics:
    - Filter by win rate, trade count, PnL
    - Sort by various performance metrics
    - Pagination support

    Args:
        min_win_rate: Minimum win rate filter
        min_trades: Minimum trade count filter
        min_pnl: Minimum PnL filter (USD)
        chain: Blockchain network filter
        sort_by: Sort field (win_rate, pnl, trades, etc.)
        order: Sort order (asc/desc)
        limit: Maximum results to return
        offset: Pagination offset

    Returns:
        List of wallet summaries matching criteria
    """
    try:
        results = await wallet_service.search_wallets(
            min_win_rate=min_win_rate,
            min_trades=min_trades,
            min_pnl=min_pnl,
            chain=chain,
            sort_by=sort_by,
            order=order,
            limit=limit,
            offset=offset,
        )

        return {
            "total": results["total"],
            "limit": limit,
            "offset": offset,
            "wallets": results["wallets"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/alerts/subscribe")
async def subscribe_to_alerts(
    webhook_url: str = Query(..., description="Webhook URL for alerts"),
    wallet_addresses: Optional[List[str]] = Query(None, description="Wallet addresses to monitor"),
    token_addresses: Optional[List[str]] = Query(None, description="Token addresses to monitor"),
    min_probability_2x: Optional[float] = Query(
        None, description="Alert when 2x probability exceeds this"
    ),
    min_probability_10x: Optional[float] = Query(
        None, description="Alert when 10x probability exceeds this"
    ),
    max_risk_score: Optional[float] = Query(None, description="Alert when risk score exceeds this"),
    dev_sell_threshold: Optional[float] = Query(
        None, description="Alert when dev sells exceed this %"
    ),
):
    """
    Subscribe to alerts via webhook.

    Configure alerts for:
    - Token probability thresholds
    - Risk score changes
    - Developer wallet activity
    - Wallet activity monitoring

    Args:
        webhook_url: URL to receive alert webhooks
        wallet_addresses: Wallets to monitor
        token_addresses: Tokens to monitor
        min_probability_2x: 2x probability threshold
        min_probability_10x: 10x probability threshold
        max_risk_score: Risk score threshold
        dev_sell_threshold: Dev sell percentage threshold

    Returns:
        Subscription confirmation
    """
    # Create subscription record
    from uuid import uuid4
    
    subscription_id = f"sub_{uuid4().hex[:12]}"
    created_at = datetime.now(timezone.utc)
    
    # In production, persist to database
    subscription_data = {
        "id": subscription_id,
        "webhook_url": webhook_url,
        "filters": {
            "token_addresses": token_addresses,
            "min_probability_2x": min_probability_2x,
            "min_probability_10x": min_probability_10x,
            "max_risk_score": max_risk_score,
            "dev_sell_threshold": dev_sell_threshold,
        },
        "active": True,
        "created_at": created_at,
    }
    
    # TODO: Store in database when DB is configured
    # db.save_subscription(subscription_data)
    
    logger.info(f"Created alert subscription {subscription_id} for {webhook_url}")
    
    return {
        "status": "subscribed",
        "webhook_url": webhook_url,
        "subscription_id": subscription_id,
        "created_at": created_at.isoformat(),
    }


@app.post("/smart-wallets/find", response_model=SmartWalletFindResponse)
async def find_smart_wallets(request: SmartWalletFindRequest):
    """
    Discover probable smart-money wallets for a given token.

    This endpoint analyzes trading activity and identifies wallets that exhibit
    smart-money characteristics through:
    - Event timing (pre-pump/pre-dump detection)
    - Profitability metrics (win rate, ROI)
    - Transaction graph analysis
    - Behavioral patterns
    - Risk filtering

    Args:
        request: Smart wallet find request parameters

    Returns:
        Ranked list of smart wallets with scores and explanations
    """
    try:
        result = smart_wallet_finder.find_smart_wallets(
            token_address=request.token_address,
            chain=request.chain,
            window_days=request.window_days,
            min_trade_size_usd=request.min_trade_size_usd,
            min_holdings_usd=request.min_holdings_usd,
            include_internal_transfers=request.include_internal_transfers,
            top_k=request.top_k,
            min_confidence=request.min_confidence,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/smart-wallets/profile", response_model=WalletProfileResponse)
async def get_smart_wallet_profile(request: WalletProfileRequest):
    """
    Get detailed profile for a specific wallet and token.

    Returns comprehensive analytics including:
    - All trades for the token
    - Realized ROI and win rate
    - Graph neighbors and connections
    - Smart-money score
    - Detailed explanation of signals

    Args:
        request: Wallet profile request parameters

    Returns:
        Detailed wallet profile with trades and explanations
    """
    try:
        result = smart_wallet_finder.get_wallet_profile(
            wallet_address=request.wallet_address,
            token_address=request.token_address,
            chain=request.chain,
            window_days=request.window_days,
        )

        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/smart-wallets/bulk-scan", response_model=BulkScanResponse)
async def bulk_scan_tokens(request: BulkScanRequest):
    """
    Scan multiple tokens for smart wallets and produce aggregate leaderboard.

    Analyzes multiple tokens in parallel and produces:
    - Top smart wallets per token
    - Global leaderboard across all tokens
    - Aggregated statistics

    Args:
        request: Bulk scan request parameters

    Returns:
        Results per token and global leaderboard
    """
    try:
        results = []
        all_wallets = {}

        # Scan each token
        for token_address in request.token_addresses:
            token_result = smart_wallet_finder.find_smart_wallets(
                token_address=token_address,
                chain=request.chain,
                window_days=request.window_days,
                top_k=request.top_k_per_token,
                min_confidence=request.min_confidence,
            )

            results.append(
                {
                    "token_address": token_address,
                    "top_wallets": token_result["ranked_wallets"],
                    "avg_score": token_result["summary_stats"].get("avg_smart_money_score", 0),
                }
            )

            # Collect all wallets for global leaderboard
            for wallet in token_result["ranked_wallets"]:
                addr = wallet["wallet_address"]
                score = wallet["smart_money_score"]

                if addr not in all_wallets or all_wallets[addr]["smart_money_score"] < score:
                    all_wallets[addr] = wallet

        # Create global leaderboard
        leaderboard = sorted(
            all_wallets.values(), key=lambda w: w["smart_money_score"], reverse=True
        )[
            :100
        ]  # Top 100 globally

        # Re-rank
        for i, wallet in enumerate(leaderboard, 1):
            wallet["rank"] = i

        return {
            "chain": request.chain,
            "tokens_analyzed": len(request.token_addresses),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "results": results,
            "leaderboard": leaderboard,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.

    Provides observability metrics for monitoring:
    - Request counts by endpoint
    - Request durations
    - Prediction counts by chain
    - Error counts by type

    Returns:
        Prometheus-formatted metrics
    """
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Prometheus client not installed. Install with: pip install prometheus-client",
        )

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main():
    """Run the API server."""
    uvicorn.run(
        "radarx.api.server:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
