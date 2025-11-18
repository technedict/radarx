"""Configuration management for RadarX."""

from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    api_title: str = "RadarX API"
    api_version: str = "0.1.0"
    api_description: str = "Production-grade memecoin analysis and wallet intelligence"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # CORS
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")

    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # Database
    database_url: str = "sqlite:///./radarx.db"
    redis_url: str = "redis://localhost:6379/0"

    # Feature Store
    feature_store_backend: str = "redis"
    feature_ttl_hours: int = 24

    # Model Settings
    model_dir: str = "./models"
    model_version: str = "v1.2.3"
    feature_version: str = "v2.0.1"
    enable_online_learning: bool = True

    # Data Sources
    dexscreener_api_key: str = ""
    etherscan_api_key: str = ""
    bscscan_api_key: str = ""
    solscan_api_key: str = ""
    helius_api_key: str = ""
    birdeye_api_key: str = ""
    quicknode_url: str = ""
    alchemy_api_key: str = ""

    # Social APIs
    twitter_bearer_token: str = ""
    telegram_bot_token: str = ""
    reddit_client_id: str = ""
    reddit_client_secret: str = ""

    # Risk Feeds
    rugcheck_api_key: str = ""
    goplus_api_key: str = ""

    # Backtesting
    backtest_start_date: str = "2023-01-01"
    backtest_end_date: str = "2024-01-01"
    backtest_fee_rate: float = 0.003
    backtest_slippage_rate: float = 0.001

    # Monitoring
    enable_prometheus: bool = True
    enable_sentry: bool = False
    sentry_dsn: str = ""
    log_level: str = "INFO"

    # Cache
    cache_ttl_seconds: int = 300
    cache_max_size: int = 10000

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Global settings instance
settings = Settings()
