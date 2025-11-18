"""Data ingestion module for RadarX.

This module provides adapters for various data sources including:
- DexScreener for token data
- On-chain indexers (Etherscan, BscScan, Solscan)
- Social media APIs (Twitter, Telegram, Reddit)
- Risk assessment feeds (RugCheck, GoPlus)
"""

from radarx.data.blockchain import (
    BlockchainIndexer,
    BscScanClient,
    EtherscanClient,
    SolscanClient,
)
from radarx.data.cache import CacheManager
from radarx.data.dexscreener import DexScreenerClient
from radarx.data.normalizer import DataNormalizer
from radarx.data.risk_feeds import GoPlusClient, RugCheckClient
from radarx.data.social import RedditClient, TelegramClient, TwitterClient

__all__ = [
    "DexScreenerClient",
    "EtherscanClient",
    "BscScanClient",
    "SolscanClient",
    "BlockchainIndexer",
    "TwitterClient",
    "TelegramClient",
    "RedditClient",
    "RugCheckClient",
    "GoPlusClient",
    "CacheManager",
    "DataNormalizer",
]
