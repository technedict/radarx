"""Data ingestion module for RadarX.

This module provides adapters for various data sources including:
- DexScreener for token data
- On-chain indexers (Etherscan, BscScan, Solscan)
- Social media APIs (Twitter, Telegram, Reddit)
- Risk assessment feeds (RugCheck, GoPlus)
"""

from radarx.data.dexscreener import DexScreenerClient
from radarx.data.blockchain import (
    EtherscanClient,
    BscScanClient,
    SolscanClient,
    BlockchainIndexer,
)
from radarx.data.social import TwitterClient, TelegramClient, RedditClient
from radarx.data.risk_feeds import RugCheckClient, GoPlusClient
from radarx.data.cache import CacheManager
from radarx.data.normalizer import DataNormalizer

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
