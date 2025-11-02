"""DexScreener API client for token data."""

import httpx
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from radarx.data.cache import CacheManager
from radarx.config import settings

logger = logging.getLogger(__name__)


class DexScreenerClient:
    """Client for DexScreener API to fetch token price, volume, and liquidity data."""
    
    BASE_URL = "https://api.dexscreener.com/latest/dex"
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize DexScreener client.
        
        Args:
            cache_manager: Optional cache manager for caching responses
        """
        self.cache = cache_manager or CacheManager()
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "RadarX/0.1.0",
            }
        )
    
    async def get_token_pairs(
        self,
        token_address: str,
        chain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all trading pairs for a token.
        
        Args:
            token_address: Token contract address
            chain: Optional chain filter (ethereum, bsc, solana, etc.)
            
        Returns:
            List of trading pair data
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        cache_key = f"dex:pairs:{chain}:{token_address}"
        cached = await self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        try:
            url = f"{self.BASE_URL}/tokens/{token_address}"
            logger.info(f"Fetching DexScreener pairs for {token_address}")
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            pairs = data.get("pairs", [])
            
            # Filter by chain if specified
            if chain:
                pairs = [p for p in pairs if p.get("chainId", "").lower() == chain.lower()]
            
            # Cache for 1 minute
            await self.cache.set(cache_key, pairs, ttl=60)
            
            logger.info(f"Found {len(pairs)} pairs for {token_address}")
            return pairs
            
        except httpx.HTTPError as e:
            logger.error(f"DexScreener API error: {e}")
            raise
    
    async def get_token_data(
        self,
        token_address: str,
        chain: str = "ethereum"
    ) -> Optional[Dict[str, Any]]:
        """Get aggregated token data (price, volume, liquidity).
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            
        Returns:
            Aggregated token data or None if not found
        """
        pairs = await self.get_token_pairs(token_address, chain)
        
        if not pairs:
            logger.warning(f"No pairs found for {token_address} on {chain}")
            return None
        
        # Use the pair with highest liquidity
        main_pair = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0))
        
        return {
            "address": token_address,
            "chain": chain,
            "symbol": main_pair.get("baseToken", {}).get("symbol"),
            "name": main_pair.get("baseToken", {}).get("name"),
            "price_usd": float(main_pair.get("priceUsd", 0) or 0),
            "price_native": float(main_pair.get("priceNative", 0) or 0),
            "volume_24h": float(main_pair.get("volume", {}).get("h24", 0) or 0),
            "liquidity_usd": float(main_pair.get("liquidity", {}).get("usd", 0) or 0),
            "price_change_24h": float(main_pair.get("priceChange", {}).get("h24", 0) or 0),
            "price_change_6h": float(main_pair.get("priceChange", {}).get("h6", 0) or 0),
            "price_change_1h": float(main_pair.get("priceChange", {}).get("h1", 0) or 0),
            "txns_24h_buys": int(main_pair.get("txns", {}).get("h24", {}).get("buys", 0) or 0),
            "txns_24h_sells": int(main_pair.get("txns", {}).get("h24", {}).get("sells", 0) or 0),
            "market_cap": float(main_pair.get("fdv", 0) or 0),  # Fully diluted valuation
            "pair_address": main_pair.get("pairAddress"),
            "dex_id": main_pair.get("dexId"),
            "pair_created_at": main_pair.get("pairCreatedAt"),
            "timestamp": datetime.utcnow().isoformat(),
        }
    
    async def search_pairs(
        self,
        query: str,
        chain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for pairs by token name or symbol.
        
        Args:
            query: Search query (token name or symbol)
            chain: Optional chain filter
            
        Returns:
            List of matching pairs
        """
        cache_key = f"dex:search:{chain}:{query}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/search"
            params = {"q": query}
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            pairs = data.get("pairs", [])
            
            if chain:
                pairs = [p for p in pairs if p.get("chainId", "").lower() == chain.lower()]
            
            # Cache for 5 minutes
            await self.cache.set(cache_key, pairs, ttl=300)
            
            return pairs
            
        except httpx.HTTPError as e:
            logger.error(f"DexScreener search error: {e}")
            return []
    
    async def get_boosted_tokens(self) -> List[Dict[str, Any]]:
        """Get list of boosted/promoted tokens on DexScreener.
        
        Returns:
            List of boosted token data
        """
        cache_key = "dex:boosted"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/tokens/boosted"
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            # Cache for 10 minutes
            await self.cache.set(cache_key, data, ttl=600)
            
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"DexScreener boosted tokens error: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
