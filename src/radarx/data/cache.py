"""Cache manager for data ingestion layer."""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


class CacheManager:
    """In-memory cache manager with TTL support and async locks.

    In production, this would use Redis or similar distributed cache.
    For now, using simple in-memory dict for development with async safety.
    """

    def __init__(self):
        """Initialize cache manager."""
        self._cache: dict = {}
        self._expiry: dict = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            # Check if key exists and not expired
            if key in self._cache:
                expiry = self._expiry.get(key)
                if expiry is None or datetime.now(timezone.utc) < expiry:
                    logger.debug(f"Cache hit: {key}")
                    return self._cache[key]
                else:
                    # Expired, remove from cache
                    logger.debug(f"Cache expired: {key}")
                    del self._cache[key]
                    del self._expiry[key]

            logger.debug(f"Cache miss: {key}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache with TTL.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default 5 minutes)
        """
        async with self._lock:
            self._cache[key] = value
            self._expiry[key] = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    async def delete(self, key: str):
        """Delete key from cache.

        Args:
            key: Cache key to delete
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._expiry:
                    del self._expiry[key]
                logger.debug(f"Cache deleted: {key}")
    
    async def clear(self):
        """Clear all cache entries."""
        async with self._lock:
            self._cache.clear()
            self._expiry.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """Get cache statistics.
        
        Returns:
            Dict with cache stats
        """
        total_keys = len(self._cache)
        expired_keys = sum(
            1 for k, exp in self._expiry.items()
            if exp and datetime.now(timezone.utc) >= exp
        )
        
        return {
            "total_keys": total_keys,
            "active_keys": total_keys - expired_keys,
            "expired_keys": expired_keys,
        }


class RedisCache(CacheManager):
    """Redis-based cache manager for production use.
    
    This is a placeholder for future Redis integration.
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis cache.
        
        Args:
            redis_url: Redis connection URL
        """
        super().__init__()
        self.redis_url = redis_url
        # TODO: Initialize Redis connection when redis library is available
        logger.warning("RedisCache not fully implemented, using in-memory cache")
