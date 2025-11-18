"""
Advanced Caching Strategies

Implements intelligent caching with:
- Prediction caching with similarity-based lookup
- Feature vector caching with TTL
- Smart cache invalidation based on market changes
- Cache warming for popular tokens
- Multi-level cache (memory + distributed)
"""

import asyncio
import hashlib
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SimilarityCache:
    """
    Cache that can return similar results for similar inputs.

    Uses feature vector similarity to find cached results even when
    exact match doesn't exist. Useful for tokens with similar characteristics.
    """

    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 300,
    ):
        """
        Initialize similarity cache.

        Args:
            max_size: Maximum cache entries
            similarity_threshold: Minimum similarity for cache hit (0-1)
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds

        self._cache: OrderedDict = OrderedDict()
        self._features: Dict[str, np.ndarray] = {}
        self._expiry: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

        # Statistics
        self.hits = 0
        self.misses = 0
        self.similarity_hits = 0

    def _compute_key(self, identifier: str, features: Optional[np.ndarray] = None) -> str:
        """Compute cache key."""
        if features is not None:
            # Include feature hash for uniqueness
            feature_hash = hashlib.md5(features.tobytes()).hexdigest()[:8]
            return f"{identifier}:{feature_hash}"
        return identifier

    def _compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """Compute cosine similarity between feature vectors."""
        try:
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(features1, features2) / (norm1 * norm2)
            return float(similarity)
        except Exception as e:
            logger.warning(f"Similarity computation failed: {e}")
            return 0.0

    async def get(self, identifier: str, features: Optional[np.ndarray] = None) -> Optional[Any]:
        """
        Get value from cache with similarity-based lookup.

        Args:
            identifier: Primary identifier (e.g., token address)
            features: Feature vector for similarity matching

        Returns:
            Cached value or None
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            # Try exact match first
            key = self._compute_key(identifier, features)

            if key in self._cache:
                # Check expiry
                if now < self._expiry[key]:
                    self.hits += 1
                    # Move to end (LRU)
                    self._cache.move_to_end(key)
                    return self._cache[key]
                else:
                    # Expired
                    del self._cache[key]
                    del self._features[key]
                    del self._expiry[key]

            # If features provided, try similarity matching
            if features is not None:
                best_similarity = 0.0
                best_key = None

                for cached_key, cached_features in self._features.items():
                    # Check if not expired
                    if now >= self._expiry.get(cached_key, now):
                        continue

                    # Compute similarity
                    similarity = self._compute_similarity(features, cached_features)

                    if similarity > best_similarity and similarity >= self.similarity_threshold:
                        best_similarity = similarity
                        best_key = cached_key

                if best_key:
                    self.similarity_hits += 1
                    logger.debug(
                        f"Similarity cache hit: {identifier} (similarity: {best_similarity:.3f})"
                    )
                    return self._cache[best_key]

            self.misses += 1
            return None

    async def set(self, identifier: str, value: Any, features: Optional[np.ndarray] = None):
        """
        Store value in cache.

        Args:
            identifier: Primary identifier
            value: Value to cache
            features: Feature vector for similarity matching
        """
        async with self._lock:
            key = self._compute_key(identifier, features)

            # Evict oldest if at capacity
            if len(self._cache) >= self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                if oldest_key in self._features:
                    del self._features[oldest_key]
                if oldest_key in self._expiry:
                    del self._expiry[oldest_key]

            # Store
            self._cache[key] = value
            if features is not None:
                self._features[key] = features
            self._expiry[key] = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)

    async def invalidate_similar(self, features: np.ndarray, threshold: float = 0.90):
        """
        Invalidate cache entries similar to given features.

        Useful when market conditions change and similar tokens
        should have their predictions refreshed.

        Args:
            features: Feature vector
            threshold: Similarity threshold for invalidation
        """
        async with self._lock:
            to_remove = []

            for cached_key, cached_features in self._features.items():
                similarity = self._compute_similarity(features, cached_features)
                if similarity >= threshold:
                    to_remove.append(cached_key)

            for key in to_remove:
                if key in self._cache:
                    del self._cache[key]
                if key in self._features:
                    del self._features[key]
                if key in self._expiry:
                    del self._expiry[key]

            if to_remove:
                logger.info(f"Invalidated {len(to_remove)} similar cache entries")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "similarity_hits": self.similarity_hits,
            "hit_rate": hit_rate,
        }


class AdaptiveTTLCache:
    """
    Cache with adaptive TTL based on data volatility.

    Frequently changing data gets shorter TTL, stable data gets longer TTL.
    """

    def __init__(
        self,
        base_ttl: int = 300,
        min_ttl: int = 60,
        max_ttl: int = 3600,
    ):
        """
        Initialize adaptive TTL cache.

        Args:
            base_ttl: Base TTL in seconds
            min_ttl: Minimum TTL
            max_ttl: Maximum TTL
        """
        self.base_ttl = base_ttl
        self.min_ttl = min_ttl
        self.max_ttl = max_ttl

        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
        self._update_history: Dict[str, List[datetime]] = {}
        self._lock = asyncio.Lock()

    def _compute_ttl(self, key: str) -> int:
        """
        Compute adaptive TTL based on update frequency.

        Args:
            key: Cache key

        Returns:
            TTL in seconds
        """
        if key not in self._update_history:
            return self.base_ttl

        updates = self._update_history[key]
        if len(updates) < 2:
            return self.base_ttl

        # Calculate average time between updates
        deltas = [(updates[i + 1] - updates[i]).total_seconds() for i in range(len(updates) - 1)]
        avg_delta = sum(deltas) / len(deltas)

        # Shorter updates = shorter TTL (data changes frequently)
        # Use fraction of average delta
        adaptive_ttl = int(avg_delta * 0.5)

        # Clamp to limits
        return max(self.min_ttl, min(self.max_ttl, adaptive_ttl))

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        async with self._lock:
            if key in self._cache:
                if datetime.now(timezone.utc) < self._expiry[key]:
                    return self._cache[key]
                else:
                    # Expired
                    del self._cache[key]
                    del self._expiry[key]
            return None

    async def set(self, key: str, value: Any):
        """
        Set value in cache with adaptive TTL.

        Args:
            key: Cache key
            value: Value to cache
        """
        async with self._lock:
            now = datetime.now(timezone.utc)

            # Track update
            if key not in self._update_history:
                self._update_history[key] = []
            self._update_history[key].append(now)

            # Keep only recent history (last 10 updates)
            if len(self._update_history[key]) > 10:
                self._update_history[key] = self._update_history[key][-10:]

            # Compute TTL
            ttl = self._compute_ttl(key)

            # Store
            self._cache[key] = value
            self._expiry[key] = now + timedelta(seconds=ttl)

            logger.debug(f"Cached {key} with adaptive TTL: {ttl}s")


class CacheWarmer:
    """
    Proactively warms cache for popular or trending tokens.

    Runs in background to ensure hot data is always cached.
    """

    def __init__(
        self,
        cache: Any,
        warm_interval: int = 60,
        top_n: int = 100,
    ):
        """
        Initialize cache warmer.

        Args:
            cache: Cache instance to warm
            warm_interval: How often to warm cache (seconds)
            top_n: Number of popular tokens to keep warm
        """
        self.cache = cache
        self.warm_interval = warm_interval
        self.top_n = top_n

        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self, token_fetcher: callable):
        """
        Start cache warming in background.

        Args:
            token_fetcher: Async function that returns popular tokens
        """
        self._running = True
        self._task = asyncio.create_task(self._warm_loop(token_fetcher))
        logger.info("Cache warmer started")

    async def stop(self):
        """Stop cache warming."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Cache warmer stopped")

    async def _warm_loop(self, token_fetcher: callable):
        """Background loop that warms cache."""
        while self._running:
            try:
                # Get popular tokens
                tokens = await token_fetcher(self.top_n)

                # Warm cache for each
                for token in tokens:
                    if not self._running:
                        break

                    # Check if already cached
                    cached = await self.cache.get(token["address"])
                    if cached is None:
                        # Fetch and cache
                        # This would call the actual scoring service
                        logger.debug(f"Warming cache for {token['address']}")
                        # await score_and_cache(token['address'])

                # Wait before next warming
                await asyncio.sleep(self.warm_interval)

            except Exception as e:
                logger.error(f"Cache warming error: {e}")
                await asyncio.sleep(self.warm_interval)


class MultiLevelCache:
    """
    Multi-level cache with memory (L1) and distributed (L2) tiers.

    Fast memory cache backed by Redis/Memcached for distributed access.
    """

    def __init__(
        self,
        l1_cache: Any,
        l2_cache: Optional[Any] = None,
    ):
        """
        Initialize multi-level cache.

        Args:
            l1_cache: Fast memory cache (L1)
            l2_cache: Distributed cache (L2), optional
        """
        self.l1 = l1_cache
        self.l2 = l2_cache

        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        # Try L1 first
        value = await self.l1.get(key)
        if value is not None:
            self.l1_hits += 1
            return value

        # Try L2
        if self.l2:
            value = await self.l2.get(key)
            if value is not None:
                self.l2_hits += 1
                # Promote to L1
                await self.l1.set(key, value)
                return value

        self.misses += 1
        return None

    async def set(self, key: str, value: Any):
        """Set value in multi-level cache."""
        # Store in both levels
        await self.l1.set(key, value)
        if self.l2:
            await self.l2.set(key, value)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.l1_hits + self.l2_hits + self.misses
        return {
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "l1_hit_rate": self.l1_hits / total if total > 0 else 0.0,
            "total_hit_rate": (self.l1_hits + self.l2_hits) / total if total > 0 else 0.0,
        }
