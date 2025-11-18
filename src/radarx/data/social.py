"""Social media API clients for sentiment and mention tracking."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from radarx.config import settings
from radarx.data.cache import CacheManager

logger = logging.getLogger(__name__)


class SocialMediaClient(ABC):
    """Abstract base class for social media API clients."""
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """Initialize social media client.
        
        Args:
            cache_manager: Optional cache manager
        """
        self.cache = cache_manager or CacheManager()
    
    @abstractmethod
    async def search_mentions(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for mentions of a query."""
        pass
    
    @abstractmethod
    async def get_user_tweets(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent tweets from a user."""
        pass


class TwitterClient(SocialMediaClient):
    """Twitter API v2 client for tweet and mention tracking."""
    
    def __init__(
        self,
        bearer_token: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """Initialize Twitter client.
        
        Args:
            bearer_token: Twitter API bearer token
            cache_manager: Optional cache manager
        """
        super().__init__(cache_manager)
        self.bearer_token = bearer_token or settings.twitter_bearer_token
        self.client = httpx.AsyncClient(
            base_url="https://api.twitter.com/2",
            headers={
                "Authorization": f"Bearer {self.bearer_token}",
                "User-Agent": "RadarX/0.1.0",
            },
            timeout=30.0,
        )
    
    async def search_mentions(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search recent tweets mentioning a query.
        
        Args:
            query: Search query (token symbol, name, etc.)
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum results
            
        Returns:
            List of tweet data
        """
        if not self.bearer_token:
            logger.warning("Twitter bearer token not configured")
            return []
        
        cache_key = f"twitter:search:{query}:{start_time}:{end_time}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            params = {
                "query": query,
                "max_results": min(limit, 100),
                "tweet.fields": "created_at,public_metrics,author_id",
            }
            
            if start_time:
                params["start_time"] = start_time.isoformat() + "Z"
            if end_time:
                params["end_time"] = end_time.isoformat() + "Z"
            
            response = await self.client.get("/tweets/search/recent", params=params)
            response.raise_for_status()
            
            data = response.json()
            tweets = data.get("data", [])
            
            # Cache for 2 minutes
            await self.cache.set(cache_key, tweets, ttl=120)
            
            logger.info(f"Found {len(tweets)} tweets for query: {query}")
            return tweets
            
        except httpx.HTTPError as e:
            logger.error(f"Twitter API error: {e}")
            return []
    
    async def get_user_tweets(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get recent tweets from a user.
        
        Args:
            user_id: Twitter user ID
            limit: Maximum results
            
        Returns:
            List of tweet data
        """
        if not self.bearer_token:
            logger.warning("Twitter bearer token not configured")
            return []
        
        try:
            params = {
                "max_results": min(limit, 100),
                "tweet.fields": "created_at,public_metrics",
            }
            
            response = await self.client.get(f"/users/{user_id}/tweets", params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("data", [])
            
        except httpx.HTTPError as e:
            logger.error(f"Twitter API error: {e}")
            return []
    
    async def get_mention_stats(
        self,
        query: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get aggregated mention statistics.
        
        Args:
            query: Search query
            hours: Time window in hours
            
        Returns:
            Stats including volume, velocity, top influencers
        """
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        tweets = await self.search_mentions(query, start_time, end_time, limit=100)
        
        if not tweets:
            return {
                "mention_volume": 0,
                "unique_authors": 0,
                "total_likes": 0,
                "total_retweets": 0,
                "average_engagement": 0,
            }
        
        unique_authors = len(set(t.get("author_id") for t in tweets))
        total_likes = sum(t.get("public_metrics", {}).get("like_count", 0) for t in tweets)
        total_retweets = sum(t.get("public_metrics", {}).get("retweet_count", 0) for t in tweets)
        
        return {
            "mention_volume": len(tweets),
            "unique_authors": unique_authors,
            "total_likes": total_likes,
            "total_retweets": total_retweets,
            "average_engagement": (total_likes + total_retweets) / len(tweets) if tweets else 0,
            "time_window_hours": hours,
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class TelegramClient(SocialMediaClient):
    """Telegram Bot API client for group/channel monitoring.
    
    Note: Requires bot token and proper permissions.
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """Initialize Telegram client.
        
        Args:
            bot_token: Telegram bot token
            cache_manager: Optional cache manager
        """
        super().__init__(cache_manager)
        self.bot_token = bot_token or settings.telegram_bot_token
        
        if self.bot_token:
            self.client = httpx.AsyncClient(
                base_url=f"https://api.telegram.org/bot{self.bot_token}",
                timeout=30.0,
            )
        else:
            self.client = None
    
    async def search_mentions(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for mentions in Telegram.
        
        Note: This is a placeholder. Full implementation would require
        monitoring specific channels/groups.
        
        Args:
            query: Search query
            start_time: Start time
            end_time: End time
            limit: Result limit
            
        Returns:
            Empty list (not implemented)
        """
        logger.warning("Telegram search not fully implemented")
        return []
    
    async def get_user_tweets(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get user messages (not applicable for Telegram)."""
        logger.warning("Telegram user messages not applicable")
        return []
    
    async def get_channel_info(
        self,
        channel_username: str
    ) -> Dict[str, Any]:
        """Get information about a Telegram channel.
        
        Args:
            channel_username: Channel username (e.g., @channelname)
            
        Returns:
            Channel information
        """
        if not self.client:
            logger.warning("Telegram bot token not configured")
            return {}
        
        try:
            response = await self.client.get(
                "/getChat",
                params={"chat_id": channel_username}
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("result", {})
            
        except httpx.HTTPError as e:
            logger.error(f"Telegram API error: {e}")
            return {}
    
    async def close(self):
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()


class RedditClient(SocialMediaClient):
    """Reddit API client for subreddit and post monitoring."""
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """Initialize Reddit client.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            cache_manager: Optional cache manager
        """
        super().__init__(cache_manager)
        self.client_id = client_id or settings.reddit_client_id
        self.client_secret = client_secret or settings.reddit_client_secret
        self.client = httpx.AsyncClient(
            base_url="https://oauth.reddit.com",
            timeout=30.0,
        )
        self.access_token: Optional[str] = None
    
    async def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token for Reddit API.
        
        Returns:
            Access token or None
        """
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit credentials not configured")
            return None
        
        try:
            auth = httpx.BasicAuth(self.client_id, self.client_secret)
            response = await httpx.AsyncClient().post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data={"grant_type": "client_credentials"},
                headers={"User-Agent": "RadarX/0.1.0"},
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get("access_token")
            
        except httpx.HTTPError as e:
            logger.error(f"Reddit auth error: {e}")
            return None
    
    async def search_mentions(
        self,
        query: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Search Reddit for mentions.
        
        Args:
            query: Search query
            start_time: Start time (not supported by Reddit API)
            end_time: End time (not supported by Reddit API)
            limit: Result limit
            
        Returns:
            List of Reddit posts/comments
        """
        if not self.access_token:
            self.access_token = await self._get_access_token()
        
        if not self.access_token:
            return []
        
        try:
            self.client.headers.update({"Authorization": f"Bearer {self.access_token}"})
            
            response = await self.client.get(
                "/search",
                params={
                    "q": query,
                    "limit": min(limit, 100),
                    "sort": "new",
                }
            )
            response.raise_for_status()
            
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            
            return [post.get("data", {}) for post in posts]
            
        except httpx.HTTPError as e:
            logger.error(f"Reddit API error: {e}")
            return []
    
    async def get_user_tweets(
        self,
        user_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get user posts from Reddit.
        
        Args:
            user_id: Reddit username
            limit: Result limit
            
        Returns:
            List of user posts
        """
        if not self.access_token:
            self.access_token = await self._get_access_token()
        
        if not self.access_token:
            return []
        
        try:
            self.client.headers.update({"Authorization": f"Bearer {self.access_token}"})
            
            response = await self.client.get(
                f"/user/{user_id}/submitted",
                params={"limit": min(limit, 100)}
            )
            response.raise_for_status()
            
            data = response.json()
            posts = data.get("data", {}).get("children", [])
            
            return [post.get("data", {}) for post in posts]
            
        except httpx.HTTPError as e:
            logger.error(f"Reddit API error: {e}")
            return []
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
