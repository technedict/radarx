"""Social signal feature extraction."""

import logging
import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from radarx.data import DataNormalizer, RedditClient, TwitterClient

logger = logging.getLogger(__name__)


class SocialFeatureExtractor:
    """Extract social media signal features."""

    def __init__(
        self,
        twitter_client: Optional[TwitterClient] = None,
        reddit_client: Optional[RedditClient] = None,
        kol_threshold: int = 10000,  # Min followers to be considered KOL
    ):
        """Initialize social feature extractor.

        Args:
            twitter_client: Twitter API client
            reddit_client: Reddit API client
            kol_threshold: Minimum followers for KOL classification
        """
        self.twitter_client = twitter_client
        self.reddit_client = reddit_client
        self.kol_threshold = kol_threshold

    async def extract_mention_volume_features(
        self, query: str, hours: int = 24
    ) -> Dict[str, float]:
        """Extract mention volume features from social media.

        Args:
            query: Search query (token symbol, name, etc.)
            hours: Time window in hours

        Returns:
            Dict of mention volume features
        """
        features = {}

        # Initialize with zeros
        features[f"twitter_mentions_{hours}h"] = 0
        features[f"reddit_mentions_{hours}h"] = 0
        features[f"total_mentions_{hours}h"] = 0

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        # Twitter mentions
        if self.twitter_client:
            try:
                tweets = await self.twitter_client.search_mentions(
                    query, start_time, end_time, limit=100
                )
                features[f"twitter_mentions_{hours}h"] = len(tweets)
            except Exception as e:
                logger.warning(f"Could not fetch Twitter mentions: {e}")

        # Reddit mentions
        if self.reddit_client:
            try:
                posts = await self.reddit_client.search_mentions(
                    query, start_time, end_time, limit=100
                )
                features[f"reddit_mentions_{hours}h"] = len(posts)
            except Exception as e:
                logger.warning(f"Could not fetch Reddit mentions: {e}")

        # Total mentions
        features[f"total_mentions_{hours}h"] = (
            features[f"twitter_mentions_{hours}h"] + features[f"reddit_mentions_{hours}h"]
        )

        return features

    async def extract_mention_velocity_features(self, query: str) -> Dict[str, float]:
        """Extract mention velocity (rate of change) features.

        Args:
            query: Search query

        Returns:
            Dict of velocity features
        """
        features = {}

        # Get mentions for different time windows
        mentions_1h = await self.extract_mention_volume_features(query, hours=1)
        mentions_6h = await self.extract_mention_volume_features(query, hours=6)
        mentions_24h = await self.extract_mention_volume_features(query, hours=24)

        # Calculate velocity (mentions per hour)
        features["mention_velocity_1h"] = mentions_1h.get("total_mentions_1h", 0)
        features["mention_velocity_6h"] = mentions_6h.get("total_mentions_6h", 0) / 6
        features["mention_velocity_24h"] = mentions_24h.get("total_mentions_24h", 0) / 24

        # Acceleration (change in velocity)
        if features["mention_velocity_6h"] > 0:
            features["mention_acceleration"] = (
                features["mention_velocity_1h"] / features["mention_velocity_6h"]
            )
        else:
            features["mention_acceleration"] = 0

        return features

    async def extract_engagement_features(self, query: str, hours: int = 24) -> Dict[str, float]:
        """Extract engagement metrics from social media.

        Args:
            query: Search query
            hours: Time window in hours

        Returns:
            Dict of engagement features
        """
        features = {}

        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)

        total_likes = 0
        total_retweets = 0
        total_replies = 0
        total_engagement = 0
        tweet_count = 0

        if self.twitter_client:
            try:
                tweets = await self.twitter_client.search_mentions(
                    query, start_time, end_time, limit=100
                )

                for tweet in tweets:
                    metrics = tweet.get("public_metrics", {})
                    total_likes += metrics.get("like_count", 0)
                    total_retweets += metrics.get("retweet_count", 0)
                    total_replies += metrics.get("reply_count", 0)

                tweet_count = len(tweets)
                total_engagement = total_likes + total_retweets + total_replies

            except Exception as e:
                logger.warning(f"Could not fetch Twitter engagement: {e}")

        features["total_likes"] = total_likes
        features["total_retweets"] = total_retweets
        features["total_replies"] = total_replies
        features["total_engagement"] = total_engagement

        # Average engagement per tweet
        if tweet_count > 0:
            features["avg_engagement_per_tweet"] = total_engagement / tweet_count
        else:
            features["avg_engagement_per_tweet"] = 0

        return features

    def extract_sentiment_features(self, texts: List[str]) -> Dict[str, float]:
        """Extract simple sentiment features from text.

        Note: This is a simple rule-based sentiment. In production,
        use a proper sentiment analysis model.

        Args:
            texts: List of text strings (tweets, posts, etc.)

        Returns:
            Dict of sentiment features
        """
        features = {}

        if not texts:
            return {
                "sentiment_score": 0.0,
                "sentiment_positive_ratio": 0.0,
                "sentiment_negative_ratio": 0.0,
            }

        # Simple positive/negative word lists
        positive_words = {
            "bullish",
            "moon",
            "pump",
            "rocket",
            "gain",
            "profit",
            "up",
            "buy",
            "long",
            "hold",
            "diamond",
            "hands",
        }
        negative_words = {
            "bearish",
            "dump",
            "rug",
            "scam",
            "loss",
            "sell",
            "down",
            "short",
            "dead",
            "crash",
            "paper",
            "hands",
        }

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for text in texts:
            text_lower = text.lower()

            # Count positive/negative words
            pos_matches = sum(1 for word in positive_words if word in text_lower)
            neg_matches = sum(1 for word in negative_words if word in text_lower)

            if pos_matches > neg_matches:
                positive_count += 1
            elif neg_matches > pos_matches:
                negative_count += 1
            else:
                neutral_count += 1

        total = len(texts)
        features["sentiment_positive_ratio"] = positive_count / total
        features["sentiment_negative_ratio"] = negative_count / total
        features["sentiment_neutral_ratio"] = neutral_count / total

        # Overall sentiment score (-1 to 1)
        features["sentiment_score"] = (positive_count - negative_count) / total

        return features

    async def detect_kol_mentions(self, query: str, hours: int = 24) -> Dict[str, float]:
        """Detect mentions from key opinion leaders (KOLs).

        Args:
            query: Search query
            hours: Time window in hours

        Returns:
            Dict of KOL-related features
        """
        features = {}

        kol_mentions = 0
        kol_total_followers = 0

        if self.twitter_client:
            try:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours)

                tweets = await self.twitter_client.search_mentions(
                    query, start_time, end_time, limit=100
                )

                # In production, would fetch user data for each author
                # For now, using simplified approach
                for tweet in tweets:
                    # Estimate: if high engagement, likely from KOL
                    metrics = tweet.get("public_metrics", {})
                    followers_estimate = metrics.get("retweet_count", 0) * 100

                    if followers_estimate >= self.kol_threshold:
                        kol_mentions += 1
                        kol_total_followers += followers_estimate

            except Exception as e:
                logger.warning(f"Could not detect KOL mentions: {e}")

        features["kol_mentions"] = kol_mentions
        features["kol_total_reach"] = kol_total_followers

        if kol_mentions > 0:
            features["kol_avg_followers"] = kol_total_followers / kol_mentions
        else:
            features["kol_avg_followers"] = 0

        return features

    def detect_bot_patterns(self, tweets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Detect bot-like posting patterns.

        Args:
            tweets: List of tweet data

        Returns:
            Dict of bot detection features
        """
        features = {}

        if not tweets:
            return {
                "bot_score": 0.0,
                "duplicate_ratio": 0.0,
            }

        # Check for duplicate or near-duplicate tweets
        texts = [t.get("text", "") for t in tweets]
        text_counts = Counter(texts)

        duplicates = sum(1 for count in text_counts.values() if count > 1)
        features["duplicate_ratio"] = duplicates / len(tweets)

        # Check for rapid-fire posting from same authors
        author_tweets = Counter(t.get("author_id") for t in tweets)
        max_tweets_per_author = max(author_tweets.values()) if author_tweets else 0

        # Simple bot score (0-1)
        bot_score = 0.0

        # High duplicate ratio suggests bots
        if features["duplicate_ratio"] > 0.5:
            bot_score += 0.4

        # Single author posting many times suggests bot
        if max_tweets_per_author > 10:
            bot_score += 0.3

        # Very similar timestamps suggest coordinated bots
        # (simplified - would analyze actual timestamp clustering in production)
        if len(tweets) > 5:
            bot_score += 0.3

        features["bot_score"] = min(1.0, bot_score)

        return features

    async def extract_all_features(self, query: str, hours: int = 24) -> Dict[str, float]:
        """Extract all social features.

        Args:
            query: Search query (token symbol, name, etc.)
            hours: Time window in hours

        Returns:
            Complete feature dictionary
        """
        all_features = {}

        # Mention volume
        mention_features = await self.extract_mention_volume_features(query, hours)
        all_features.update(mention_features)

        # Mention velocity
        velocity_features = await self.extract_mention_velocity_features(query)
        all_features.update(velocity_features)

        # Engagement metrics
        engagement_features = await self.extract_engagement_features(query, hours)
        all_features.update(engagement_features)

        # KOL detection
        kol_features = await self.detect_kol_mentions(query, hours)
        all_features.update(kol_features)

        # Get tweets for sentiment and bot detection
        if self.twitter_client:
            try:
                end_time = datetime.utcnow()
                start_time = end_time - timedelta(hours=hours)
                tweets = await self.twitter_client.search_mentions(
                    query, start_time, end_time, limit=100
                )

                # Sentiment analysis
                texts = [t.get("text", "") for t in tweets]
                sentiment_features = self.extract_sentiment_features(texts)
                all_features.update(sentiment_features)

                # Bot detection
                bot_features = self.detect_bot_patterns(tweets)
                all_features.update(bot_features)

            except Exception as e:
                logger.warning(f"Could not extract tweet features: {e}")

        return all_features

    async def close(self):
        """Close HTTP clients."""
        if self.twitter_client:
            await self.twitter_client.close()
        if self.reddit_client:
            await self.reddit_client.close()
