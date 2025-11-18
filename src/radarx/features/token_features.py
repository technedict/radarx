"""Token-level feature extraction."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from radarx.data import DataNormalizer, DexScreenerClient
from radarx.data.blockchain import BlockchainIndexer

logger = logging.getLogger(__name__)


class TokenFeatureExtractor:
    """Extract comprehensive token-level features."""

    def __init__(
        self,
        dex_client: Optional[DexScreenerClient] = None,
        blockchain_client: Optional[BlockchainIndexer] = None,
    ):
        """Initialize token feature extractor.

        Args:
            dex_client: DexScreener client for price/volume data
            blockchain_client: Blockchain indexer for on-chain data
        """
        self.dex_client = dex_client or DexScreenerClient()
        self.blockchain_client = blockchain_client

    async def extract_market_features(self, token_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract market-related features.

        Args:
            token_data: Token data from DexScreener or similar

        Returns:
            Dict of market features
        """
        features = {}

        # Basic market metrics
        features["market_cap"] = DataNormalizer.clean_numeric(token_data.get("market_cap"), 0)
        features["price_usd"] = DataNormalizer.clean_numeric(token_data.get("price_usd"), 0)
        features["volume_24h"] = DataNormalizer.clean_numeric(token_data.get("volume_24h"), 0)

        # Price changes
        features["price_change_1h"] = DataNormalizer.clean_numeric(
            token_data.get("price_change_1h"), 0
        )
        features["price_change_6h"] = DataNormalizer.clean_numeric(
            token_data.get("price_change_6h"), 0
        )
        features["price_change_24h"] = DataNormalizer.clean_numeric(
            token_data.get("price_change_24h"), 0
        )

        # Volume momentum (volume relative to market cap)
        if features["market_cap"] > 0:
            features["volume_to_mcap_ratio"] = features["volume_24h"] / features["market_cap"]
        else:
            features["volume_to_mcap_ratio"] = 0

        # Transaction counts
        features["txns_24h_buys"] = DataNormalizer.clean_numeric(token_data.get("txns_24h_buys"), 0)
        features["txns_24h_sells"] = DataNormalizer.clean_numeric(
            token_data.get("txns_24h_sells"), 0
        )
        features["txns_24h_total"] = features["txns_24h_buys"] + features["txns_24h_sells"]

        # Buy/sell pressure
        if features["txns_24h_total"] > 0:
            features["buy_sell_ratio"] = features["txns_24h_buys"] / features["txns_24h_total"]
        else:
            features["buy_sell_ratio"] = 0.5  # Neutral

        # Token age (hours since creation)
        if "pair_created_at" in token_data and token_data["pair_created_at"]:
            created_at = DataNormalizer.normalize_timestamp(token_data["pair_created_at"])
            age = (datetime.now(timezone.utc) - created_at).total_seconds() / 3600
            features["token_age_hours"] = age
        else:
            features["token_age_hours"] = 0

        return features

    async def extract_liquidity_features(self, token_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract liquidity-related features.

        Args:
            token_data: Token data

        Returns:
            Dict of liquidity features
        """
        features = {}

        liquidity_usd = DataNormalizer.clean_numeric(token_data.get("liquidity_usd"), 0)
        features["liquidity_usd"] = liquidity_usd

        # Liquidity to market cap ratio
        market_cap = DataNormalizer.clean_numeric(token_data.get("market_cap"), 0)
        if market_cap > 0:
            features["liquidity_to_mcap"] = liquidity_usd / market_cap
        else:
            features["liquidity_to_mcap"] = 0

        # Estimate liquidity depth at different price impacts
        # Simple model: assume linear slippage
        features["liquidity_depth_1pct"] = liquidity_usd * 0.01  # 1% price impact
        features["liquidity_depth_5pct"] = liquidity_usd * 0.05  # 5% price impact
        features["liquidity_depth_10pct"] = liquidity_usd * 0.10  # 10% price impact

        return features

    async def extract_holder_features(
        self, token_address: str, chain: str, holders: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Extract holder distribution features.

        Args:
            token_address: Token contract address
            chain: Blockchain network
            holders: Optional pre-fetched holder list

        Returns:
            Dict of holder distribution features
        """
        features = {}

        # If holders not provided and we have blockchain client, fetch them
        if holders is None and self.blockchain_client:
            try:
                holders = await self.blockchain_client.get_token_holders(token_address)
            except Exception as e:
                logger.warning(f"Could not fetch holders: {e}")
                holders = []

        if not holders:
            # Return zero features if no holder data
            return {
                "total_holders": 0,
                "holder_gini": 0,
                "top10_concentration": 0,
                "top50_concentration": 0,
            }

        # Use normalizer to calculate holder stats
        stats = DataNormalizer.aggregate_holder_stats(holders)

        features["total_holders"] = stats["total_holders"]
        features["holder_gini"] = stats["gini_coefficient"]
        features["top10_concentration"] = stats["top10_percentage"]
        features["top50_concentration"] = stats["top50_percentage"]

        return features

    async def extract_smart_money_features(
        self, transfers: List[Dict[str, Any]], known_smart_wallets: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Extract smart money activity features.

        Args:
            transfers: List of token transfer events
            known_smart_wallets: List of known smart money wallet addresses

        Returns:
            Dict of smart money features
        """
        features = {}

        if not transfers:
            return {
                "smart_money_activity": 0,
                "smart_money_buy_ratio": 0,
                "smart_money_volume": 0,
            }

        smart_wallets = set(known_smart_wallets or [])

        smart_buys = 0
        smart_sells = 0
        smart_volume = 0.0

        for transfer in transfers:
            from_addr = transfer.get("from_address", "").lower()
            to_addr = transfer.get("to_address", "").lower()
            value = DataNormalizer.clean_numeric(transfer.get("value"), 0)

            if from_addr in smart_wallets:
                smart_sells += 1
                smart_volume += value
            if to_addr in smart_wallets:
                smart_buys += 1
                smart_volume += value

        total_smart = smart_buys + smart_sells
        features["smart_money_activity"] = total_smart
        features["smart_money_volume"] = smart_volume

        if total_smart > 0:
            features["smart_money_buy_ratio"] = smart_buys / total_smart
        else:
            features["smart_money_buy_ratio"] = 0

        return features

    async def extract_all_features(
        self, token_address: str, chain: str, token_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Extract all token features.

        Args:
            token_address: Token contract address
            chain: Blockchain network
            token_data: Optional pre-fetched token data

        Returns:
            Complete feature dictionary
        """
        all_features = {}

        # Fetch token data if not provided
        if token_data is None:
            try:
                token_data = await self.dex_client.get_token_data(token_address, chain)
                if not token_data:
                    logger.warning(f"No token data found for {token_address}")
                    return all_features
            except Exception as e:
                logger.error(f"Error fetching token data: {e}")
                return all_features

        # Extract different feature groups
        market_features = await self.extract_market_features(token_data)
        all_features.update(market_features)

        liquidity_features = await self.extract_liquidity_features(token_data)
        all_features.update(liquidity_features)

        # Holder features (if blockchain client available)
        if self.blockchain_client:
            try:
                holder_features = await self.extract_holder_features(token_address, chain)
                all_features.update(holder_features)
            except Exception as e:
                logger.warning(f"Could not extract holder features: {e}")

        return all_features

    async def close(self):
        """Close HTTP clients."""
        if self.dex_client:
            await self.dex_client.close()
        if self.blockchain_client:
            await self.blockchain_client.close()
