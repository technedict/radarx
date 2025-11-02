"""Unit tests for data ingestion clients."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from radarx.data.dexscreener import DexScreenerClient
from radarx.data.blockchain import EtherscanClient, BscScanClient, SolscanClient
from radarx.data.cache import CacheManager
from radarx.data.normalizer import DataNormalizer


class TestCacheManager:
    """Tests for CacheManager."""
    
    @pytest.mark.asyncio
    async def test_cache_set_and_get(self):
        """Test setting and getting cache values."""
        cache = CacheManager()
        
        await cache.set("test_key", {"data": "value"}, ttl=60)
        result = await cache.get("test_key")
        
        assert result == {"data": "value"}
    
    @pytest.mark.asyncio
    async def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = CacheManager()
        result = await cache.get("nonexistent_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_delete(self):
        """Test deleting cache entry."""
        cache = CacheManager()
        
        await cache.set("test_key", "value")
        await cache.delete("test_key")
        result = await cache.get("test_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """Test clearing all cache."""
        cache = CacheManager()
        
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.clear()
        
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = CacheManager()
        stats = cache.get_stats()
        
        assert "total_keys" in stats
        assert "active_keys" in stats


class TestDataNormalizer:
    """Tests for DataNormalizer."""
    
    def test_normalize_address_ethereum(self):
        """Test normalizing Ethereum address."""
        address = "0xABCDEF1234567890ABCDEF1234567890ABCDEF12"
        normalized = DataNormalizer.normalize_address(address, "ethereum")
        
        assert normalized == "0xabcdef1234567890abcdef1234567890abcdef12"
        assert normalized.startswith("0x")
    
    def test_normalize_address_without_0x(self):
        """Test normalizing address without 0x prefix."""
        address = "ABCDEF1234567890ABCDEF1234567890ABCDEF12"
        normalized = DataNormalizer.normalize_address(address, "ethereum")
        
        assert normalized.startswith("0x")
        assert len(normalized) == 42
    
    def test_normalize_timestamp_datetime(self):
        """Test normalizing datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 0)
        normalized = DataNormalizer.normalize_timestamp(dt)
        
        assert normalized == dt
    
    def test_normalize_timestamp_unix(self):
        """Test normalizing Unix timestamp."""
        unix_ts = 1705318200  # 2024-01-15 10:30:00
        normalized = DataNormalizer.normalize_timestamp(unix_ts)
        
        assert isinstance(normalized, datetime)
    
    def test_normalize_timestamp_milliseconds(self):
        """Test normalizing Unix timestamp in milliseconds."""
        unix_ms = 1705318200000
        normalized = DataNormalizer.normalize_timestamp(unix_ms)
        
        assert isinstance(normalized, datetime)
    
    def test_normalize_chain_name(self):
        """Test normalizing chain names."""
        assert DataNormalizer.normalize_chain_name("ETH") == "ethereum"
        assert DataNormalizer.normalize_chain_name("BSC") == "bsc"
        assert DataNormalizer.normalize_chain_name("BNB") == "bsc"
        assert DataNormalizer.normalize_chain_name("SOL") == "solana"
        assert DataNormalizer.normalize_chain_name("MATIC") == "polygon"
    
    def test_validate_wallet_address_ethereum(self):
        """Test validating Ethereum wallet address."""
        valid = "0xabcdef1234567890abcdef1234567890abcdef12"
        invalid = "0xinvalid"
        
        assert DataNormalizer.validate_wallet_address(valid, "ethereum")
        assert not DataNormalizer.validate_wallet_address(invalid, "ethereum")
    
    def test_clean_numeric(self):
        """Test cleaning numeric values."""
        assert DataNormalizer.clean_numeric("1,234.56") == 1234.56
        assert DataNormalizer.clean_numeric("$100") == 100.0
        assert DataNormalizer.clean_numeric(None, default=0) == 0
        assert DataNormalizer.clean_numeric("invalid", default=10) == 10
    
    def test_aggregate_holder_stats(self):
        """Test holder statistics aggregation."""
        holders = [
            {"address": "0x1", "balance": "1000"},
            {"address": "0x2", "balance": "500"},
            {"address": "0x3", "balance": "250"},
        ]
        
        stats = DataNormalizer.aggregate_holder_stats(holders)
        
        assert stats["total_holders"] == 3
        assert "top10_percentage" in stats
        assert "gini_coefficient" in stats


@pytest.mark.asyncio
class TestDexScreenerClient:
    """Tests for DexScreenerClient."""
    
    async def test_get_token_pairs_cached(self):
        """Test getting token pairs with cache."""
        cache = CacheManager()
        client = DexScreenerClient(cache_manager=cache)
        
        # Mock cached data
        await cache.set("dex:pairs:ethereum:0xtest", [{"pairAddress": "0xpair"}])
        
        result = await client.get_token_pairs("0xtest", "ethereum")
        
        assert len(result) == 1
        assert result[0]["pairAddress"] == "0xpair"
    
    @patch("httpx.AsyncClient.get")
    async def test_get_token_data(self, mock_get):
        """Test getting token data."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "pairs": [{
                "baseToken": {"symbol": "TEST", "name": "Test Token"},
                "priceUsd": "1.5",
                "volume": {"h24": "1000000"},
                "liquidity": {"usd": "500000"},
                "priceChange": {"h24": "10.5"},
            }]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        client = DexScreenerClient()
        result = await client.get_token_data("0xtest", "ethereum")
        
        assert result is not None
        assert result["symbol"] == "TEST"
        assert result["price_usd"] == 1.5
        
        await client.close()


@pytest.mark.asyncio
class TestBlockchainIndexers:
    """Tests for blockchain indexer clients."""
    
    async def test_etherscan_normalize_address(self):
        """Test Etherscan client."""
        client = EtherscanClient(api_key="test_key")
        assert client.api_key == "test_key"
        await client.close()
    
    async def test_bscscan_client(self):
        """Test BscScan client initialization."""
        client = BscScanClient(api_key="test_key")
        assert client.base_url == "https://api.bscscan.com/api"
        await client.close()
    
    async def test_solscan_client(self):
        """Test Solscan client initialization."""
        client = SolscanClient(api_key="test_key")
        assert client.base_url == "https://public-api.solscan.io"
        await client.close()


def test_validate_token_data():
    """Test token data validation."""
    valid_data = {
        "address": "0x1234567890abcdef1234567890abcdef12345678",
        "chain": "ethereum",
    }
    
    invalid_data = {
        "address": "",  # Missing address
        "chain": "ethereum",
    }
    
    assert DataNormalizer.validate_token_data(valid_data)
    assert not DataNormalizer.validate_token_data(invalid_data)
