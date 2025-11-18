"""
Unit tests for Solana Data Fetcher.
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from radarx.smart_wallet_finder.solana_data_fetcher import SolanaDataFetcher


class TestSolanaDataFetcher:
    """Test Solana-specific data fetching."""

    @pytest.fixture
    def fetcher(self):
        """Create SolanaDataFetcher instance."""
        return SolanaDataFetcher(
            solscan_api_key="test_key",
            helius_api_key="test_helius_key",
        )

    def test_initialization(self, fetcher):
        """Test fetcher initializes correctly."""
        assert fetcher.solscan_api_key == "test_key"
        assert fetcher.helius_api_key == "test_helius_key"
        assert fetcher.solscan_base_url == "https://pro-api.solscan.io/v1.0"
        assert "jupiter" in fetcher.dex_program_ids

    def test_identify_dex(self, fetcher):
        """Test DEX identification."""
        transfer = {"program": "jupiter"}
        result = fetcher._identify_dex(transfer)
        assert isinstance(result, str)

    def test_parse_transfers_to_trades(self, fetcher):
        """Test parsing transfers into trades."""
        now = datetime.utcnow()
        transfers = [
            {
                "transactionHash": "test_tx_1",
                "source": "wallet1",
                "destination": "wallet2",
                "amount": 1000000000,  # 1 token with 9 decimals
                "decimals": 9,
                "blockTime": int(now.timestamp()),
                "changeAmount": 1000000000,
            },
            {
                "transactionHash": "test_tx_1",
                "source": "wallet2",
                "destination": "wallet3",
                "amount": 500000000,
                "decimals": 9,
                "blockTime": int(now.timestamp()),
                "changeAmount": -500000000,
            },
        ]

        trades = fetcher._parse_transfers_to_trades(transfers)

        assert isinstance(trades, list)
        assert len(trades) > 0

        # Check trade structure
        for trade in trades:
            assert "side" in trade
            assert trade["side"] in ["buy", "sell"]
            assert "buyer" in trade
            assert "seller" in trade
            assert "timestamp" in trade
            assert "amount_tokens" in trade

    @pytest.mark.asyncio
    async def test_fetch_jupiter_price(self, fetcher):
        """Test fetching price from Jupiter."""
        mock_response = Mock()
        mock_response.json.return_value = {"data": {"test_token": {"price": 1.23}}}
        mock_response.raise_for_status = Mock()

        with patch.object(fetcher.http_client, "get", return_value=mock_response):
            price = await fetcher._fetch_jupiter_price("test_token")
            assert price == 1.23

    @pytest.mark.asyncio
    async def test_fetch_jupiter_price_error(self, fetcher):
        """Test Jupiter price fetch handles errors."""
        with patch.object(fetcher.http_client, "get", side_effect=Exception("API error")):
            price = await fetcher._fetch_jupiter_price("test_token")
            assert price == 0.0

    @pytest.mark.asyncio
    async def test_fetch_solana_token_metadata_async(self, fetcher):
        """Test fetching Solana token metadata."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "name": "Test Token",
            "symbol": "TEST",
            "decimals": 9,
            "supply": "1000000000",
            "holder": 1000,
        }
        mock_response.raise_for_status = Mock()

        with patch.object(fetcher.http_client, "get", return_value=mock_response):
            metadata = await fetcher._fetch_solana_token_metadata_async("test_address")

            assert metadata["name"] == "Test Token"
            assert metadata["symbol"] == "TEST"
            assert metadata["chain"] == "solana"
            assert metadata["decimals"] == 9

    @pytest.mark.asyncio
    async def test_fetch_solana_token_metadata_no_key(self):
        """Test metadata fetch without API key."""
        fetcher = SolanaDataFetcher(solscan_api_key=None)

        metadata = await fetcher._fetch_solana_token_metadata_async("test_address")

        assert metadata["name"] == "Unknown"
        assert metadata["symbol"] == "UNK"
        assert metadata["chain"] == "solana"

    def test_fetch_dex_trades_non_solana(self, fetcher):
        """Test that non-Solana chains return empty list."""
        now = datetime.utcnow()
        trades = fetcher._fetch_dex_trades(
            token_address="0x123",
            chain="ethereum",
            start_time=now - timedelta(days=1),
            end_time=now,
        )

        assert trades == []

    def test_fetch_price_timeline_non_solana(self, fetcher):
        """Test that non-Solana chains return empty list."""
        now = datetime.utcnow()
        timeline = fetcher._fetch_price_timeline(
            token_address="0x123",
            chain="ethereum",
            start_time=now - timedelta(days=1),
            end_time=now,
        )

        assert timeline == []

    @pytest.mark.asyncio
    async def test_fetch_solscan_transfers(self, fetcher):
        """Test fetching transfers from Solscan."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "data": [
                {
                    "blockTime": int(datetime.utcnow().timestamp()),
                    "source": "wallet1",
                    "destination": "wallet2",
                    "amount": 1000000000,
                    "decimals": 9,
                }
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch.object(fetcher.http_client, "get", return_value=mock_response):
            transfers = await fetcher._fetch_solscan_transfers(
                token_address="test_token",
                start_time=datetime.utcnow() - timedelta(days=1),
                end_time=datetime.utcnow(),
            )

            assert isinstance(transfers, list)
            assert len(transfers) > 0

    @pytest.mark.asyncio
    async def test_fetch_solscan_transfers_no_key(self):
        """Test Solscan fetch without API key."""
        fetcher = SolanaDataFetcher(solscan_api_key=None)

        transfers = await fetcher._fetch_solscan_transfers(
            token_address="test_token",
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow(),
        )

        assert transfers == []

    @pytest.mark.asyncio
    async def test_close(self, fetcher):
        """Test closing HTTP clients."""
        await fetcher.close()
        # Should not raise an error


class TestDataFetcherFactory:
    """Test DataFetcher factory method."""

    def test_create_for_solana(self):
        """Test creating Solana data fetcher."""
        from radarx.smart_wallet_finder.data_fetcher import DataFetcher

        fetcher = DataFetcher.create_for_chain("solana")

        assert fetcher.__class__.__name__ == "SolanaDataFetcher"

    def test_create_for_ethereum(self):
        """Test creating Ethereum data fetcher."""
        from radarx.smart_wallet_finder.data_fetcher import DataFetcher

        fetcher = DataFetcher.create_for_chain("ethereum")

        # Should return default DataFetcher for now
        assert fetcher.__class__.__name__ == "DataFetcher"

    def test_create_for_bsc(self):
        """Test creating BSC data fetcher."""
        from radarx.smart_wallet_finder.data_fetcher import DataFetcher

        fetcher = DataFetcher.create_for_chain("bsc")

        # Should return default DataFetcher for now
        assert fetcher.__class__.__name__ == "DataFetcher"

    def test_create_for_unknown_chain(self):
        """Test creating fetcher for unknown chain."""
        from radarx.smart_wallet_finder.data_fetcher import DataFetcher

        fetcher = DataFetcher.create_for_chain("unknown_chain")

        # Should return default DataFetcher with warning
        assert fetcher.__class__.__name__ == "DataFetcher"
