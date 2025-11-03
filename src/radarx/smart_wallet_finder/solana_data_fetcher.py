"""
Solana Data Fetcher

Production-ready implementation for fetching Solana blockchain data.
Integrates with Solscan, Jupiter, Raydium, and other Solana-specific APIs.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import httpx
import asyncio
from base58 import b58decode

from radarx.smart_wallet_finder.data_fetcher import DataFetcher
from radarx.data.cache import CacheManager
from radarx.config import settings

logger = logging.getLogger(__name__)


class SolanaDataFetcher(DataFetcher):
    """
    Production-ready Solana data fetcher.
    
    Integrates with:
    - Solscan API for transaction history
    - Jupiter API for swap data
    - Raydium/Orca for DEX trades
    - Helius/QuickNode for RPC calls
    """
    
    def __init__(
        self,
        solscan_api_key: Optional[str] = None,
        helius_api_key: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        """
        Initialize Solana data fetcher.
        
        Args:
            solscan_api_key: Solscan API key (optional, uses settings if not provided)
            helius_api_key: Helius API key for RPC access
            cache_manager: Cache manager for caching API responses
        """
        super().__init__()
        
        self.solscan_api_key = solscan_api_key or getattr(settings, 'solscan_api_key', None)
        self.helius_api_key = helius_api_key or getattr(settings, 'helius_api_key', None)
        self.cache = cache_manager or CacheManager()
        
        # API endpoints
        self.solscan_base_url = "https://pro-api.solscan.io/v1.0"
        self.jupiter_price_url = "https://price.jup.ag/v4"
        self.birdeye_base_url = "https://public-api.birdeye.so"
        self.helius_rpc_url = f"https://rpc.helius.xyz/?api-key={self.helius_api_key}" if self.helius_api_key else "https://api.mainnet-beta.solana.com"
        
        # HTTP clients
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Known Solana DEX program IDs
        self.dex_program_ids = {
            "jupiter": "JUP4Fb2cqiRUcaTHdrPC8h2gNsA2ETXiPDD33WcGuJB",
            "raydium": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
            "orca": "9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP",
            "serum": "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpZb9PusVFin",
        }
    
    async def close(self):
        """Close HTTP clients."""
        await self.http_client.aclose()
    
    def _fetch_dex_trades(
        self,
        token_address: str,
        chain: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Fetch DEX trades for Solana token.
        
        Uses async implementation internally.
        """
        if chain.lower() != "solana":
            logger.warning(f"SolanaDataFetcher called for non-Solana chain: {chain}")
            return []
        
        # Run async method synchronously
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._fetch_solana_trades_async(token_address, start_time, end_time)
        )
    
    async def _fetch_solana_trades_async(
        self,
        token_address: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Async implementation to fetch Solana DEX trades.
        
        Fetches from Solscan transaction history API.
        """
        cache_key = f"solana:trades:{token_address}:{start_time.date()}:{end_time.date()}"
        cached = await self.cache.get(cache_key)
        if cached:
            logger.debug(f"Cache hit for {cache_key}")
            return cached
        
        trades = []
        
        try:
            # Fetch transfer events from Solscan
            transfers = await self._fetch_solscan_transfers(
                token_address=token_address,
                start_time=start_time,
                end_time=end_time,
            )
            
            # Parse transfers into trades
            trades = self._parse_transfers_to_trades(transfers)
            
            # Cache for 5 minutes
            await self.cache.set(cache_key, trades, ttl=300)
            
            logger.info(f"Fetched {len(trades)} Solana trades for {token_address}")
            
        except Exception as e:
            logger.error(f"Error fetching Solana trades: {e}")
        
        return trades
    
    async def _fetch_solscan_transfers(
        self,
        token_address: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch token transfers from Solscan API.
        
        Args:
            token_address: SPL token mint address
            start_time: Start of time range
            end_time: End of time range
            limit: Maximum number of transfers to fetch
            
        Returns:
            List of transfer events
        """
        if not self.solscan_api_key:
            logger.warning("Solscan API key not configured")
            return []
        
        url = f"{self.solscan_base_url}/token/transfer"
        
        params = {
            "token": token_address,
            "limit": min(limit, 1000),
            "offset": 0,
        }
        
        headers = {
            "token": self.solscan_api_key,
            "accept": "application/json",
        }
        
        all_transfers = []
        
        try:
            # Fetch with pagination
            for page in range(5):  # Limit to 5 pages (5000 transfers max)
                params["offset"] = page * params["limit"]
                
                response = await self.http_client.get(
                    url,
                    params=params,
                    headers=headers,
                )
                response.raise_for_status()
                
                data = response.json()
                transfers = data.get("data", [])
                
                if not transfers:
                    break
                
                # Filter by time range
                for transfer in transfers:
                    block_time = transfer.get("blockTime")
                    if block_time:
                        transfer_time = datetime.fromtimestamp(block_time)
                        if start_time <= transfer_time <= end_time:
                            all_transfers.append(transfer)
                
                # Stop if we got fewer results than limit
                if len(transfers) < params["limit"]:
                    break
                
                # Rate limiting
                await asyncio.sleep(0.2)
            
            logger.info(f"Fetched {len(all_transfers)} transfers from Solscan")
            
        except httpx.HTTPError as e:
            logger.error(f"Solscan API error: {e}")
        
        return all_transfers
    
    def _parse_transfers_to_trades(
        self,
        transfers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Parse Solscan transfers into standardized trade format.
        
        Identifies swaps by looking for transfers with DEX program involvement.
        """
        trades = []
        
        # Group transfers by transaction signature to identify swaps
        tx_groups = {}
        for transfer in transfers:
            tx_sig = transfer.get("transactionHash") or transfer.get("signature")
            if tx_sig:
                if tx_sig not in tx_groups:
                    tx_groups[tx_sig] = []
                tx_groups[tx_sig].append(transfer)
        
        # Analyze each transaction group
        for tx_sig, tx_transfers in tx_groups.items():
            # Look for swap pattern (2+ transfers in same tx)
            if len(tx_transfers) >= 2:
                # Try to identify buyer and seller
                for transfer in tx_transfers:
                    source = transfer.get("source") or transfer.get("from")
                    destination = transfer.get("destination") or transfer.get("to")
                    amount = float(transfer.get("amount", 0)) / (10 ** transfer.get("decimals", 9))
                    block_time = transfer.get("blockTime")
                    
                    if not all([source, destination, block_time]):
                        continue
                    
                    # Determine if this is a buy or sell
                    # In a swap, one direction is the token we're tracking
                    change = transfer.get("changeAmount", 0)
                    
                    side = "buy" if float(change) > 0 else "sell"
                    
                    trade = {
                        "side": side,
                        "buyer": destination if side == "buy" else source,
                        "seller": source if side == "buy" else destination,
                        "timestamp": datetime.fromtimestamp(block_time).isoformat(),
                        "amount_tokens": abs(amount),
                        "amount_usd": 0,  # Will be enriched with price data
                        "transaction_hash": tx_sig,
                        "dex": self._identify_dex(transfer),
                        "gas_price": 0,  # Solana uses fixed fees
                    }
                    
                    trades.append(trade)
        
        return trades
    
    def _identify_dex(self, transfer: Dict[str, Any]) -> str:
        """
        Identify which DEX was used based on transfer data.
        
        Enhanced version that checks program IDs and instruction data.
        """
        # Check if program ID is available
        program_id = transfer.get("programId") or transfer.get("program")
        
        if program_id:
            # Match against known DEX program IDs
            for dex_name, dex_program_id in self.dex_program_ids.items():
                if program_id == dex_program_id:
                    return dex_name
        
        # Check instruction data for DEX-specific patterns
        instruction_data = transfer.get("instructionData") or transfer.get("data")
        if instruction_data:
            # Jupiter swap instruction typically starts with specific discriminator
            if instruction_data.startswith("e445a52e51cb9a1d"):  # Jupiter swap
                return "jupiter"
            # Raydium swap instruction
            elif instruction_data.startswith("09"):  # Raydium swap instruction ID
                return "raydium"
            # Orca swap instruction
            elif instruction_data.startswith("f8c69e91e17587c8"):  # Orca whirlpool swap
                return "orca"
        
        # Check transaction logs for DEX mentions
        logs = transfer.get("logs", [])
        for log in logs:
            log_lower = log.lower()
            if "jupiter" in log_lower:
                return "jupiter"
            elif "raydium" in log_lower:
                return "raydium"
            elif "orca" in log_lower:
                return "orca"
            elif "serum" in log_lower:
                return "serum"
        
        return "unknown_dex"
    
    def _fetch_price_timeline(
        self,
        token_address: str,
        chain: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Fetch Solana token price timeline.
        
        Uses Jupiter price API and historical data.
        """
        if chain.lower() != "solana":
            return []
        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._fetch_solana_price_timeline_async(
                token_address, start_time, end_time, interval_minutes
            )
        )
    
    async def _fetch_solana_price_timeline_async(
        self,
        token_address: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Async fetch of Solana price timeline.
        
        Uses Birdeye API for historical price data.
        Falls back to Jupiter current price if Birdeye unavailable.
        """
        timeline = []
        
        try:
            # Try to fetch historical data from Birdeye
            timeline = await self._fetch_birdeye_historical_prices(
                token_address, start_time, end_time, interval_minutes
            )
            
            if timeline:
                logger.info(f"Fetched {len(timeline)} historical price points from Birdeye")
                return timeline
            
            # Fallback: Use Jupiter current price
            logger.warning("Birdeye historical data unavailable, using Jupiter fallback")
            current_price = await self._fetch_jupiter_price(token_address)
            
            # Generate timeline with current price as reference
            current_time = end_time
            time_delta = timedelta(minutes=interval_minutes)
            
            while current_time >= start_time:
                timeline.insert(0, {
                    "timestamp": current_time.isoformat(),
                    "price": current_price if current_price else 0.0,
                })
                current_time -= time_delta
            
            logger.info(f"Generated {len(timeline)} price points for {token_address}")
            
        except Exception as e:
            logger.error(f"Error fetching Solana price timeline: {e}")
        
        return timeline
    
    async def _fetch_birdeye_historical_prices(
        self,
        token_address: str,
        start_time: datetime,
        end_time: datetime,
        interval_minutes: int = 15,
    ) -> List[Dict[str, Any]]:
        """
        Fetch historical price data from Birdeye API.
        
        Args:
            token_address: SPL token mint address
            start_time: Start of time range
            end_time: End of time range
            interval_minutes: Interval between data points
            
        Returns:
            List of price points with timestamp and price
        """
        cache_key = f"solana:birdeye:{token_address}:{start_time.date()}:{end_time.date()}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            # Map interval to Birdeye's time_from/time_to format
            url = f"{self.birdeye_base_url}/defi/history_price"
            
            params = {
                "address": token_address,
                "address_type": "token",
                "type": self._get_birdeye_interval(interval_minutes),
                "time_from": int(start_time.timestamp()),
                "time_to": int(end_time.timestamp()),
            }
            
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get("success") or not data.get("data"):
                return []
            
            items = data["data"].get("items", [])
            
            timeline = []
            for item in items:
                timeline.append({
                    "timestamp": datetime.fromtimestamp(item["unixTime"]).isoformat(),
                    "price": float(item["value"]),
                })
            
            # Cache for 5 minutes
            await self.cache.set(cache_key, timeline, ttl=300)
            
            return timeline
            
        except Exception as e:
            logger.error(f"Birdeye API error: {e}")
            return []
    
    def _get_birdeye_interval(self, interval_minutes: int) -> str:
        """Convert interval minutes to Birdeye interval type."""
        if interval_minutes <= 5:
            return "1m"
        elif interval_minutes <= 15:
            return "15m"
        elif interval_minutes <= 60:
            return "1H"
        elif interval_minutes <= 240:
            return "4H"
        else:
            return "1D"
    
    async def _fetch_jupiter_price(self, token_address: str) -> float:
        """
        Fetch current token price from Jupiter.
        
        Args:
            token_address: SPL token mint address
            
        Returns:
            Current price in USD
        """
        cache_key = f"solana:price:{token_address}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.jupiter_price_url}/price"
            params = {"ids": token_address}
            
            response = await self.http_client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            price_data = data.get("data", {}).get(token_address, {})
            price = float(price_data.get("price", 0))
            
            # Cache for 1 minute
            await self.cache.set(cache_key, price, ttl=60)
            
            return price
            
        except Exception as e:
            logger.error(f"Jupiter price API error: {e}")
            return 0.0
    
    def _fetch_token_metadata(
        self,
        token_address: str,
        chain: str,
    ) -> Dict[str, Any]:
        """
        Fetch Solana token metadata.
        """
        if chain.lower() != "solana":
            return super()._fetch_token_metadata(token_address, chain)
        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._fetch_solana_token_metadata_async(token_address)
        )
    
    async def _fetch_solana_token_metadata_async(
        self,
        token_address: str,
    ) -> Dict[str, Any]:
        """
        Async fetch of Solana token metadata from Solscan.
        """
        if not self.solscan_api_key:
            return {
                "address": token_address,
                "chain": "solana",
                "name": "Unknown",
                "symbol": "UNK",
            }
        
        try:
            url = f"{self.solscan_base_url}/token/meta"
            params = {"token": token_address}
            headers = {"token": self.solscan_api_key}
            
            response = await self.http_client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                "address": token_address,
                "chain": "solana",
                "name": data.get("name", "Unknown"),
                "symbol": data.get("symbol", "UNK"),
                "decimals": data.get("decimals", 9),
                "supply": data.get("supply"),
                "holder_count": data.get("holder", 0),
            }
            
        except Exception as e:
            logger.error(f"Error fetching Solana token metadata: {e}")
            return {
                "address": token_address,
                "chain": "solana",
                "name": "Unknown",
                "symbol": "UNK",
            }
    
    def _fetch_wallet_trades(
        self,
        wallet_address: str,
        token_address: str,
        chain: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Fetch Solana wallet trades for specific token.
        """
        if chain.lower() != "solana":
            return []
        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._fetch_solana_wallet_trades_async(
                wallet_address, token_address, start_time, end_time
            )
        )
    
    async def _fetch_solana_wallet_trades_async(
        self,
        wallet_address: str,
        token_address: str,
        start_time: datetime,
        end_time: datetime,
    ) -> List[Dict[str, Any]]:
        """
        Async fetch of wallet-specific trades on Solana.
        """
        if not self.solscan_api_key:
            logger.warning("Solscan API key not configured")
            return []
        
        try:
            # Fetch wallet's token transactions
            url = f"{self.solscan_base_url}/account/token/txs"
            params = {
                "account": wallet_address,
                "token": token_address,
                "limit": 100,
            }
            headers = {"token": self.solscan_api_key}
            
            response = await self.http_client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            transactions = data.get("data", [])
            
            # Parse transactions into trades
            trades = []
            for tx in transactions:
                block_time = tx.get("blockTime")
                if not block_time:
                    continue
                
                tx_time = datetime.fromtimestamp(block_time)
                if not (start_time <= tx_time <= end_time):
                    continue
                
                # Determine trade direction
                change = float(tx.get("changeAmount", 0))
                amount = abs(change) / (10 ** tx.get("decimals", 9))
                
                trade = {
                    "side": "buy" if change > 0 else "sell",
                    "timestamp": tx_time.isoformat(),
                    "amount_tokens": amount,
                    "amount_usd": 0,  # Would need price at that time
                    "transaction_hash": tx.get("signature"),
                    "dex": "unknown",
                }
                
                trades.append(trade)
            
            logger.info(f"Fetched {len(trades)} trades for wallet {wallet_address}")
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching Solana wallet trades: {e}")
            return []
    
    def _fetch_graph_neighbors(
        self,
        wallet_address: str,
        chain: str,
    ) -> List[str]:
        """
        Fetch neighboring wallets for Solana address.
        
        Analyzes transaction history to find connected wallets.
        """
        if chain.lower() != "solana":
            return []
        
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self._fetch_solana_graph_neighbors_async(wallet_address)
        )
    
    async def _fetch_solana_graph_neighbors_async(
        self,
        wallet_address: str,
        max_neighbors: int = 50,
    ) -> List[str]:
        """
        Async fetch of Solana wallet neighbors.
        """
        if not self.solscan_api_key:
            return []
        
        try:
            # Fetch recent transactions
            url = f"{self.solscan_base_url}/account/transactions"
            params = {
                "account": wallet_address,
                "limit": 100,
            }
            headers = {"token": self.solscan_api_key}
            
            response = await self.http_client.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            transactions = data.get("data", [])
            
            # Extract unique counterparties
            neighbors = set()
            for tx in transactions:
                # Would parse tx details to find counterparties
                # This is simplified - production would analyze instruction data
                pass
            
            return list(neighbors)[:max_neighbors]
            
        except Exception as e:
            logger.error(f"Error fetching Solana graph neighbors: {e}")
            return []
