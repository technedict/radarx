"""Blockchain indexer clients for on-chain data."""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from radarx.config import settings
from radarx.data.cache import CacheManager

logger = logging.getLogger(__name__)


class BlockchainIndexer(ABC):
    """Abstract base class for blockchain indexers."""
    
    def __init__(self, api_key: str, base_url: str, cache_manager: Optional[CacheManager] = None):
        """Initialize blockchain indexer.
        
        Args:
            api_key: API key for the indexer service
            base_url: Base URL for API requests
            cache_manager: Optional cache manager
        """
        self.api_key = api_key
        self.base_url = base_url
        self.cache = cache_manager or CacheManager()
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "RadarX/0.1.0"}
        )
    
    @abstractmethod
    async def get_token_transfers(
        self,
        contract_address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get token transfer events."""
        pass
    
    @abstractmethod
    async def get_token_holders(
        self,
        contract_address: str,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of token holders."""
        pass
    
    @abstractmethod
    async def get_contract_info(
        self,
        contract_address: str
    ) -> Dict[str, Any]:
        """Get contract information."""
        pass
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class EtherscanClient(BlockchainIndexer):
    """Etherscan API client for Ethereum blockchain data."""
    
    def __init__(self, api_key: Optional[str] = None, cache_manager: Optional[CacheManager] = None):
        """Initialize Etherscan client.
        
        Args:
            api_key: Etherscan API key (defaults to settings.etherscan_api_key)
            cache_manager: Optional cache manager
        """
        api_key = api_key or settings.etherscan_api_key
        super().__init__(
            api_key=api_key,
            base_url="https://api.etherscan.io/api",
            cache_manager=cache_manager
        )
    
    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to Etherscan.
        
        Args:
            params: Query parameters
            
        Returns:
            API response data
        """
        params["apikey"] = self.api_key
        
        try:
            response = await self.client.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") != "1" and data.get("message") != "OK":
                logger.warning(f"Etherscan API warning: {data.get('message')}")
            
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"Etherscan API error: {e}")
            raise
    
    async def get_token_transfers(
        self,
        contract_address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get ERC-20 token transfer events.
        
        Args:
            contract_address: Token contract address
            start_block: Starting block number
            end_block: Ending block number
            page: Page number
            offset: Number of results per page
            
        Returns:
            List of transfer events
        """
        cache_key = f"eth:transfers:{contract_address}:{start_block}:{end_block}:{page}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract_address,
            "page": page,
            "offset": offset,
            "sort": "desc",
        }
        
        if start_block:
            params["startblock"] = start_block
        if end_block:
            params["endblock"] = end_block
        
        data = await self._make_request(params)
        result = data.get("result", [])
        
        # Cache for 2 minutes
        await self.cache.set(cache_key, result, ttl=120)
        
        return result if isinstance(result, list) else []
    
    async def get_token_holders(
        self,
        contract_address: str,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get list of token holders.
        
        Note: Etherscan doesn't provide a direct holder list endpoint.
        This would need to be derived from transfer events.
        
        Args:
            contract_address: Token contract address
            page: Page number
            offset: Number of results
            
        Returns:
            Empty list (not directly supported)
        """
        logger.warning("Etherscan doesn't provide direct holder list endpoint")
        return []
    
    async def get_contract_info(
        self,
        contract_address: str
    ) -> Dict[str, Any]:
        """Get contract source code and ABI.
        
        Args:
            contract_address: Contract address
            
        Returns:
            Contract information including source code and ABI
        """
        cache_key = f"eth:contract:{contract_address}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        params = {
            "module": "contract",
            "action": "getsourcecode",
            "address": contract_address,
        }
        
        data = await self._make_request(params)
        result = data.get("result", [])
        
        if result and isinstance(result, list) and len(result) > 0:
            contract_data = result[0]
            # Cache for 1 hour (contracts don't change)
            await self.cache.set(cache_key, contract_data, ttl=3600)
            return contract_data
        
        return {}
    
    async def get_normal_transactions(
        self,
        address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get normal transactions for an address.
        
        Args:
            address: Wallet address
            start_block: Starting block
            end_block: Ending block
            page: Page number
            offset: Results per page
            
        Returns:
            List of transactions
        """
        params = {
            "module": "account",
            "action": "txlist",
            "address": address,
            "page": page,
            "offset": offset,
            "sort": "desc",
        }
        
        if start_block:
            params["startblock"] = start_block
        if end_block:
            params["endblock"] = end_block
        
        data = await self._make_request(params)
        return data.get("result", []) if isinstance(data.get("result"), list) else []


class BscScanClient(BlockchainIndexer):
    """BscScan API client for Binance Smart Chain data."""
    
    def __init__(self, api_key: Optional[str] = None, cache_manager: Optional[CacheManager] = None):
        """Initialize BscScan client.
        
        Args:
            api_key: BscScan API key
            cache_manager: Optional cache manager
        """
        api_key = api_key or settings.bscscan_api_key
        super().__init__(
            api_key=api_key,
            base_url="https://api.bscscan.com/api",
            cache_manager=cache_manager
        )
    
    async def _make_request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request to BscScan."""
        params["apikey"] = self.api_key
        
        try:
            response = await self.client.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"BscScan API error: {e}")
            raise
    
    async def get_token_transfers(
        self,
        contract_address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get BEP-20 token transfers."""
        params = {
            "module": "account",
            "action": "tokentx",
            "contractaddress": contract_address,
            "page": page,
            "offset": offset,
            "sort": "desc",
        }
        
        if start_block:
            params["startblock"] = start_block
        if end_block:
            params["endblock"] = end_block
        
        data = await self._make_request(params)
        return data.get("result", []) if isinstance(data.get("result"), list) else []
    
    async def get_token_holders(
        self,
        contract_address: str,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get token holders (not directly supported)."""
        logger.warning("BscScan doesn't provide direct holder list endpoint")
        return []
    
    async def get_contract_info(
        self,
        contract_address: str
    ) -> Dict[str, Any]:
        """Get contract information."""
        cache_key = f"bsc:contract:{contract_address}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        params = {
            "module": "contract",
            "action": "getsourcecode",
            "address": contract_address,
        }
        
        data = await self._make_request(params)
        result = data.get("result", [])
        
        if result and isinstance(result, list) and len(result) > 0:
            contract_data = result[0]
            await self.cache.set(cache_key, contract_data, ttl=3600)
            return contract_data
        
        return {}


class SolscanClient(BlockchainIndexer):
    """Solscan API client for Solana blockchain data."""
    
    def __init__(self, api_key: Optional[str] = None, cache_manager: Optional[CacheManager] = None):
        """Initialize Solscan client.
        
        Args:
            api_key: Solscan API key
            cache_manager: Optional cache manager
        """
        api_key = api_key or settings.solscan_api_key
        super().__init__(
            api_key=api_key,
            base_url="https://public-api.solscan.io",
            cache_manager=cache_manager
        )
    
    async def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make API request to Solscan."""
        headers = {"token": self.api_key} if self.api_key else {}
        
        try:
            response = await self.client.get(
                f"{self.base_url}/{endpoint}",
                params=params,
                headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"Solscan API error: {e}")
            raise
    
    async def get_token_transfers(
        self,
        contract_address: str,
        start_block: Optional[int] = None,
        end_block: Optional[int] = None,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get SPL token transfers."""
        data = await self._make_request(
            f"token/transfer",
            params={
                "token": contract_address,
                "limit": offset,
                "offset": (page - 1) * offset
            }
        )
        return data.get("data", [])
    
    async def get_token_holders(
        self,
        contract_address: str,
        page: int = 1,
        offset: int = 100
    ) -> List[Dict[str, Any]]:
        """Get SPL token holders."""
        cache_key = f"sol:holders:{contract_address}:{page}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        data = await self._make_request(
            f"token/holders",
            params={
                "tokenAddress": contract_address,
                "limit": offset,
                "offset": (page - 1) * offset
            }
        )
        
        holders = data.get("data", [])
        # Cache for 5 minutes
        await self.cache.set(cache_key, holders, ttl=300)
        return holders
    
    async def get_contract_info(
        self,
        contract_address: str
    ) -> Dict[str, Any]:
        """Get token metadata."""
        cache_key = f"sol:token:{contract_address}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        data = await self._make_request(f"token/meta", params={"tokenAddress": contract_address})
        
        # Cache for 1 hour
        await self.cache.set(cache_key, data, ttl=3600)
        return data
