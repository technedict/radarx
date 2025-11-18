"""Risk assessment feed clients (RugCheck, GoPlus)."""

import logging
from typing import Any, Dict, List, Optional

import httpx

from radarx.config import settings
from radarx.data.cache import CacheManager

logger = logging.getLogger(__name__)


class RugCheckClient:
    """RugCheck API client for Solana token risk assessment."""
    
    BASE_URL = "https://api.rugcheck.xyz/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """Initialize RugCheck client.
        
        Args:
            api_key: RugCheck API key (if required)
            cache_manager: Optional cache manager
        """
        self.api_key = api_key or settings.rugcheck_api_key
        self.cache = cache_manager or CacheManager()
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "RadarX/0.1.0"}
        )
    
    async def get_token_report(
        self,
        token_address: str
    ) -> Dict[str, Any]:
        """Get comprehensive risk report for a Solana token.
        
        Args:
            token_address: Solana token mint address
            
        Returns:
            Risk report with scores and flags
        """
        cache_key = f"rugcheck:{token_address}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/tokens/{token_address}/report"
            logger.info(f"Fetching RugCheck report for {token_address}")
            
            response = await self.client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache for 10 minutes
            await self.cache.set(cache_key, data, ttl=600)
            
            return data
            
        except httpx.HTTPError as e:
            logger.error(f"RugCheck API error: {e}")
            return {}
    
    async def get_risk_score(
        self,
        token_address: str
    ) -> Dict[str, Any]:
        """Get simplified risk score for a token.
        
        Args:
            token_address: Token address
            
        Returns:
            Dict with risk score and main flags
        """
        report = await self.get_token_report(token_address)
        
        if not report:
            return {
                "risk_level": "unknown",
                "risk_score": 50,  # Neutral score
                "flags": ["no_data"],
            }
        
        # Extract key risk indicators
        risks = report.get("risks", [])
        risk_level = report.get("riskLevel", "unknown")
        score = report.get("score", 50)
        
        return {
            "risk_level": risk_level,
            "risk_score": score,
            "flags": [r.get("name") for r in risks if isinstance(r, dict)],
            "description": report.get("description", ""),
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class GoPlusClient:
    """GoPlus Security API client for multi-chain token security."""
    
    BASE_URL = "https://api.gopluslabs.io/api/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_manager: Optional[CacheManager] = None
    ):
        """Initialize GoPlus client.
        
        Args:
            api_key: GoPlus API key (if required)
            cache_manager: Optional cache manager
        """
        self.api_key = api_key or settings.goplus_api_key
        self.cache = cache_manager or CacheManager()
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": "RadarX/0.1.0"}
        )
    
    async def get_token_security(
        self,
        token_address: str,
        chain_id: str = "1"  # 1=Ethereum, 56=BSC, etc.
    ) -> Dict[str, Any]:
        """Get token security information.
        
        Args:
            token_address: Token contract address
            chain_id: Chain ID (1=Ethereum, 56=BSC, 137=Polygon, etc.)
            
        Returns:
            Security information and risk flags
        """
        cache_key = f"goplus:{chain_id}:{token_address}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
        
        try:
            url = f"{self.BASE_URL}/token_security/{chain_id}"
            params = {"contract_addresses": token_address}
            
            logger.info(f"Fetching GoPlus security for {token_address} on chain {chain_id}")
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            result = data.get("result", {}).get(token_address.lower(), {})
            
            # Cache for 10 minutes
            await self.cache.set(cache_key, result, ttl=600)
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"GoPlus API error: {e}")
            return {}
    
    async def get_address_security(
        self,
        address: str,
        chain_id: str = "1"
    ) -> Dict[str, Any]:
        """Get security information for a wallet address.
        
        Args:
            address: Wallet address
            chain_id: Chain ID
            
        Returns:
            Address security information
        """
        try:
            url = f"{self.BASE_URL}/address_security/{address}"
            params = {"chain_id": chain_id}
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("result", {})
            
        except httpx.HTTPError as e:
            logger.error(f"GoPlus address security error: {e}")
            return {}
    
    async def get_risk_assessment(
        self,
        token_address: str,
        chain_id: str = "1"
    ) -> Dict[str, Any]:
        """Get simplified risk assessment.
        
        Args:
            token_address: Token address
            chain_id: Chain ID
            
        Returns:
            Risk assessment with score and flags
        """
        security = await self.get_token_security(token_address, chain_id)
        
        if not security:
            return {
                "risk_level": "unknown",
                "risk_score": 50,
                "flags": ["no_data"],
            }
        
        # Calculate risk score from various flags
        risk_flags = []
        risk_score = 0
        
        # Check various risk indicators
        if security.get("is_honeypot") == "1":
            risk_flags.append("honeypot")
            risk_score += 40
        
        if security.get("is_open_source") != "1":
            risk_flags.append("not_open_source")
            risk_score += 10
        
        if security.get("is_proxy") == "1":
            risk_flags.append("proxy_contract")
            risk_score += 15
        
        if security.get("is_mintable") == "1":
            risk_flags.append("mintable")
            risk_score += 20
        
        if security.get("can_take_back_ownership") == "1":
            risk_flags.append("can_take_back_ownership")
            risk_score += 25
        
        if security.get("owner_change_balance") == "1":
            risk_flags.append("owner_can_change_balance")
            risk_score += 30
        
        if security.get("hidden_owner") == "1":
            risk_flags.append("hidden_owner")
            risk_score += 20
        
        if security.get("selfdestruct") == "1":
            risk_flags.append("selfdestruct")
            risk_score += 35
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "high"
        elif risk_score >= 40:
            risk_level = "medium"
        elif risk_score >= 20:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        return {
            "risk_level": risk_level,
            "risk_score": min(100, risk_score),
            "flags": risk_flags,
            "holder_count": int(security.get("holder_count", 0)),
            "total_supply": security.get("total_supply", "0"),
            "creator_address": security.get("creator_address", ""),
            "is_verified": security.get("is_open_source") == "1",
        }
    
    async def get_nft_security(
        self,
        nft_address: str,
        chain_id: str = "1"
    ) -> Dict[str, Any]:
        """Get NFT contract security information.
        
        Args:
            nft_address: NFT contract address
            chain_id: Chain ID
            
        Returns:
            NFT security information
        """
        try:
            url = f"{self.BASE_URL}/nft_security/{chain_id}"
            params = {"contract_addresses": nft_address}
            
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data.get("result", {}).get(nft_address.lower(), {})
            
        except httpx.HTTPError as e:
            logger.error(f"GoPlus NFT security error: {e}")
            return {}
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


class RiskAggregator:
    """Aggregate risk scores from multiple sources."""
    
    def __init__(
        self,
        rugcheck_client: Optional[RugCheckClient] = None,
        goplus_client: Optional[GoPlusClient] = None,
    ):
        """Initialize risk aggregator.
        
        Args:
            rugcheck_client: RugCheck client instance
            goplus_client: GoPlus client instance
        """
        self.rugcheck = rugcheck_client or RugCheckClient()
        self.goplus = goplus_client or GoPlusClient()
    
    async def get_aggregate_risk(
        self,
        token_address: str,
        chain: str = "ethereum"
    ) -> Dict[str, Any]:
        """Get aggregated risk assessment from multiple sources.
        
        Args:
            token_address: Token address
            chain: Blockchain network
            
        Returns:
            Aggregated risk assessment
        """
        chain_id_map = {
            "ethereum": "1",
            "bsc": "56",
            "polygon": "137",
            "avalanche": "43114",
            "arbitrum": "42161",
        }
        
        all_flags = []
        risk_scores = []
        
        # Get GoPlus assessment for EVM chains
        if chain in chain_id_map:
            chain_id = chain_id_map[chain]
            goplus_risk = await self.goplus.get_risk_assessment(token_address, chain_id)
            
            if goplus_risk.get("risk_score"):
                risk_scores.append(goplus_risk["risk_score"])
                all_flags.extend(goplus_risk.get("flags", []))
        
        # Get RugCheck assessment for Solana
        if chain == "solana":
            rugcheck_risk = await self.rugcheck.get_risk_score(token_address)
            
            if rugcheck_risk.get("risk_score"):
                risk_scores.append(rugcheck_risk["risk_score"])
                all_flags.extend(rugcheck_risk.get("flags", []))
        
        # Calculate aggregate score
        if risk_scores:
            avg_score = sum(risk_scores) / len(risk_scores)
        else:
            avg_score = 50  # Neutral if no data
        
        # Determine overall risk level
        if avg_score >= 70:
            risk_level = "high"
        elif avg_score >= 40:
            risk_level = "medium"
        elif avg_score >= 20:
            risk_level = "low"
        else:
            risk_level = "minimal"
        
        # Deduplicate flags
        unique_flags = list(set(all_flags))
        
        return {
            "aggregate_risk_score": avg_score,
            "risk_level": risk_level,
            "flags": unique_flags,
            "sources_checked": len(risk_scores),
            "chain": chain,
        }
    
    async def close(self):
        """Close all clients."""
        await self.rugcheck.close()
        await self.goplus.close()
