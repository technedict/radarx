"""Data normalization and validation utilities."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalize and validate data from various sources."""
    
    @staticmethod
    def normalize_address(address: str, chain: str = "ethereum") -> str:
        """Normalize blockchain address format.
        
        Args:
            address: Blockchain address
            chain: Chain identifier
            
        Returns:
            Normalized address
        """
        address = address.strip()
        
        if chain in ["ethereum", "bsc", "polygon", "avalanche", "arbitrum", "base"]:
            # EVM chains - ensure 0x prefix and lowercase
            if not address.startswith("0x"):
                address = "0x" + address
            return address.lower()
        elif chain == "solana":
            # Solana addresses are base58 encoded
            return address
        else:
            return address.lower()
    
    @staticmethod
    def normalize_timestamp(ts: Any) -> datetime:
        """Normalize various timestamp formats to datetime.
        
        Args:
            ts: Timestamp in various formats
            
        Returns:
            UTC datetime object
        """
        if isinstance(ts, datetime):
            return ts
        elif isinstance(ts, str):
            # Try ISO format
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except:
                pass
            # Try common formats
            for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]:
                try:
                    return datetime.strptime(ts, fmt)
                except:
                    continue
        elif isinstance(ts, (int, float)):
            # Unix timestamp (assume seconds, convert if milliseconds)
            if ts > 10**10:  # Likely milliseconds
                ts = ts / 1000
            return datetime.fromtimestamp(ts)
        
        raise ValueError(f"Cannot normalize timestamp: {ts}")
    
    @staticmethod
    def normalize_chain_name(chain: str) -> str:
        """Normalize chain name to standard format.
        
        Args:
            chain: Chain name in various formats
            
        Returns:
            Normalized chain name
        """
        chain = chain.lower().strip()
        
        # Map common variations
        chain_map = {
            "eth": "ethereum",
            "ether": "ethereum",
            "bsc": "bsc",
            "bnb": "bsc",
            "binance": "bsc",
            "sol": "solana",
            "matic": "polygon",
            "poly": "polygon",
            "avax": "avalanche",
            "arb": "arbitrum",
        }
        
        return chain_map.get(chain, chain)
    
    @staticmethod
    def validate_token_data(data: Dict[str, Any]) -> bool:
        """Validate token data has required fields.
        
        Args:
            data: Token data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ["address", "chain"]
        
        for field in required_fields:
            if field not in data or not data[field]:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Validate address format
        address = data.get("address", "")
        chain = data.get("chain", "")
        
        if chain in ["ethereum", "bsc", "polygon"]:
            if not re.match(r"^0x[a-fA-F0-9]{40}$", address):
                logger.warning(f"Invalid EVM address format: {address}")
                return False
        
        return True
    
    @staticmethod
    def validate_wallet_address(address: str, chain: str) -> bool:
        """Validate wallet address format.
        
        Args:
            address: Wallet address
            chain: Blockchain network
            
        Returns:
            True if valid format
        """
        chain = DataNormalizer.normalize_chain_name(chain)
        
        if chain in ["ethereum", "bsc", "polygon", "avalanche", "arbitrum", "base"]:
            # EVM address: 0x followed by 40 hex characters
            return bool(re.match(r"^0x[a-fA-F0-9]{40}$", address))
        elif chain == "solana":
            # Solana address: 32-44 base58 characters
            return bool(re.match(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$", address))
        else:
            logger.warning(f"Unknown chain for validation: {chain}")
            return False
    
    @staticmethod
    def clean_numeric(value: Any, default: float = 0.0) -> float:
        """Clean and convert numeric values.
        
        Args:
            value: Value to clean
            default: Default value if conversion fails
            
        Returns:
            Float value
        """
        if value is None:
            return default
        
        try:
            if isinstance(value, str):
                # Remove common formatting
                value = value.replace(",", "").replace("$", "").strip()
            return float(value)
        except (ValueError, TypeError):
            logger.debug(f"Could not convert to numeric: {value}")
            return default
    
    @staticmethod
    def normalize_token_transfer(transfer: Dict[str, Any], chain: str) -> Dict[str, Any]:
        """Normalize token transfer event from different blockchain indexers.
        
        Args:
            transfer: Raw transfer data
            chain: Blockchain network
            
        Returns:
            Normalized transfer data
        """
        chain = DataNormalizer.normalize_chain_name(chain)
        
        if chain in ["ethereum", "bsc"]:
            # Etherscan/BscScan format
            return {
                "hash": transfer.get("hash"),
                "from_address": DataNormalizer.normalize_address(transfer.get("from", ""), chain),
                "to_address": DataNormalizer.normalize_address(transfer.get("to", ""), chain),
                "value": transfer.get("value", "0"),
                "token_symbol": transfer.get("tokenSymbol"),
                "token_name": transfer.get("tokenName"),
                "token_decimal": int(transfer.get("tokenDecimal", 18)),
                "timestamp": DataNormalizer.normalize_timestamp(int(transfer.get("timeStamp", 0))),
                "block_number": int(transfer.get("blockNumber", 0)),
                "chain": chain,
            }
        elif chain == "solana":
            # Solscan format
            return {
                "hash": transfer.get("signature"),
                "from_address": transfer.get("src"),
                "to_address": transfer.get("dst"),
                "value": transfer.get("amount", "0"),
                "token_symbol": transfer.get("symbol"),
                "token_name": transfer.get("name"),
                "token_decimal": int(transfer.get("decimals", 9)),
                "timestamp": DataNormalizer.normalize_timestamp(transfer.get("blockTime", 0)),
                "block_number": int(transfer.get("slot", 0)),
                "chain": chain,
            }
        else:
            # Generic format
            return transfer
    
    @staticmethod
    def aggregate_holder_stats(holders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate holder distribution statistics.
        
        Args:
            holders: List of holder data
            
        Returns:
            Aggregated statistics
        """
        if not holders:
            return {
                "total_holders": 0,
                "top10_percentage": 0,
                "top50_percentage": 0,
                "gini_coefficient": 0,
            }
        
        # Sort by balance descending
        sorted_holders = sorted(
            holders,
            key=lambda h: DataNormalizer.clean_numeric(h.get("balance", 0)),
            reverse=True
        )
        
        balances = [DataNormalizer.clean_numeric(h.get("balance", 0)) for h in sorted_holders]
        total_supply = sum(balances)
        
        if total_supply == 0:
            return {
                "total_holders": len(holders),
                "top10_percentage": 0,
                "top50_percentage": 0,
                "gini_coefficient": 0,
            }
        
        # Top 10 concentration
        top10_balance = sum(balances[:10])
        top10_pct = (top10_balance / total_supply) * 100
        
        # Top 50 concentration
        top50_balance = sum(balances[:50])
        top50_pct = (top50_balance / total_supply) * 100
        
        # Simple Gini coefficient calculation
        gini = DataNormalizer._calculate_gini(balances)
        
        return {
            "total_holders": len(holders),
            "top10_percentage": top10_pct,
            "top50_percentage": top50_pct,
            "gini_coefficient": gini,
        }
    
    @staticmethod
    def _calculate_gini(balances: List[float]) -> float:
        """Calculate Gini coefficient for holder distribution.
        
        Args:
            balances: List of holder balances
            
        Returns:
            Gini coefficient (0-1, higher = more unequal)
        """
        if not balances or sum(balances) == 0:
            return 0.0
        
        # Sort balances
        sorted_balances = sorted(balances)
        n = len(sorted_balances)
        
        # Calculate Gini
        cumsum = 0
        for i, balance in enumerate(sorted_balances):
            cumsum += (n - i) * balance
        
        total = sum(sorted_balances)
        gini = (2 * cumsum) / (n * total) - (n + 1) / n
        
        return max(0.0, min(1.0, gini))
