"""
Trade Matcher

Matches buy and sell trades, accounting for routing, slippage,
internal transfers, and liquidity events.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class TradeMatcher:
    """
    Matches trades to compute realized PnL and trade metrics.
    """
    
    def __init__(
        self,
        slippage_tolerance_pct: float = 2.0,
        max_routing_hops: int = 3,
    ):
        """
        Initialize trade matcher.
        
        Args:
            slippage_tolerance_pct: Allowed slippage percentage
            max_routing_hops: Maximum DEX routing hops
        """
        self.slippage_tolerance = slippage_tolerance_pct / 100.0
        self.max_routing_hops = max_routing_hops
    
    def match_trades(
        self,
        trades: List[Dict[str, Any]],
        min_trade_size_usd: Optional[float] = None,
        min_holdings_usd: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Match trades per wallet and compute metrics.
        
        Args:
            trades: List of all trades
            min_trade_size_usd: Minimum trade size filter
            min_holdings_usd: Minimum holdings filter
            
        Returns:
            Dictionary mapping wallet address to trade data
        """
        # Group trades by wallet
        wallet_trades = self._group_by_wallet(trades)
        
        # Match buys and sells for each wallet
        matched_wallets = {}
        
        for wallet_address, wallet_trade_list in wallet_trades.items():
            # Filter by trade size
            if min_trade_size_usd:
                wallet_trade_list = [
                    t for t in wallet_trade_list
                    if t.get("amount_usd", 0) >= min_trade_size_usd
                ]
            
            if not wallet_trade_list:
                continue
            
            # Match buys and sells using FIFO
            matched_trades = self._match_buys_and_sells(wallet_trade_list)
            
            # Calculate holdings
            current_holdings_usd = self._calculate_current_holdings(
                wallet_trade_list
            )
            
            # Filter by minimum holdings
            if min_holdings_usd and current_holdings_usd < min_holdings_usd:
                continue
            
            matched_wallets[wallet_address] = {
                "trades": matched_trades,
                "current_holdings_usd": current_holdings_usd,
                "total_buy_volume_usd": sum(
                    t.get("amount_usd", 0) for t in wallet_trade_list
                    if t.get("side") == "buy"
                ),
                "total_sell_volume_usd": sum(
                    t.get("amount_usd", 0) for t in wallet_trade_list
                    if t.get("side") == "sell"
                ),
            }
        
        return matched_wallets
    
    def _group_by_wallet(
        self,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group trades by wallet address."""
        wallet_trades = {}
        
        for trade in trades:
            # Identify wallet address (buyer or seller)
            if trade.get("side") == "buy":
                wallet = trade.get("buyer")
            else:
                wallet = trade.get("seller")
            
            if not wallet:
                continue
            
            if wallet not in wallet_trades:
                wallet_trades[wallet] = []
            
            wallet_trades[wallet].append(trade)
        
        return wallet_trades
    
    def _match_buys_and_sells(
        self,
        trades: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Match buys and sells using FIFO to compute realized PnL.
        
        Args:
            trades: List of trades for a wallet
            
        Returns:
            List of matched trades with PnL and ROI
        """
        # Sort by timestamp
        sorted_trades = sorted(
            trades,
            key=lambda t: self._parse_timestamp(t.get("timestamp"))
        )
        
        # Separate buys and sells
        buys = [t for t in sorted_trades if t.get("side") == "buy"]
        sells = [t for t in sorted_trades if t.get("side") == "sell"]
        
        matched = []
        buy_queue = list(buys)  # FIFO queue
        
        for sell in sells:
            sell_amount = sell.get("amount_tokens", 0)
            sell_value_usd = sell.get("amount_usd", 0)
            sell_time = self._parse_timestamp(sell.get("timestamp"))
            
            # Match against buys in FIFO order
            remaining_sell_amount = sell_amount
            total_cost_basis = 0
            matched_buys = []
            
            while buy_queue and remaining_sell_amount > 0:
                buy = buy_queue[0]
                buy_amount = buy.get("amount_tokens", 0)
                buy_value_usd = buy.get("amount_usd", 0)
                buy_time = self._parse_timestamp(buy.get("timestamp"))
                
                # Match as much as possible from this buy
                match_amount = min(buy_amount, remaining_sell_amount)
                match_cost = (match_amount / buy_amount) * buy_value_usd if buy_amount > 0 else 0
                
                total_cost_basis += match_cost
                matched_buys.append({
                    "buy_time": buy_time,
                    "amount": match_amount,
                    "cost": match_cost,
                })
                
                # Update quantities
                remaining_sell_amount -= match_amount
                buy["amount_tokens"] = buy_amount - match_amount
                
                # Remove buy if fully matched
                if buy["amount_tokens"] <= 0:
                    buy_queue.pop(0)
            
            # Calculate PnL and ROI for this sell
            if matched_buys:
                pnl = sell_value_usd - total_cost_basis
                roi = (pnl / total_cost_basis) if total_cost_basis > 0 else 0
                
                # Calculate holding duration (weighted average)
                total_holding_seconds = 0
                for mb in matched_buys:
                    holding_seconds = (sell_time - mb["buy_time"]).total_seconds()
                    weight = mb["cost"] / total_cost_basis if total_cost_basis > 0 else 0
                    total_holding_seconds += holding_seconds * weight
                
                matched.append({
                    **sell,
                    "entry_time": matched_buys[0]["buy_time"].isoformat(),
                    "exit_time": sell_time.isoformat(),
                    "cost_basis_usd": total_cost_basis,
                    "sell_value_usd": sell_value_usd,
                    "pnl": pnl,
                    "roi": roi,
                    "holding_duration_hours": total_holding_seconds / 3600,
                    "matched_buys": len(matched_buys),
                })
        
        return matched
    
    def _calculate_current_holdings(
        self,
        trades: List[Dict[str, Any]],
    ) -> float:
        """Calculate current unrealized holdings value."""
        # Sum all buys minus all sells
        total_bought = sum(
            t.get("amount_usd", 0) for t in trades if t.get("side") == "buy"
        )
        total_sold = sum(
            t.get("amount_usd", 0) for t in trades if t.get("side") == "sell"
        )
        
        # Simplified - assumes current value equals buy value
        # In production, would use current token price
        return max(0, total_bought - total_sold)
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp to datetime object."""
        if isinstance(timestamp, datetime):
            return timestamp
        if isinstance(timestamp, str):
            ts_str = timestamp.replace('Z', '+00:00')
            return datetime.fromisoformat(ts_str)
        return datetime.utcnow()
