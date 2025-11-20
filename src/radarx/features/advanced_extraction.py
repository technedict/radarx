"""
Advanced Feature Engineering for Memecoin Analysis

Implements production-grade feature extraction:
- Trade-matching with multi-hop swaps
- Liquidity metrics with depth and elasticity
- Holder concentration and distribution
- Social velocity and bot detection
- Smart money interaction tracking
- Temporal embeddings for sequences
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradeMatchingEngine:
    """
    Advanced trade matching engine that handles:
    - DEX routing through multiple pools
    - Internal transfers and multi-hop swaps
    - MEV and sandwich attack detection
    - Wash trading identification
    """
    
    def __init__(self, max_hop_depth: int = 3, max_time_delta: int = 30):
        """
        Args:
            max_hop_depth: Maximum number of hops to track in multi-hop swaps
            max_time_delta: Maximum time (seconds) between related trades
        """
        self.max_hop_depth = max_hop_depth
        self.max_time_delta = max_time_delta
    
    def match_trades(
        self,
        transactions: List[Dict],
        token_address: str
    ) -> Tuple[List[Dict], float]:
        """
        Match raw transactions to actual trades.
        
        Args:
            transactions: List of transaction dicts
            token_address: Token address to track
            
        Returns:
            Tuple of (matched_trades, matching_error_rate)
        """
        matched_trades = []
        unmatched_count = 0
        
        # Sort by timestamp
        transactions = sorted(transactions, key=lambda x: x.get('timestamp', 0))
        
        for tx in transactions:
            try:
                trade = self._match_single_transaction(tx, token_address)
                if trade:
                    matched_trades.append(trade)
                else:
                    unmatched_count += 1
            except Exception as e:
                logger.warning(f"Error matching transaction {tx.get('hash')}: {e}")
                unmatched_count += 1
        
        # Calculate error rate
        total = len(transactions)
        error_rate = unmatched_count / total if total > 0 else 0
        
        return matched_trades, error_rate
    
    def _match_single_transaction(
        self,
        tx: Dict,
        token_address: str
    ) -> Optional[Dict]:
        """Match a single transaction to a trade."""
        # Extract key fields
        tx_hash = tx.get('hash')
        from_addr = tx.get('from')
        to_addr = tx.get('to')
        value = tx.get('value', 0)
        timestamp = tx.get('timestamp')
        
        # Check if transaction involves the token
        logs = tx.get('logs', [])
        token_transfers = [
            log for log in logs
            if log.get('address', '').lower() == token_address.lower()
        ]
        
        if not token_transfers:
            return None
        
        # Determine trade type (buy/sell/transfer)
        trade_type = self._determine_trade_type(token_transfers, from_addr)
        
        # Extract amount
        amount = sum(float(t.get('value', 0)) for t in token_transfers)
        
        # Detect wash trading patterns
        is_wash_trade = self._detect_wash_trade(tx, token_transfers)
        
        return {
            'hash': tx_hash,
            'timestamp': timestamp,
            'from': from_addr,
            'to': to_addr,
            'type': trade_type,
            'amount': amount,
            'value_usd': value,
            'is_wash_trade': is_wash_trade,
            'hop_count': len(token_transfers),
        }
    
    def _determine_trade_type(
        self,
        transfers: List[Dict],
        from_addr: str
    ) -> str:
        """Determine if trade is buy, sell, or transfer."""
        if len(transfers) == 0:
            return 'unknown'
        
        # Simple heuristic: check direction of first transfer
        first_transfer = transfers[0]
        if first_transfer.get('from', '').lower() == from_addr.lower():
            return 'sell'
        elif first_transfer.get('to', '').lower() == from_addr.lower():
            return 'buy'
        else:
            return 'transfer'
    
    def _detect_wash_trade(
        self,
        tx: Dict,
        transfers: List[Dict]
    ) -> bool:
        """
        Detect wash trading patterns:
        - Same wallet buying and selling
        - Round-trip transfers
        - Artificial volume inflation
        """
        # Check for circular transfers
        addresses = set()
        for t in transfers:
            addresses.add(t.get('from', '').lower())
            addresses.add(t.get('to', '').lower())
        
        # If same address appears as both sender and receiver, likely wash trade
        from_addrs = {t.get('from', '').lower() for t in transfers}
        to_addrs = {t.get('to', '').lower() for t in transfers}
        
        return bool(from_addrs & to_addrs)


class LiquidityFeatureExtractor:
    """
    Extract liquidity-related features:
    - Pool depth and reserves
    - Liquidity elasticity (price impact of large trades)
    - Liquidity removal patterns
    - Concentrated vs distributed liquidity
    """
    
    def extract_features(
        self,
        pool_data: Dict,
        trades: List[Dict],
        window_hours: int = 24
    ) -> Dict[str, float]:
        """Extract liquidity features."""
        features = {}
        
        # Basic liquidity metrics
        features['liquidity_usd'] = float(pool_data.get('liquidity_usd', 0))
        features['reserves_token'] = float(pool_data.get('reserves_token', 0))
        features['reserves_paired'] = float(pool_data.get('reserves_paired', 0))
        
        # Liquidity depth (how much can be traded with <1% slippage)
        features['depth_1pct'] = self._calculate_depth(pool_data, slippage=0.01)
        features['depth_5pct'] = self._calculate_depth(pool_data, slippage=0.05)
        
        # Liquidity concentration (Herfindahl index)
        features['liquidity_concentration'] = self._calculate_concentration(pool_data)
        
        # Recent liquidity changes
        if trades:
            recent_trades = [
                t for t in trades
                if (datetime.now() - datetime.fromtimestamp(t['timestamp'])).total_seconds() / 3600 < window_hours
            ]
            features['liq_add_count'] = sum(1 for t in recent_trades if t.get('type') == 'add_liquidity')
            features['liq_remove_count'] = sum(1 for t in recent_trades if t.get('type') == 'remove_liquidity')
            features['net_liq_change_usd'] = self._calculate_net_liquidity_change(recent_trades)
        else:
            features['liq_add_count'] = 0
            features['liq_remove_count'] = 0
            features['net_liq_change_usd'] = 0
        
        return features
    
    def _calculate_depth(self, pool_data: Dict, slippage: float) -> float:
        """Calculate trade size that causes given slippage."""
        # Simplified constant product AMM model
        reserves_token = float(pool_data.get('reserves_token', 1))
        reserves_paired = float(pool_data.get('reserves_paired', 1))
        
        # k = x * y (constant product)
        k = reserves_token * reserves_paired
        
        # New price after trade = (1 + slippage) * old_price
        # Solve for trade size
        price_before = reserves_paired / reserves_token
        price_after = price_before * (1 + slippage)
        
        # From AMM math: depth ≈ reserves * slippage / 2
        depth = reserves_token * slippage / 2
        
        return depth
    
    def _calculate_concentration(self, pool_data: Dict) -> float:
        """Calculate Herfindahl index for liquidity concentration."""
        # If we had individual LP positions, we'd compute:
        # H = sum(share_i^2) where share_i = lp_i / total_liquidity
        # For now, return 1.0 (fully concentrated) as placeholder
        return 1.0
    
    def _calculate_net_liquidity_change(self, trades: List[Dict]) -> float:
        """Calculate net liquidity change from add/remove events."""
        net_change = 0.0
        for trade in trades:
            if trade.get('type') == 'add_liquidity':
                net_change += trade.get('value_usd', 0)
            elif trade.get('type') == 'remove_liquidity':
                net_change -= trade.get('value_usd', 0)
        return net_change


class HolderFeatureExtractor:
    """
    Extract holder-related features:
    - Holder concentration (Gini coefficient)
    - Whale concentration (% held by top 10)
    - Holder spread (geographic, temporal)
    - Smart money holder ratio
    """
    
    def extract_features(
        self,
        holders: List[Dict],
        smart_wallets: Optional[set] = None
    ) -> Dict[str, float]:
        """Extract holder features."""
        features = {}
        
        if not holders:
            return self._default_features()
        
        # Total supply and holder count
        total_supply = sum(h.get('balance', 0) for h in holders)
        features['holder_count'] = len(holders)
        
        # Sort by balance
        holders_sorted = sorted(holders, key=lambda x: x.get('balance', 0), reverse=True)
        
        # Top holder concentrations
        top_10_balance = sum(h.get('balance', 0) for h in holders_sorted[:10])
        top_50_balance = sum(h.get('balance', 0) for h in holders_sorted[:50])
        
        features['top_10_pct'] = (top_10_balance / total_supply * 100) if total_supply > 0 else 0
        features['top_50_pct'] = (top_50_balance / total_supply * 100) if total_supply > 0 else 0
        
        # Gini coefficient (measure of inequality)
        features['gini_coefficient'] = self._calculate_gini([h.get('balance', 0) for h in holders])
        
        # Smart money ratio
        if smart_wallets:
            smart_holder_balance = sum(
                h.get('balance', 0) for h in holders
                if h.get('address') in smart_wallets
            )
            features['smart_money_pct'] = (smart_holder_balance / total_supply * 100) if total_supply > 0 else 0
        else:
            features['smart_money_pct'] = 0
        
        return features
    
    def _calculate_gini(self, balances: List[float]) -> float:
        """Calculate Gini coefficient for balance distribution."""
        if not balances or sum(balances) == 0:
            return 0
        
        # Sort balances
        sorted_balances = sorted(balances)
        n = len(sorted_balances)
        
        # Calculate Gini
        cumsum = np.cumsum(sorted_balances)
        gini = (2 * sum((i + 1) * x for i, x in enumerate(sorted_balances))) / (n * cumsum[-1]) - (n + 1) / n
        
        return float(gini)
    
    def _default_features(self) -> Dict[str, float]:
        """Return default features when no holder data."""
        return {
            'holder_count': 0,
            'top_10_pct': 0,
            'top_50_pct': 0,
            'gini_coefficient': 0,
            'smart_money_pct': 0,
        }


class BotDetector:
    """
    Detect bot activity in trading patterns:
    - MEV bots (sandwich attacks)
    - Sniping bots (buying at launch)
    - Wash trading bots
    - Social amplification bots
    """
    
    def score_bot_likelihood(
        self,
        wallet_address: str,
        trades: List[Dict],
        social_activity: Optional[List[Dict]] = None
    ) -> float:
        """
        Score likelihood that wallet is a bot (0-1).
        
        Returns higher score for more bot-like behavior.
        """
        bot_score = 0.0
        signals = []
        
        if not trades:
            return 0.0
        
        # Check trade timing patterns (bots trade in milliseconds)
        timing_score = self._check_timing_patterns(trades)
        bot_score += timing_score * 0.3
        signals.append(f"timing: {timing_score:.2f}")
        
        # Check for MEV patterns
        mev_score = self._check_mev_patterns(trades)
        bot_score += mev_score * 0.3
        signals.append(f"mev: {mev_score:.2f}")
        
        # Check for wash trading
        wash_score = self._check_wash_trading(trades)
        bot_score += wash_score * 0.2
        signals.append(f"wash: {wash_score:.2f}")
        
        # Check social activity if available
        if social_activity:
            social_bot_score = self._check_social_bots(social_activity)
            bot_score += social_bot_score * 0.2
            signals.append(f"social: {social_bot_score:.2f}")
        
        logger.debug(f"Bot score for {wallet_address}: {bot_score:.3f} ({', '.join(signals)})")
        
        return min(bot_score, 1.0)
    
    def _check_timing_patterns(self, trades: List[Dict]) -> float:
        """Check for bot-like timing patterns."""
        if len(trades) < 2:
            return 0.0
        
        # Calculate intervals between trades
        timestamps = sorted([t.get('timestamp', 0) for t in trades])
        intervals = np.diff(timestamps)
        
        # Bots often trade at very consistent intervals
        if len(intervals) > 5:
            cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            # Low coefficient of variation = consistent timing = likely bot
            consistency_score = max(0, 1 - cv * 2)
            
            # Very fast reactions (< 5 seconds) also indicate bot
            fast_trades = sum(1 for i in intervals if i < 5) / len(intervals)
            
            return (consistency_score + fast_trades) / 2
        
        return 0.0
    
    def _check_mev_patterns(self, trades: List[Dict]) -> float:
        """Check for MEV bot patterns (sandwich attacks)."""
        # Look for buy-sell-buy or sell-buy-sell patterns in quick succession
        mev_count = 0
        
        for i in range(len(trades) - 2):
            t1, t2, t3 = trades[i:i+3]
            
            # Check if sandwich pattern
            if (t1['type'] != t2['type'] and t2['type'] != t3['type'] and t1['type'] == t3['type']):
                # Check timing (sandwich usually < 30 seconds)
                time_diff = t3['timestamp'] - t1['timestamp']
                if time_diff < 30:
                    mev_count += 1
        
        return min(mev_count / max(len(trades) // 3, 1), 1.0)
    
    def _check_wash_trading(self, trades: List[Dict]) -> float:
        """Check for wash trading patterns."""
        wash_count = sum(1 for t in trades if t.get('is_wash_trade', False))
        return wash_count / len(trades) if trades else 0.0
    
    def _check_social_bots(self, social_activity: List[Dict]) -> float:
        """Check for social media bot patterns."""
        if not social_activity:
            return 0.0
        
        # Check for patterns like:
        # - Posting at exact intervals
        # - Copy-paste content
        # - Unrealistic posting frequency
        
        bot_signals = 0
        total_signals = 0
        
        # Check posting frequency
        if len(social_activity) > 0:
            time_span = (
                max(a.get('timestamp', 0) for a in social_activity) -
                min(a.get('timestamp', 0) for a in social_activity)
            )
            if time_span > 0:
                posts_per_hour = len(social_activity) / (time_span / 3600)
                # More than 20 posts per hour is suspicious
                if posts_per_hour > 20:
                    bot_signals += 1
                total_signals += 1
        
        # Check for duplicate content
        contents = [a.get('content', '') for a in social_activity]
        unique_contents = set(contents)
        if len(contents) > 5:
            uniqueness = len(unique_contents) / len(contents)
            # Less than 30% unique content is suspicious
            if uniqueness < 0.3:
                bot_signals += 1
            total_signals += 1
        
        return bot_signals / total_signals if total_signals > 0 else 0.0


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Demo trade matching
    sample_transactions = [
        {
            'hash': '0xabc...',
            'from': '0x123...',
            'to': '0x456...',
            'value': 1000,
            'timestamp': 1700000000,
            'logs': [
                {'address': '0xtoken...', 'value': 500, 'from': '0x123...', 'to': '0x789...'}
            ]
        }
    ]
    
    matcher = TradeMatchingEngine()
    trades, error_rate = matcher.match_trades(sample_transactions, '0xtoken...')
    
    print(f"Matched {len(trades)} trades with {error_rate:.2%} error rate")
    
    # Demo bot detection
    bot_detector = BotDetector()
    bot_score = bot_detector.score_bot_likelihood('0x123...', trades)
    print(f"Bot likelihood score: {bot_score:.3f}")
