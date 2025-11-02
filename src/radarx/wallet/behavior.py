"""
Behavioral Pattern Detector

Detects trading patterns and behavioral characteristics in wallet activity.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np


class BehaviorDetector:
    """Detects behavioral patterns in wallet trading activity."""
    
    # Pattern thresholds
    EARLY_ADOPTER_HOURS = 24  # Trades within 24h of listing
    DIAMOND_HANDS_DAYS = 30  # Holds > 30 days
    DAY_TRADER_HOURS = 24  # Positions held < 24h
    SWING_TRADER_MIN_DAYS = 1
    SWING_TRADER_MAX_DAYS = 14
    SMART_MONEY_WIN_RATE = 0.6  # 60%+ win rate
    HIGH_FREQUENCY_TRADES_PER_DAY = 10
    
    def __init__(self):
        """Initialize behavior detector."""
        pass
    
    def detect_patterns(
        self,
        trades: List[Dict[str, Any]],
        wallet_address: str,
        token_listings: Optional[Dict[str, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect all behavioral patterns for a wallet.
        
        Args:
            trades: List of wallet trades
            wallet_address: Wallet address being analyzed
            token_listings: Optional dict mapping token addresses to listing timestamps
            
        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []
        
        # Early adopter
        if token_listings:
            early_conf = self._check_early_adopter(trades, token_listings)
            if early_conf > 0:
                patterns.append({
                    'pattern': 'early_adopter',
                    'confidence': early_conf
                })
        
        # Diamond hands
        diamond_conf = self._check_diamond_hands(trades)
        if diamond_conf > 0:
            patterns.append({
                'pattern': 'diamond_hands',
                'confidence': diamond_conf
            })
        
        # Swing trader
        swing_conf = self._check_swing_trader(trades)
        if swing_conf > 0:
            patterns.append({
                'pattern': 'swing_trader',
                'confidence': swing_conf
            })
        
        # Day trader
        day_conf = self._check_day_trader(trades)
        if day_conf > 0:
            patterns.append({
                'pattern': 'day_trader',
                'confidence': day_conf
            })
        
        # Smart money
        smart_conf = self._check_smart_money(trades)
        if smart_conf > 0:
            patterns.append({
                'pattern': 'smart_money',
                'confidence': smart_conf
            })
        
        # Wash trader
        wash_score = self.calculate_wash_trading_score(trades)
        if wash_score > 0.3:
            patterns.append({
                'pattern': 'wash_trader',
                'confidence': wash_score
            })
        
        # Pump chaser
        pump_conf = self._check_pump_chaser(trades)
        if pump_conf > 0:
            patterns.append({
                'pattern': 'pump_chaser',
                'confidence': pump_conf
            })
        
        # Profit taker
        profit_conf = self._check_profit_taker(trades)
        if profit_conf > 0:
            patterns.append({
                'pattern': 'profit_taker',
                'confidence': profit_conf
            })
        
        # Loss cutter
        loss_conf = self._check_loss_cutter(trades)
        if loss_conf > 0:
            patterns.append({
                'pattern': 'loss_cutter',
                'confidence': loss_conf
            })
        
        return patterns
    
    def _check_early_adopter(
        self,
        trades: List[Dict[str, Any]],
        token_listings: Dict[str, datetime]
    ) -> float:
        """Check if wallet is an early adopter."""
        early_trades = 0
        total_tradeable = 0
        
        for trade in trades:
            token = trade.get('token')
            if not token or token not in token_listings:
                continue
            
            total_tradeable += 1
            listing_time = token_listings[token]
            trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
            
            hours_since_listing = (trade_time - listing_time).total_seconds() / 3600
            if hours_since_listing <= self.EARLY_ADOPTER_HOURS:
                early_trades += 1
        
        if total_tradeable == 0:
            return 0.0
        
        return early_trades / total_tradeable
    
    def _check_diamond_hands(self, trades: List[Dict[str, Any]]) -> float:
        """Check if wallet exhibits diamond hands behavior."""
        if not trades:
            return 0.0
        
        hold_times = []
        for trade in trades:
            if 'buy_timestamp' in trade and 'sell_timestamp' in trade:
                buy = datetime.fromisoformat(trade['buy_timestamp'].replace('Z', '+00:00'))
                sell = datetime.fromisoformat(trade['sell_timestamp'].replace('Z', '+00:00'))
                hold_days = (sell - buy).days
                hold_times.append(hold_days)
        
        if not hold_times:
            return 0.0
        
        long_holds = sum(1 for h in hold_times if h >= self.DIAMOND_HANDS_DAYS)
        return long_holds / len(hold_times)
    
    def _check_swing_trader(self, trades: List[Dict[str, Any]]) -> float:
        """Check if wallet is a swing trader."""
        if not trades:
            return 0.0
        
        hold_times = []
        for trade in trades:
            if 'buy_timestamp' in trade and 'sell_timestamp' in trade:
                buy = datetime.fromisoformat(trade['buy_timestamp'].replace('Z', '+00:00'))
                sell = datetime.fromisoformat(trade['sell_timestamp'].replace('Z', '+00:00'))
                hold_days = (sell - buy).days
                hold_times.append(hold_days)
        
        if not hold_times:
            return 0.0
        
        swing_trades = sum(
            1 for h in hold_times
            if self.SWING_TRADER_MIN_DAYS <= h <= self.SWING_TRADER_MAX_DAYS
        )
        
        return swing_trades / len(hold_times)
    
    def _check_day_trader(self, trades: List[Dict[str, Any]]) -> float:
        """Check if wallet is a day trader."""
        if not trades:
            return 0.0
        
        hold_times = []
        for trade in trades:
            if 'buy_timestamp' in trade and 'sell_timestamp' in trade:
                buy = datetime.fromisoformat(trade['buy_timestamp'].replace('Z', '+00:00'))
                sell = datetime.fromisoformat(trade['sell_timestamp'].replace('Z', '+00:00'))
                hold_hours = (sell - buy).total_seconds() / 3600
                hold_times.append(hold_hours)
        
        if not hold_times:
            return 0.0
        
        day_trades = sum(1 for h in hold_times if h <= self.DAY_TRADER_HOURS)
        return day_trades / len(hold_times)
    
    def _check_smart_money(self, trades: List[Dict[str, Any]]) -> float:
        """Check if wallet exhibits smart money characteristics."""
        if not trades:
            return 0.0
        
        profitable = sum(1 for t in trades if t.get('pnl', 0) > 0)
        win_rate = profitable / len(trades)
        
        if win_rate < self.SMART_MONEY_WIN_RATE:
            return 0.0
        
        # Scale confidence based on how much above threshold
        confidence = min(1.0, (win_rate - self.SMART_MONEY_WIN_RATE) / (1.0 - self.SMART_MONEY_WIN_RATE))
        return confidence
    
    def _check_pump_chaser(self, trades: List[Dict[str, Any]]) -> float:
        """Check if wallet chases pumps."""
        if not trades:
            return 0.0
        
        pump_buys = 0
        for trade in trades:
            # Check if bought during rapid price increase
            if trade.get('action') == 'buy' and trade.get('price_change_5m', 0) > 0.1:  # >10% in 5min
                pump_buys += 1
        
        return pump_buys / len(trades) if trades else 0.0
    
    def _check_profit_taker(self, trades: List[Dict[str, Any]]) -> float:
        """Check if wallet consistently takes profits."""
        if not trades:
            return 0.0
        
        sells = [t for t in trades if t.get('action') == 'sell']
        if not sells:
            return 0.0
        
        profit_sells = sum(1 for t in sells if t.get('pnl', 0) > 0)
        return profit_sells / len(sells)
    
    def _check_loss_cutter(self, trades: List[Dict[str, Any]]) -> float:
        """Check if wallet cuts losses quickly."""
        if not trades:
            return 0.0
        
        loss_trades = [t for t in trades if t.get('pnl', 0) < 0]
        if not loss_trades:
            return 0.0
        
        quick_cuts = 0
        for trade in loss_trades:
            if 'buy_timestamp' in trade and 'sell_timestamp' in trade:
                buy = datetime.fromisoformat(trade['buy_timestamp'].replace('Z', '+00:00'))
                sell = datetime.fromisoformat(trade['sell_timestamp'].replace('Z', '+00:00'))
                hold_hours = (sell - buy).total_seconds() / 3600
                
                # Quick cut if sold within 24h of buying at a loss
                if hold_hours <= 24 and abs(trade['pnl']) / trade.get('buy_amount', 1) < 0.2:  # <20% loss
                    quick_cuts += 1
        
        return quick_cuts / len(loss_trades)
    
    def calculate_wash_trading_score(self, trades: List[Dict[str, Any]]) -> float:
        """
        Calculate likelihood of wash trading (0-1).
        
        Wash trading indicators:
        - Circular trades (buy and sell same token repeatedly)
        - Self-trading patterns
        - Artificially inflated volume
        """
        if len(trades) < 5:
            return 0.0
        
        # Check for circular trading patterns
        token_trades = defaultdict(list)
        for trade in trades:
            token = trade.get('token')
            if token:
                token_trades[token].append(trade)
        
        circular_count = 0
        total_patterns = 0
        
        for token, token_trade_list in token_trades.items():
            if len(token_trade_list) < 4:
                continue
            
            total_patterns += 1
            
            # Check for buy-sell-buy-sell pattern
            actions = [t.get('action') for t in token_trade_list]
            
            # Count alternating buy/sell sequences
            alternating = 0
            for i in range(len(actions) - 1):
                if actions[i] != actions[i + 1]:
                    alternating += 1
            
            # High alternation suggests wash trading
            if alternating / (len(actions) - 1) > 0.7:  # >70% alternation
                circular_count += 1
        
        if total_patterns == 0:
            return 0.0
        
        return circular_count / total_patterns
    
    def is_smart_money(self, trades: List[Dict[str, Any]], threshold: float = 0.6) -> bool:
        """Check if wallet qualifies as smart money."""
        return self._check_smart_money(trades) >= threshold
    
    def is_wash_trader(self, trades: List[Dict[str, Any]], threshold: float = 0.5) -> bool:
        """Check if wallet exhibits wash trading behavior."""
        return self.calculate_wash_trading_score(trades) >= threshold
    
    def is_early_adopter(
        self,
        trades: List[Dict[str, Any]],
        token_listings: Dict[str, datetime],
        threshold: float = 0.5
    ) -> bool:
        """Check if wallet is an early adopter."""
        return self._check_early_adopter(trades, token_listings) >= threshold
