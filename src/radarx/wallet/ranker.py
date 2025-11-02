"""
Wallet Ranker

Ranks wallets by performance metrics with global and chain-specific rankings.
"""

from typing import Dict, List, Any, Optional
import numpy as np


class WalletRanker:
    """Ranks wallets based on performance metrics."""
    
    def __init__(self):
        """Initialize wallet ranker."""
        self.wallets = []
        self.chain_wallets = {}
    
    def add_wallet(
        self,
        wallet_address: str,
        win_rate: float,
        pnl: float,
        sharpe_ratio: float = 0.0,
        trades: int = 0,
        chain: Optional[str] = None
    ):
        """
        Add wallet to ranking database.
        
        Args:
            wallet_address: Wallet address
            win_rate: Win rate (0-1)
            pnl: Total realized PnL
            sharpe_ratio: Sharpe ratio
            trades: Number of trades
            chain: Blockchain chain (optional)
        """
        wallet_data = {
            'address': wallet_address,
            'win_rate': win_rate,
            'pnl': pnl,
            'sharpe_ratio': sharpe_ratio,
            'trades': trades,
            'chain': chain
        }
        
        self.wallets.append(wallet_data)
        
        if chain:
            if chain not in self.chain_wallets:
                self.chain_wallets[chain] = []
            self.chain_wallets[chain].append(wallet_data)
    
    def get_rankings(
        self,
        metric: str = 'win_rate',
        chain: Optional[str] = None,
        limit: Optional[int] = None,
        min_trades: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get wallet rankings.
        
        Args:
            metric: Metric to rank by ('win_rate', 'pnl', 'sharpe_ratio', 'trades')
            chain: Optional chain filter
            limit: Maximum number of results
            min_trades: Minimum trade count threshold
            
        Returns:
            List of ranked wallets with their metrics
        """
        # Select wallet set
        if chain:
            wallet_set = self.chain_wallets.get(chain, [])
        else:
            wallet_set = self.wallets
        
        # Filter by minimum trades
        filtered = [w for w in wallet_set if w['trades'] >= min_trades]
        
        if not filtered:
            return []
        
        # Sort by metric
        if metric not in ['win_rate', 'pnl', 'sharpe_ratio', 'trades']:
            metric = 'win_rate'
        
        sorted_wallets = sorted(
            filtered,
            key=lambda x: x.get(metric, 0),
            reverse=True
        )
        
        # Add rank numbers
        for i, wallet in enumerate(sorted_wallets):
            wallet['rank'] = i + 1
        
        # Apply limit
        if limit:
            sorted_wallets = sorted_wallets[:limit]
        
        return sorted_wallets
    
    def get_wallet_rank(
        self,
        wallet_address: str,
        metric: str = 'win_rate',
        chain: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get rank information for a specific wallet.
        
        Args:
            wallet_address: Wallet to look up
            metric: Metric to rank by
            chain: Optional chain filter
            
        Returns:
            Dictionary with rank, percentile, and total count
        """
        rankings = self.get_rankings(metric=metric, chain=chain)
        
        if not rankings:
            return None
        
        for i, wallet in enumerate(rankings):
            if wallet['address'] == wallet_address:
                return {
                    'rank': i + 1,
                    'total': len(rankings),
                    'percentile': ((len(rankings) - i) / len(rankings)) * 100,
                    'metric': metric,
                    'value': wallet.get(metric, 0)
                }
        
        return None
    
    def get_percentile(
        self,
        wallet_address: str,
        metric: str = 'win_rate',
        chain: Optional[str] = None
    ) -> Optional[float]:
        """
        Calculate percentile for a wallet (0-100).
        
        Args:
            wallet_address: Wallet to calculate for
            metric: Metric to use
            chain: Optional chain filter
            
        Returns:
            Percentile value (0-100) or None if not found
        """
        rank_info = self.get_wallet_rank(wallet_address, metric, chain)
        return rank_info['percentile'] if rank_info else None
    
    def get_top_wallets(
        self,
        n: int = 10,
        metric: str = 'pnl',
        chain: Optional[str] = None,
        min_trades: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get top N wallets by metric.
        
        Args:
            n: Number of top wallets to return
            metric: Metric to rank by
            chain: Optional chain filter
            min_trades: Minimum trades required
            
        Returns:
            List of top performing wallets
        """
        return self.get_rankings(
            metric=metric,
            chain=chain,
            limit=n,
            min_trades=min_trades
        )
    
    def get_chain_statistics(self, chain: str) -> Dict[str, Any]:
        """
        Get aggregate statistics for a chain.
        
        Args:
            chain: Blockchain to analyze
            
        Returns:
            Dictionary with chain-level statistics
        """
        wallets = self.chain_wallets.get(chain, [])
        
        if not wallets:
            return {
                'total_wallets': 0,
                'avg_win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'total_trades': 0
            }
        
        win_rates = [w['win_rate'] for w in wallets]
        pnls = [w['pnl'] for w in wallets]
        trades = [w['trades'] for w in wallets]
        
        return {
            'total_wallets': len(wallets),
            'avg_win_rate': np.mean(win_rates),
            'median_win_rate': np.median(win_rates),
            'total_pnl': sum(pnls),
            'avg_pnl': np.mean(pnls),
            'median_pnl': np.median(pnls),
            'total_trades': sum(trades)
        }
    
    def clear(self):
        """Clear all wallet data."""
        self.wallets = []
        self.chain_wallets = {}
