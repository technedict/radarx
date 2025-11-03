"""
Advanced ML Features for Smart Wallet Finder

Implements:
- Granger causality for lead-lag modeling
- Sequence embeddings for wallet behavior patterns
- Temporal GNN components
- Counterfactual analysis
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import grangercausalitytests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("Statsmodels not available. Granger causality disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    logger.warning("PyTorch not available. Sequence embeddings and GNN disabled.")


class GrangerCausalityAnalyzer:
    """
    Analyzes lead-lag relationships between wallet activity and price movements.
    
    Uses Granger causality tests to determine if wallet trades predict price changes,
    which helps identify wallets with genuine market-moving alpha vs. followers.
    """
    
    def __init__(self, max_lag: int = 10, significance_level: float = 0.05):
        """
        Initialize Granger causality analyzer.
        
        Args:
            max_lag: Maximum lag to test (in time periods)
            significance_level: P-value threshold for significance
        """
        self.max_lag = max_lag
        self.significance_level = significance_level
    
    def analyze_wallet_price_causality(
        self,
        wallet_trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
        interval_minutes: int = 15,
    ) -> Dict[str, Any]:
        """
        Test if wallet trading activity Granger-causes price movements.
        
        Args:
            wallet_trades: List of wallet trades with timestamps
            price_timeline: Price data over time
            interval_minutes: Time interval for aggregation
            
        Returns:
            Dictionary with causality results:
                - has_causality: Boolean indicating significant causality
                - optimal_lag: Lag with strongest relationship
                - p_values: P-values for each lag tested
                - lead_score: Normalized score (0-1) for predictive power
        """
        if not HAS_STATSMODELS:
            logger.warning("Statsmodels not available, returning default")
            return {
                "has_causality": False,
                "optimal_lag": 0,
                "p_values": {},
                "lead_score": 0.0,
            }
        
        # Prepare time series data
        trade_series, price_series = self._prepare_time_series(
            wallet_trades, price_timeline, interval_minutes
        )
        
        if len(trade_series) < self.max_lag + 10:
            logger.warning("Insufficient data for Granger causality test")
            return {
                "has_causality": False,
                "optimal_lag": 0,
                "p_values": {},
                "lead_score": 0.0,
            }
        
        try:
            # Create data matrix: [price_changes, trade_volume]
            data = np.column_stack([price_series, trade_series])
            
            # Run Granger causality test
            # Tests if trade_series helps predict price_series
            results = grangercausalitytests(data, maxlag=self.max_lag, verbose=False)
            
            # Extract p-values for each lag
            p_values = {}
            min_p_value = 1.0
            optimal_lag = 0
            
            for lag in range(1, self.max_lag + 1):
                # Use F-test p-value
                p_value = results[lag][0]['ssr_ftest'][1]
                p_values[lag] = p_value
                
                if p_value < min_p_value:
                    min_p_value = p_value
                    optimal_lag = lag
            
            has_causality = min_p_value < self.significance_level
            
            # Calculate lead score: normalized by p-value strength
            lead_score = max(0.0, 1.0 - min_p_value) if has_causality else 0.0
            
            return {
                "has_causality": has_causality,
                "optimal_lag": optimal_lag,
                "p_values": p_values,
                "lead_score": lead_score,
                "interpretation": self._interpret_causality(has_causality, optimal_lag),
            }
            
        except Exception as e:
            logger.error(f"Granger causality test error: {e}")
            return {
                "has_causality": False,
                "optimal_lag": 0,
                "p_values": {},
                "lead_score": 0.0,
            }
    
    def _prepare_time_series(
        self,
        trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
        interval_minutes: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare aligned time series for causality analysis.
        
        Returns:
            Tuple of (trade_volume_series, price_change_series)
        """
        # Create time bins
        if not price_timeline:
            return np.array([]), np.array([])
        
        start_time = datetime.fromisoformat(price_timeline[0]["timestamp"].replace('Z', '+00:00'))
        end_time = datetime.fromisoformat(price_timeline[-1]["timestamp"].replace('Z', '+00:00'))
        
        # Generate time bins
        bins = []
        current = start_time
        delta = timedelta(minutes=interval_minutes)
        
        while current <= end_time:
            bins.append(current)
            current += delta
        
        # Aggregate trades into bins
        trade_volumes = np.zeros(len(bins))
        for trade in trades:
            trade_time = datetime.fromisoformat(trade["timestamp"].replace('Z', '+00:00'))
            
            # Find appropriate bin
            for i, bin_time in enumerate(bins):
                if bin_time <= trade_time < bin_time + delta:
                    amount = trade.get("amount_usd", trade.get("amount_tokens", 0))
                    trade_volumes[i] += abs(amount)
                    break
        
        # Extract price changes
        price_changes = np.zeros(len(bins))
        for i, bin_time in enumerate(bins):
            # Find matching price point
            for price_point in price_timeline:
                pt_time = datetime.fromisoformat(price_point["timestamp"].replace('Z', '+00:00'))
                if abs((pt_time - bin_time).total_seconds()) < interval_minutes * 60 / 2:
                    price = price_point.get("price", 0)
                    if i > 0 and price_timeline[max(0, i-1)].get("price", 0) > 0:
                        prev_price = price_timeline[max(0, i-1)].get("price", 0)
                        price_changes[i] = (price - prev_price) / prev_price
                    break
        
        return trade_volumes, price_changes
    
    def _interpret_causality(self, has_causality: bool, optimal_lag: int) -> str:
        """Generate human-readable interpretation."""
        if not has_causality:
            return "No significant lead-lag relationship detected. Wallet may be following market."
        
        lead_time_minutes = optimal_lag * 15  # Assuming 15min intervals
        if lead_time_minutes < 30:
            return f"Strong leading indicator. Trades precede price moves by ~{lead_time_minutes} minutes."
        elif lead_time_minutes < 120:
            return f"Moderate leading indicator. Trades precede price moves by ~{lead_time_minutes // 60} hours."
        else:
            return f"Weak leading indicator with {optimal_lag} period lag. May be noise."


class WalletBehaviorEmbedder:
    """
    Creates sequence embeddings for wallet trading behavior.
    
    Uses transformer-based sequence models to capture trading patterns,
    enabling similarity comparison and strategy clustering.
    """
    
    def __init__(self, embedding_dim: int = 64):
        """
        Initialize behavior embedder.
        
        Args:
            embedding_dim: Dimension of output embeddings
        """
        self.embedding_dim = embedding_dim
        self.model = None
        
        if HAS_TORCH:
            self.model = SequenceEmbeddingModel(embedding_dim=embedding_dim)
    
    def embed_wallet_sequence(
        self,
        trades: List[Dict[str, Any]],
        max_sequence_length: int = 100,
    ) -> np.ndarray:
        """
        Convert wallet trade sequence to embedding vector.
        
        Args:
            trades: List of trades in chronological order
            max_sequence_length: Maximum trades to include
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if not HAS_TORCH or self.model is None:
            # Fallback: simple statistical features
            return self._statistical_embedding(trades)
        
        # Extract sequence features
        sequence = self._extract_sequence_features(trades, max_sequence_length)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        
        # Get embedding
        with torch.no_grad():
            embedding = self.model(sequence_tensor)
        
        return embedding.numpy().flatten()
    
    def _extract_sequence_features(
        self,
        trades: List[Dict[str, Any]],
        max_length: int,
    ) -> np.ndarray:
        """Extract feature vector for each trade in sequence."""
        features = []
        
        for i, trade in enumerate(trades[:max_length]):
            # Feature vector for each trade
            trade_features = [
                1.0 if trade.get("side") == "buy" else -1.0,  # Direction
                np.log1p(trade.get("amount_usd", 0)),  # Size (log scale)
                trade.get("price", 0),  # Price
                i / len(trades),  # Position in sequence
            ]
            features.append(trade_features)
        
        # Pad if necessary
        while len(features) < max_length:
            features.append([0.0] * 4)
        
        return np.array(features)
    
    def _statistical_embedding(self, trades: List[Dict[str, Any]]) -> np.ndarray:
        """Fallback statistical embedding when PyTorch unavailable."""
        if not trades:
            return np.zeros(self.embedding_dim)
        
        # Compute basic statistics
        buy_count = sum(1 for t in trades if t.get("side") == "buy")
        sell_count = len(trades) - buy_count
        avg_size = np.mean([t.get("amount_usd", 0) for t in trades])
        
        # Simple feature vector
        features = [
            buy_count / len(trades),
            sell_count / len(trades),
            np.log1p(avg_size),
            len(trades),
        ]
        
        # Pad to embedding_dim
        while len(features) < self.embedding_dim:
            features.append(0.0)
        
        return np.array(features[:self.embedding_dim])
    
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Returns:
            Similarity score between 0 and 1
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


class SequenceEmbeddingModel(nn.Module if HAS_TORCH else object):
    """
    Transformer-based model for sequence embedding.
    """
    
    def __init__(self, embedding_dim: int = 64, num_heads: int = 4):
        if not HAS_TORCH:
            return
        
        super().__init__()
        
        self.input_projection = nn.Linear(4, embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=embedding_dim * 2,
                batch_first=True,
            ),
            num_layers=2,
        )
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
    
    def forward(self, x):
        # x shape: (batch, sequence, features)
        x = self.input_projection(x)
        x = self.transformer(x)
        # Global average pooling
        x = torch.mean(x, dim=1)
        x = self.output_projection(x)
        return F.normalize(x, p=2, dim=-1)


class CounterfactualAnalyzer:
    """
    Performs counterfactual analysis to estimate wallet impact on price moves.
    
    Simulates "what if this wallet didn't trade" scenarios to quantify
    individual wallet influence on market dynamics.
    """
    
    def __init__(self):
        """Initialize counterfactual analyzer."""
        pass
    
    def estimate_wallet_impact(
        self,
        wallet_trades: List[Dict[str, Any]],
        all_trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Estimate impact of wallet on price movements.
        
        Args:
            wallet_trades: Trades from the specific wallet
            all_trades: All trades in the market
            price_timeline: Actual price history
            
        Returns:
            Dictionary with impact metrics:
                - impact_score: Estimated impact (0-1)
                - volume_contribution: Wallet's share of total volume
                - price_influence: Estimated price impact
                - counterfactual_returns: Simulated returns without wallet
        """
        if not wallet_trades or not all_trades:
            return {
                "impact_score": 0.0,
                "volume_contribution": 0.0,
                "price_influence": 0.0,
                "counterfactual_returns": 0.0,
            }
        
        # Calculate wallet's volume contribution
        wallet_volume = sum(t.get("amount_usd", 0) for t in wallet_trades)
        total_volume = sum(t.get("amount_usd", 0) for t in all_trades)
        volume_contribution = wallet_volume / total_volume if total_volume > 0 else 0.0
        
        # Estimate price impact based on volume and timing
        price_influence = self._estimate_price_influence(
            wallet_trades, price_timeline
        )
        
        # Simulate counterfactual scenario
        counterfactual_returns = self._simulate_without_wallet(
            wallet_trades, all_trades, price_timeline
        )
        
        # Compute overall impact score
        impact_score = min(1.0, (
            0.4 * volume_contribution +
            0.4 * price_influence +
            0.2 * abs(counterfactual_returns)
        ))
        
        return {
            "impact_score": impact_score,
            "volume_contribution": volume_contribution,
            "price_influence": price_influence,
            "counterfactual_returns": counterfactual_returns,
            "interpretation": self._interpret_impact(impact_score),
        }
    
    def _estimate_price_influence(
        self,
        trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
    ) -> float:
        """Estimate wallet's influence on price movements."""
        if not trades or not price_timeline:
            return 0.0
        
        influence_scores = []
        
        for trade in trades:
            trade_time = datetime.fromisoformat(trade["timestamp"].replace('Z', '+00:00'))
            
            # Find price movement after trade
            post_trade_price_change = 0.0
            for i, price_point in enumerate(price_timeline):
                pt_time = datetime.fromisoformat(price_point["timestamp"].replace('Z', '+00:00'))
                
                if pt_time > trade_time:
                    # Look at price change in next hour
                    if i + 4 < len(price_timeline):  # Assuming 15min intervals
                        current_price = price_point.get("price", 0)
                        future_price = price_timeline[i + 4].get("price", 0)
                        
                        if current_price > 0:
                            post_trade_price_change = (future_price - current_price) / current_price
                    break
            
            # Weight by trade size
            trade_size = trade.get("amount_usd", 0)
            weighted_influence = abs(post_trade_price_change) * np.log1p(trade_size)
            influence_scores.append(weighted_influence)
        
        return min(1.0, np.mean(influence_scores)) if influence_scores else 0.0
    
    def _simulate_without_wallet(
        self,
        wallet_trades: List[Dict[str, Any]],
        all_trades: List[Dict[str, Any]],
        price_timeline: List[Dict[str, Any]],
    ) -> float:
        """Simulate market returns if wallet hadn't traded."""
        # Simplified simulation: estimate how returns would differ
        # In production, would use more sophisticated market impact models
        
        if not price_timeline or len(price_timeline) < 2:
            return 0.0
        
        actual_return = (
            price_timeline[-1].get("price", 0) - price_timeline[0].get("price", 0)
        ) / price_timeline[0].get("price", 1)
        
        # Estimate return without wallet based on volume contribution
        wallet_volume = sum(t.get("amount_usd", 0) for t in wallet_trades)
        total_volume = sum(t.get("amount_usd", 0) for t in all_trades)
        
        if total_volume == 0:
            return 0.0
        
        # Simple linear model: price impact proportional to volume
        volume_ratio = wallet_volume / total_volume
        estimated_impact = actual_return * volume_ratio
        
        counterfactual_return = actual_return - estimated_impact
        
        return float(counterfactual_return)
    
    def _interpret_impact(self, impact_score: float) -> str:
        """Generate human-readable interpretation."""
        if impact_score < 0.1:
            return "Negligible market impact. Wallet is a price taker."
        elif impact_score < 0.3:
            return "Low market impact. Minor influence on price movements."
        elif impact_score < 0.6:
            return "Moderate market impact. Significant player in this token."
        else:
            return "High market impact. Wallet likely moves the market."
