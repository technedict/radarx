"""
Related Wallet Finder

Discovers related wallets through fund flow analysis and pattern similarity.
"""

from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set

import numpy as np


class RelatedWalletFinder:
    """Finds related wallets through various analysis methods."""

    def __init__(self):
        """Initialize related wallet finder."""
        pass

    def find_by_fund_flow(
        self,
        wallet_address: str,
        transfers: List[Dict[str, Any]],
        min_amount: float = 0,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """
        Find wallets connected by fund transfers.

        Args:
            wallet_address: Starting wallet
            transfers: List of transfer events with 'from', 'to', 'amount'
            min_amount: Minimum transfer amount to consider
            max_depth: Maximum depth of graph traversal

        Returns:
            List of related wallets with relationship metadata
        """
        # Build transfer graph
        graph = defaultdict(list)
        for transfer in transfers:
            from_addr = transfer.get("from", "").lower()
            to_addr = transfer.get("to", "").lower()
            amount = transfer.get("amount", 0)

            if amount >= min_amount:
                graph[from_addr].append(
                    {"wallet": to_addr, "amount": amount, "timestamp": transfer.get("timestamp")}
                )
                # Also add reverse for bidirectional search
                graph[to_addr].append(
                    {"wallet": from_addr, "amount": amount, "timestamp": transfer.get("timestamp")}
                )

        # BFS to find connected wallets
        wallet_lower = wallet_address.lower()
        visited = {wallet_lower}
        queue = deque([(wallet_lower, 0)])  # (wallet, depth)
        related = []

        while queue:
            current_wallet, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for connection in graph.get(current_wallet, []):
                connected_wallet = connection["wallet"]

                if connected_wallet not in visited:
                    visited.add(connected_wallet)
                    queue.append((connected_wallet, depth + 1))

                    related.append(
                        {
                            "wallet": connected_wallet,
                            "relationship_type": "fund_transfer",
                            "depth": depth + 1,
                            "amount": connection["amount"],
                            "strength": min(1.0, connection["amount"] / 10000),  # Normalize
                        }
                    )

        return related

    def find_by_pattern_similarity(
        self,
        wallet_address: str,
        all_wallets_features: Dict[str, np.ndarray],
        threshold: float = 0.8,
    ) -> List[Dict[str, Any]]:
        """
        Find wallets with similar trading patterns.

        Args:
            wallet_address: Target wallet
            all_wallets_features: Dict mapping wallet addresses to feature vectors
            threshold: Minimum similarity score (0-1)

        Returns:
            List of similar wallets with similarity scores
        """
        if wallet_address not in all_wallets_features:
            return []

        target_features = all_wallets_features[wallet_address]
        similar_wallets = []

        for other_wallet, other_features in all_wallets_features.items():
            if other_wallet == wallet_address:
                continue

            # Calculate cosine similarity
            similarity = self._cosine_similarity(target_features, other_features)

            if similarity >= threshold:
                similar_wallets.append(
                    {
                        "wallet": other_wallet,
                        "relationship_type": "pattern_similarity",
                        "strength": similarity,
                        "similarity_score": similarity,
                    }
                )

        # Sort by similarity descending
        similar_wallets.sort(key=lambda x: x["strength"], reverse=True)

        return similar_wallets

    def find_by_token_overlap(
        self, wallet_address: str, all_wallet_tokens: Dict[str, Set[str]], min_overlap: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find wallets that trade the same tokens.

        Args:
            wallet_address: Target wallet
            all_wallet_tokens: Dict mapping wallets to sets of token addresses
            min_overlap: Minimum number of overlapping tokens

        Returns:
            List of wallets with token overlap
        """
        if wallet_address not in all_wallet_tokens:
            return []

        target_tokens = all_wallet_tokens[wallet_address]
        related = []

        for other_wallet, other_tokens in all_wallet_tokens.items():
            if other_wallet == wallet_address:
                continue

            # Calculate token overlap
            overlap = target_tokens & other_tokens  # Set intersection
            overlap_count = len(overlap)

            if overlap_count >= min_overlap:
                # Jaccard similarity
                union = target_tokens | other_tokens
                jaccard = overlap_count / len(union) if union else 0

                related.append(
                    {
                        "wallet": other_wallet,
                        "relationship_type": "token_overlap",
                        "strength": jaccard,
                        "overlap_count": overlap_count,
                        "common_tokens": list(overlap)[:10],  # Top 10
                    }
                )

        # Sort by overlap count descending
        related.sort(key=lambda x: x["overlap_count"], reverse=True)

        return related

    def find_by_temporal_correlation(
        self,
        wallet_address: str,
        all_wallet_trades: Dict[str, List[Dict[str, Any]]],
        time_window_minutes: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find wallets that trade at similar times (possible copy trading).

        Args:
            wallet_address: Target wallet
            all_wallet_trades: Dict mapping wallets to trade lists
            time_window_minutes: Time window for considering trades correlated

        Returns:
            List of temporally correlated wallets
        """
        if wallet_address not in all_wallet_trades:
            return []

        from datetime import datetime, timedelta

        target_trades = all_wallet_trades[wallet_address]
        related = []

        for other_wallet, other_trades in all_wallet_trades.items():
            if other_wallet == wallet_address or not other_trades:
                continue

            # Count correlated trades
            correlated_count = 0

            for target_trade in target_trades:
                target_time = datetime.fromisoformat(
                    target_trade["timestamp"].replace("Z", "+00:00")
                )
                target_token = target_trade.get("token")

                for other_trade in other_trades:
                    other_time = datetime.fromisoformat(
                        other_trade["timestamp"].replace("Z", "+00:00")
                    )
                    other_token = other_trade.get("token")

                    # Check if same token traded within time window
                    time_diff = abs((target_time - other_time).total_seconds() / 60)

                    if target_token == other_token and time_diff <= time_window_minutes:
                        correlated_count += 1
                        break  # Count each target trade only once

            if correlated_count > 0:
                correlation_ratio = correlated_count / len(target_trades)

                if correlation_ratio > 0.1:  # >10% correlation
                    related.append(
                        {
                            "wallet": other_wallet,
                            "relationship_type": "temporal_correlation",
                            "strength": correlation_ratio,
                            "correlated_trades": correlated_count,
                            "total_trades": len(target_trades),
                        }
                    )

        # Sort by correlation strength
        related.sort(key=lambda x: x["strength"], reverse=True)

        return related

    def get_related_wallets(
        self,
        wallet_address: str,
        transfers: Optional[List[Dict[str, Any]]] = None,
        wallet_features: Optional[Dict[str, np.ndarray]] = None,
        wallet_tokens: Optional[Dict[str, Set[str]]] = None,
        wallet_trades: Optional[Dict[str, List[Dict[str, Any]]]] = None,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Find all related wallets using available data sources.

        Args:
            wallet_address: Target wallet
            transfers: Transfer data for fund flow analysis
            wallet_features: Feature vectors for pattern similarity
            wallet_tokens: Token sets for overlap analysis
            wallet_trades: Trade history for temporal correlation
            threshold: Minimum relationship strength

        Returns:
            Consolidated list of related wallets
        """
        all_related = {}

        # Fund flow analysis
        if transfers:
            fund_related = self.find_by_fund_flow(wallet_address, transfers)
            for rel in fund_related:
                addr = rel["wallet"]
                if addr not in all_related or rel["strength"] > all_related[addr]["strength"]:
                    all_related[addr] = rel

        # Pattern similarity
        if wallet_features:
            pattern_related = self.find_by_pattern_similarity(
                wallet_address, wallet_features, threshold=threshold
            )
            for rel in pattern_related:
                addr = rel["wallet"]
                if addr not in all_related or rel["strength"] > all_related[addr]["strength"]:
                    all_related[addr] = rel

        # Token overlap
        if wallet_tokens:
            token_related = self.find_by_token_overlap(wallet_address, wallet_tokens)
            for rel in token_related:
                addr = rel["wallet"]
                if addr not in all_related or rel["strength"] > all_related[addr]["strength"]:
                    all_related[addr] = rel

        # Temporal correlation
        if wallet_trades:
            temporal_related = self.find_by_temporal_correlation(wallet_address, wallet_trades)
            for rel in temporal_related:
                addr = rel["wallet"]
                if addr not in all_related or rel["strength"] > all_related[addr]["strength"]:
                    all_related[addr] = rel

        # Filter by threshold and sort
        filtered = [rel for rel in all_related.values() if rel["strength"] >= threshold]
        filtered.sort(key=lambda x: x["strength"], reverse=True)

        return filtered

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(vec1) == 0 or len(vec2) == 0:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
