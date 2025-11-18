"""Wallet clustering heuristics."""

import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)


class WalletClusterer:
    """Cluster wallets based on behavioral patterns and transaction flows."""

    def __init__(self):
        """Initialize wallet clusterer."""
        self.clusters: Dict[str, Set[str]] = defaultdict(set)

    def cluster_by_fund_flow(self, transfers: List[Dict[str, any]]) -> Dict[str, Set[str]]:
        """Cluster wallets by fund flow patterns.

        Wallets that frequently transfer to each other are likely
        controlled by the same entity.

        Args:
            transfers: List of transfer events

        Returns:
            Dict mapping cluster_id -> set of wallet addresses
        """
        # Build transfer graph
        transfer_graph = defaultdict(lambda: defaultdict(int))

        for transfer in transfers:
            from_addr = transfer.get("from_address", "").lower()
            to_addr = transfer.get("to_address", "").lower()

            if from_addr and to_addr:
                transfer_graph[from_addr][to_addr] += 1
                transfer_graph[to_addr][from_addr] += 1

        # Find connected components (simple DFS)
        visited = set()
        clusters = {}
        cluster_id = 0

        def dfs(wallet: str, cluster: Set[str]):
            """DFS to find connected wallets."""
            if wallet in visited:
                return
            visited.add(wallet)
            cluster.add(wallet)

            # Visit wallets with significant transfer volume
            for neighbor, count in transfer_graph[wallet].items():
                if count >= 2:  # At least 2 transfers
                    dfs(neighbor, cluster)

        for wallet in transfer_graph:
            if wallet not in visited:
                cluster = set()
                dfs(wallet, cluster)
                if len(cluster) > 1:  # Only keep clusters with multiple wallets
                    clusters[f"cluster_{cluster_id}"] = cluster
                    cluster_id += 1

        self.clusters = clusters
        return clusters

    def cluster_by_trading_patterns(
        self, wallet_trades: Dict[str, List[Dict[str, any]]]
    ) -> Dict[str, Set[str]]:
        """Cluster wallets by similar trading patterns.

        Args:
            wallet_trades: Dict mapping wallet -> list of trades

        Returns:
            Dict mapping cluster_id -> set of wallet addresses
        """
        # Extract trading signatures
        signatures = {}

        for wallet, trades in wallet_trades.items():
            if not trades:
                continue

            # Create simple signature based on:
            # - Average hold time
            # - Win rate
            # - Trade frequency

            avg_hold_time = (
                sum(t.get("hold_time", 0) for t in trades) / len(trades) if trades else 0
            )

            profitable = sum(1 for t in trades if t.get("pnl", 0) > 0)
            win_rate = profitable / len(trades) if trades else 0

            trade_freq = len(trades)

            signatures[wallet] = (avg_hold_time, win_rate, trade_freq)

        # Cluster similar signatures
        # (simplified - in production would use proper clustering algorithm)
        clusters = {}
        cluster_id = 0

        wallets = list(signatures.keys())
        visited = set()

        for i, wallet1 in enumerate(wallets):
            if wallet1 in visited:
                continue

            cluster = {wallet1}
            visited.add(wallet1)

            for wallet2 in wallets[i + 1 :]:
                if wallet2 in visited:
                    continue

                # Check similarity
                sig1 = signatures[wallet1]
                sig2 = signatures[wallet2]

                if self._signatures_similar(sig1, sig2):
                    cluster.add(wallet2)
                    visited.add(wallet2)

            if len(cluster) > 1:
                clusters[f"pattern_cluster_{cluster_id}"] = cluster
                cluster_id += 1

        return clusters

    def _signatures_similar(
        self,
        sig1: Tuple[float, float, float],
        sig2: Tuple[float, float, float],
        threshold: float = 0.3,
    ) -> bool:
        """Check if two trading signatures are similar.

        Args:
            sig1: First signature (hold_time, win_rate, freq)
            sig2: Second signature
            threshold: Similarity threshold

        Returns:
            True if similar
        """
        # Normalize and compute distance
        # (simplified - would use proper normalization in production)

        hold_diff = abs(sig1[0] - sig2[0]) / (max(sig1[0], sig2[0]) + 1)
        win_diff = abs(sig1[1] - sig2[1])
        freq_diff = abs(sig1[2] - sig2[2]) / (max(sig1[2], sig2[2]) + 1)

        avg_diff = (hold_diff + win_diff + freq_diff) / 3

        return avg_diff < threshold

    def get_cluster_for_wallet(self, wallet: str) -> Optional[str]:
        """Get cluster ID for a wallet.

        Args:
            wallet: Wallet address

        Returns:
            Cluster ID or None
        """
        wallet = wallet.lower()

        for cluster_id, wallets in self.clusters.items():
            if wallet in wallets:
                return cluster_id

        return None

    def get_related_wallets(self, wallet: str) -> Set[str]:
        """Get wallets in the same cluster.

        Args:
            wallet: Wallet address

        Returns:
            Set of related wallet addresses
        """
        cluster_id = self.get_cluster_for_wallet(wallet)

        if cluster_id:
            return self.clusters[cluster_id] - {wallet.lower()}

        return set()
