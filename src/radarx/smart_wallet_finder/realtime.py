"""
Real-time Data Streaming for Smart Wallet Finder

WebSocket support for live trade monitoring and signal updates.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import websockets

    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("websockets not available. Real-time streaming disabled.")


class RealTimeMonitor:
    """
    Real-time monitoring of smart wallet activity and signal updates.

    Uses WebSocket connections to DEX and blockchain APIs for live data.
    """

    def __init__(self):
        """Initialize real-time monitor."""
        self.active_connections = {}
        self.subscribers = {}
        self.is_running = False

    async def start_monitoring(
        self,
        token_address: str,
        chain: str,
        callback: Callable[[Dict[str, Any]], None],
    ):
        """
        Start monitoring token trades in real-time.

        Args:
            token_address: Token to monitor
            chain: Blockchain network
            callback: Function to call with each new trade
        """
        if not HAS_WEBSOCKETS:
            logger.error("WebSockets not available")
            return

        self.is_running = True

        if chain.lower() == "solana":
            await self._monitor_solana_trades(token_address, callback)
        elif chain.lower() in ["ethereum", "eth"]:
            await self._monitor_ethereum_trades(token_address, callback)
        else:
            logger.warning(f"Real-time monitoring not supported for chain: {chain}")

    async def stop_monitoring(self):
        """Stop all monitoring connections."""
        self.is_running = False

        for conn in self.active_connections.values():
            try:
                await conn.close()
            except Exception as e:
                logger.error(f"Error closing connection: {e}")

        self.active_connections.clear()

    async def _monitor_solana_trades(
        self,
        token_address: str,
        callback: Callable[[Dict[str, Any]], None],
    ):
        """
        Monitor Solana trades via WebSocket.

        Connects to Helius or QuickNode WebSocket for real-time updates.
        """
        # Helius WebSocket endpoint
        ws_url = "wss://atlas-mainnet.helius-rpc.com"

        try:
            async with websockets.connect(ws_url) as websocket:
                self.active_connections["solana"] = websocket

                # Subscribe to token account changes
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "logsSubscribe",
                    "params": [{"mentions": [token_address]}, {"commitment": "confirmed"}],
                }

                await websocket.send(json.dumps(subscribe_msg))

                logger.info(f"Subscribed to Solana trades for {token_address}")

                # Listen for updates
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)

                        data = json.loads(message)

                        # Parse and process trade
                        trade = self._parse_solana_log(data)

                        if trade:
                            callback(trade)

                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        await websocket.ping()

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

    async def _monitor_ethereum_trades(
        self,
        token_address: str,
        callback: Callable[[Dict[str, Any]], None],
    ):
        """
        Monitor Ethereum trades via WebSocket.

        Connects to Alchemy or Infura WebSocket for real-time updates.
        """
        # Alchemy WebSocket endpoint
        ws_url = "wss://eth-mainnet.g.alchemy.com/v2/demo"

        try:
            async with websockets.connect(ws_url) as websocket:
                self.active_connections["ethereum"] = websocket

                # Subscribe to token transfers
                subscribe_msg = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": [
                        "logs",
                        {
                            "address": token_address,
                            "topics": [
                                "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"  # Transfer event
                            ],
                        },
                    ],
                }

                await websocket.send(json.dumps(subscribe_msg))

                logger.info(f"Subscribed to Ethereum trades for {token_address}")

                # Listen for updates
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)

                        data = json.loads(message)

                        # Parse and process trade
                        trade = self._parse_ethereum_log(data)

                        if trade:
                            callback(trade)

                    except asyncio.TimeoutError:
                        # Send ping to keep connection alive
                        pong = await websocket.ping()
                        await pong

                    except Exception as e:
                        logger.error(f"Error processing message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")

    def _parse_solana_log(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Solana log data into trade format."""
        try:
            if "params" not in data or "result" not in data["params"]:
                return None

            result = data["params"]["result"]

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chain": "solana",
                "signature": result.get("signature"),
                "logs": result.get("logs", []),
                "type": "live_trade",
            }

        except Exception as e:
            logger.error(f"Error parsing Solana log: {e}")
            return None

    def _parse_ethereum_log(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse Ethereum log data into trade format."""
        try:
            if "params" not in data or "result" not in data["params"]:
                return None

            result = data["params"]["result"]

            return {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "chain": "ethereum",
                "transaction_hash": result.get("transactionHash"),
                "from": result.get("from"),
                "to": result.get("to"),
                "type": "live_trade",
            }

        except Exception as e:
            logger.error(f"Error parsing Ethereum log: {e}")
            return None


class LiveSignalUpdater:
    """
    Updates smart wallet signals in real-time as new trades arrive.

    Maintains incremental signal calculations for efficiency.
    """

    def __init__(self):
        """Initialize live signal updater."""
        self.wallet_signals = {}
        self.trade_buffer = []
        self.update_callbacks = []

    def register_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Register callback for signal updates."""
        self.update_callbacks.append(callback)

    async def process_live_trade(self, trade: Dict[str, Any]):
        """
        Process a live trade and update signals.

        Args:
            trade: New trade data from WebSocket
        """
        self.trade_buffer.append(trade)

        # Extract wallet
        wallet = trade.get("buyer") or trade.get("from") or trade.get("wallet")

        if not wallet:
            return

        # Update signals incrementally
        updated_signals = self._update_wallet_signals(wallet, trade)

        # Notify subscribers
        for callback in self.update_callbacks:
            try:
                callback(
                    {
                        "wallet": wallet,
                        "signals": updated_signals,
                        "trade": trade,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _update_wallet_signals(
        self,
        wallet: str,
        new_trade: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Incrementally update wallet signals with new trade.

        Returns:
            Updated signal values
        """
        if wallet not in self.wallet_signals:
            self.wallet_signals[wallet] = {
                "trade_count": 0,
                "total_volume": 0.0,
                "buy_count": 0,
                "sell_count": 0,
                "last_trade_time": None,
            }

        signals = self.wallet_signals[wallet]

        # Update counts
        signals["trade_count"] += 1
        signals["total_volume"] += new_trade.get("amount_usd", 0)

        if new_trade.get("side") == "buy":
            signals["buy_count"] += 1
        else:
            signals["sell_count"] += 1

        signals["last_trade_time"] = new_trade.get("timestamp")

        return signals


class EventDrivenArchitecture:
    """
    Event-driven architecture for real-time smart wallet analysis.

    Coordinates between data streams, signal updates, and notifications.
    """

    def __init__(self):
        """Initialize event-driven architecture."""
        self.monitor = RealTimeMonitor()
        self.signal_updater = LiveSignalUpdater()
        self.event_handlers = {}

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def start(self, token_address: str, chain: str):
        """Start event-driven monitoring."""
        # Set up signal update callback
        self.signal_updater.register_callback(self._on_signal_update)

        # Start real-time monitoring
        await self.monitor.start_monitoring(
            token_address,
            chain,
            self._on_new_trade,
        )

    async def stop(self):
        """Stop event-driven monitoring."""
        await self.monitor.stop_monitoring()

    def _on_new_trade(self, trade: Dict[str, Any]):
        """Handle new trade event."""
        # Process trade through signal updater
        asyncio.create_task(self.signal_updater.process_live_trade(trade))

        # Trigger handlers
        self._trigger_handlers("new_trade", trade)

    def _on_signal_update(self, update: Dict[str, Any]):
        """Handle signal update event."""
        # Trigger handlers
        self._trigger_handlers("signal_update", update)

    def _trigger_handlers(self, event_type: str, data: Dict[str, Any]):
        """Trigger all handlers for event type."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Handler error for {event_type}: {e}")
