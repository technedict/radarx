"""Rate limiting middleware."""

import threading
from datetime import datetime, timedelta, timezone
from typing import Dict


class RateLimiter:
    """Simple in-memory rate limiter with thread safety."""

    def __init__(self):
        self.requests: Dict[str, list] = {}
        self._lock = threading.Lock()

    def check_rate_limit(
        self, client_id: str, max_requests: int = 60, window_seconds: int = 60
    ) -> bool:
        """
        Check if client has exceeded rate limit.

        Args:
            client_id: Client identifier (IP, API key, etc.)
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds

        Returns:
            True if within rate limit, False otherwise
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            window_start = now - timedelta(seconds=window_seconds)

            # Initialize or clean old requests
            if client_id not in self.requests:
                self.requests[client_id] = []

            # Remove requests outside window
            self.requests[client_id] = [
                req_time for req_time in self.requests[client_id] if req_time > window_start
            ]

            # Check if limit exceeded
            if len(self.requests[client_id]) >= max_requests:
                return False

            # Add current request
            self.requests[client_id].append(now)
            return True
