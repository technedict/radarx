"""
Enhanced API utilities for better error handling, validation, and responses.

Improvements:
- Structured error responses with error codes
- Request validation with detailed feedback
- Response streaming for large datasets
- Better logging and tracing
"""

import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class ErrorCode:
    """Standardized error codes for API responses."""

    # Client errors (4xx)
    INVALID_INPUT = "INVALID_INPUT"
    MISSING_PARAMETER = "MISSING_PARAMETER"
    INVALID_TOKEN_ADDRESS = "INVALID_TOKEN_ADDRESS"
    INVALID_CHAIN = "INVALID_CHAIN"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    UNAUTHORIZED = "UNAUTHORIZED"

    # Server errors (5xx)
    INTERNAL_ERROR = "INTERNAL_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    DATA_SOURCE_ERROR = "DATA_SOURCE_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    CACHE_ERROR = "CACHE_ERROR"


class APIError(BaseModel):
    """Structured error response."""

    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    timestamp: str


class APIResponse(BaseModel):
    """Standardized API response wrapper."""

    success: bool
    data: Optional[Any] = None
    error: Optional[APIError] = None
    metadata: Optional[Dict[str, Any]] = None
    request_id: str


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = status.HTTP_400_BAD_REQUEST,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """
    Create standardized error response.

    Args:
        error_code: Error code from ErrorCode class
        message: Human-readable error message
        status_code: HTTP status code
        details: Additional error details
        request_id: Request tracking ID

    Returns:
        JSONResponse with error details
    """
    from datetime import datetime, timezone

    error = APIError(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id or str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    response = APIResponse(success=False, error=error, data=None, request_id=error.request_id)

    return JSONResponse(status_code=status_code, content=response.model_dump())


def create_success_response(
    data: Any, metadata: Optional[Dict[str, Any]] = None, request_id: Optional[str] = None
) -> APIResponse:
    """
    Create standardized success response.

    Args:
        data: Response data
        metadata: Optional metadata (timing, pagination, etc.)
        request_id: Request tracking ID

    Returns:
        APIResponse with data
    """
    return APIResponse(
        success=True,
        data=data,
        error=None,
        metadata=metadata,
        request_id=request_id or str(uuid.uuid4()),
    )


class RequestValidator:
    """Enhanced request validation with detailed feedback."""

    @staticmethod
    def validate_token_address(address: str, chain: str) -> None:
        """
        Validate token address format.

        Args:
            address: Token address
            chain: Blockchain network

        Raises:
            HTTPException: If validation fails
        """
        if not address:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Token address is required",
            )

        # Chain-specific validation
        if chain in ["ethereum", "bsc", "polygon"]:
            # EVM address validation
            if not address.startswith("0x") or len(address) != 42:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid EVM address format for {chain}. Expected 0x followed by 40 hex characters.",
                )
        elif chain == "solana":
            # Solana address validation (base58, typically 32-44 chars)
            if len(address) < 32 or len(address) > 44:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid Solana address format. Expected 32-44 character base58 string.",
                )

    @staticmethod
    def validate_chain(chain: str) -> None:
        """
        Validate blockchain network.

        Args:
            chain: Blockchain identifier

        Raises:
            HTTPException: If chain not supported
        """
        supported_chains = [
            "ethereum",
            "bsc",
            "polygon",
            "solana",
            "arbitrum",
            "optimism",
            "avalanche",
        ]

        if chain.lower() not in supported_chains:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported chain: {chain}. Supported chains: {', '.join(supported_chains)}",
            )

    @staticmethod
    def validate_pagination(page: int, per_page: int, max_per_page: int = 1000) -> None:
        """
        Validate pagination parameters.

        Args:
            page: Page number
            per_page: Items per page
            max_per_page: Maximum allowed items per page

        Raises:
            HTTPException: If parameters invalid
        """
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page number must be >= 1",
            )

        if per_page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Items per page must be >= 1",
            )

        if per_page > max_per_page:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Items per page cannot exceed {max_per_page}",
            )

    @staticmethod
    def validate_time_horizons(horizons: List[str]) -> None:
        """
        Validate time horizon parameters.

        Args:
            horizons: List of time horizon strings

        Raises:
            HTTPException: If horizons invalid
        """
        valid_horizons = ["1h", "4h", "24h", "7d", "30d", "90d"]

        for horizon in horizons:
            if horizon not in valid_horizons:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid horizon: {horizon}. Valid horizons: {', '.join(valid_horizons)}",
                )


async def stream_response(data_generator, chunk_size: int = 100):
    """
    Stream large responses in chunks.

    Args:
        data_generator: Async generator yielding data items
        chunk_size: Number of items per chunk

    Yields:
        JSON chunks
    """
    import json

    yield "["

    first = True
    chunk = []

    async for item in data_generator:
        chunk.append(item)

        if len(chunk) >= chunk_size:
            if not first:
                yield ","
            yield json.dumps(chunk)
            chunk = []
            first = False

    # Yield remaining items
    if chunk:
        if not first:
            yield ","
        yield json.dumps(chunk)

    yield "]"


class RequestLogger:
    """Enhanced request logging with structured format."""

    @staticmethod
    async def log_request(request: Request, start_time: float) -> Dict[str, Any]:
        """
        Log API request with details.

        Args:
            request: FastAPI request object
            start_time: Request start time

        Returns:
            Request metadata
        """
        request_id = str(uuid.uuid4())

        # Extract request info
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path
        query_params = dict(request.query_params)

        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "client_ip": client_ip,
                "query_params": query_params,
            },
        )

        return {"request_id": request_id, "start_time": start_time}

    @staticmethod
    def log_response(
        request_id: str,
        status_code: int,
        start_time: float,
        error: Optional[str] = None,
    ):
        """
        Log API response with timing.

        Args:
            request_id: Request tracking ID
            status_code: HTTP status code
            start_time: Request start time
            error: Optional error message
        """
        duration_ms = (time.time() - start_time) * 1000

        log_data = {
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": round(duration_ms, 2),
        }

        if error:
            log_data["error"] = error
            logger.error("Request failed", extra=log_data)
        else:
            logger.info("Request completed", extra=log_data)


class RateLimitExceeded(HTTPException):
    """Custom exception for rate limit errors."""

    def __init__(self, retry_after: int):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )


class CircuitBreaker:
    """
    Circuit breaker pattern for external service calls.

    Prevents cascading failures by stopping calls to failing services.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            expected_exception: Exception type to catch
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open

    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to call
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            # Check if recovery timeout passed
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = "half-open"
                logger.info("Circuit breaker entering half-open state")
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Service temporarily unavailable (circuit breaker open)",
                )

        try:
            result = await func(*args, **kwargs)

            # Success - reset if in half-open
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
                logger.info("Circuit breaker closed")

            return result

        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.error(f"Circuit breaker opened after {self.failure_count} failures")

            raise e


class APIMetrics:
    """Track API metrics for monitoring."""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_duration_ms = 0.0
        self.endpoint_stats: Dict[str, Dict[str, Any]] = {}

    def record_request(self, endpoint: str, duration_ms: float, status_code: int):
        """Record API request metrics."""
        self.request_count += 1
        self.total_duration_ms += duration_ms

        if status_code >= 400:
            self.error_count += 1

        # Per-endpoint stats
        if endpoint not in self.endpoint_stats:
            self.endpoint_stats[endpoint] = {
                "count": 0,
                "errors": 0,
                "total_duration": 0.0,
            }

        stats = self.endpoint_stats[endpoint]
        stats["count"] += 1
        stats["total_duration"] += duration_ms

        if status_code >= 400:
            stats["errors"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get current metrics."""
        avg_duration = self.total_duration_ms / self.request_count if self.request_count > 0 else 0
        error_rate = self.error_count / self.request_count if self.request_count > 0 else 0

        endpoint_stats = {}
        for endpoint, stats in self.endpoint_stats.items():
            endpoint_stats[endpoint] = {
                "count": stats["count"],
                "errors": stats["errors"],
                "error_rate": stats["errors"] / stats["count"] if stats["count"] > 0 else 0,
                "avg_duration_ms": (
                    stats["total_duration"] / stats["count"] if stats["count"] > 0 else 0
                ),
            }

        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": error_rate,
            "avg_duration_ms": avg_duration,
            "endpoints": endpoint_stats,
        }


# Global metrics instance
api_metrics = APIMetrics()
