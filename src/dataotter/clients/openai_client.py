import asyncio
from collections.abc import Callable, Mapping
from typing import Any, Self

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

# Global rate limiter for OpenAI API requests
_DEFAULT_MAX_OPENAI_REQUESTS = 10
_OPENAI_RATE_LIMITER: asyncio.Semaphore | None = None


def get_openai_rate_limiter(max_requests: int | None = None) -> asyncio.Semaphore:
    """Get or create the global OpenAI rate limiter semaphore.

    Args:
        max_requests: Override the default limit. Only applied on first call.

    Returns:
        The global rate limiter semaphore.
    """
    global _OPENAI_RATE_LIMITER
    if _OPENAI_RATE_LIMITER is None:
        limit = max_requests if max_requests is not None else _DEFAULT_MAX_OPENAI_REQUESTS
        if limit <= 0:
            raise ValueError("max_requests must be > 0")
        _OPENAI_RATE_LIMITER = asyncio.Semaphore(limit)

    return _OPENAI_RATE_LIMITER


def set_openai_rate_limit(max_requests: int) -> None:
    """Set the global OpenAI rate limit.

    Must be called before any OpenAIClient is used.

    Args:
        max_requests: Maximum number of concurrent OpenAI API requests.
    """
    if max_requests <= 0:
        raise ValueError("max_requests must be > 0")

    global _OPENAI_RATE_LIMITER
    _OPENAI_RATE_LIMITER = asyncio.Semaphore(max_requests)


class OpenAIClient:
    """
    Thin async wrapper over `openai.AsyncOpenAI` with retry/throttle helpers.

    All OpenAI API requests are rate-limited by a global semaphore (default: 10 concurrent
    requests). This ensures multiple clients and workflows coordinate to respect OpenAI's
    rate limits.

    Request kwargs are passed directly to the OpenAI SDK methods.
    Raw SDK responses are returned unchanged.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        max_retries: int = 5,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        client_kwargs: Mapping[str, Any] | None = None,
        rate_limiter: asyncio.Semaphore | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")

        if initial_backoff <= 0:
            raise ValueError("initial_backoff must be > 0")

        if max_backoff < initial_backoff:
            raise ValueError("max_backoff must be >= initial_backoff")

        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff

        kwargs = dict(client_kwargs or {})
        if timeout is not None:
            kwargs.setdefault("timeout", timeout)
        if api_key is not None:
            kwargs["api_key"] = api_key

        if base_url is not None:
            kwargs["base_url"] = base_url

        self._client = AsyncOpenAI(**kwargs)
        self._rate_limiter = rate_limiter or get_openai_rate_limiter()
        self._throttle_lock = asyncio.Lock()
        self._throttle_until = 0.0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.close()

    async def responses_create(
        self,
        model: str,
        input: list[dict],
        store: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = await self._request_with_retry(
            self._client.responses.create,
            model=model,
            input=input,
            store=store,
            **kwargs,
        )
        return result.model_dump()

    async def chat_completions_create(
        self,
        model: str,
        messages: list[dict],
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = await self._request_with_retry(
            self._client.chat.completions.create,
            model=model,
            messages=messages,
            **kwargs,
        )
        return result.model_dump()

    async def _request_with_retry(
        self,
        func: Callable[..., Any],
        **kwargs: Any,
    ) -> Any:
        attempt = 0
        while True:
            await self._wait_for_throttle()
            try:
                async with self._rate_limiter:
                    return await func(**kwargs)
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                if attempt >= self.max_retries:
                    raise
                delay = self._compute_retry_delay(exc, attempt)
                await self._set_throttle(delay)
                await asyncio.sleep(delay)
                attempt += 1

    async def _wait_for_throttle(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            async with self._throttle_lock:
                delay = self._throttle_until - loop.time()

            if delay <= 0:
                return

            await asyncio.sleep(delay)

    async def _set_throttle(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        next_allowed = loop.time() + max(0.0, seconds)
        async with self._throttle_lock:
            if next_allowed > self._throttle_until:
                self._throttle_until = next_allowed

    def _compute_retry_delay(self, exc: Exception, attempt: int) -> float:
        retry_after = self._extract_retry_after_seconds(exc)
        if retry_after is not None:
            return min(self.max_backoff, max(0.0, retry_after))

        backoff = self.initial_backoff * (2**attempt)
        return min(self.max_backoff, backoff)

    @staticmethod
    def _extract_retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if not headers:
            return None

        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is None:
            return None

        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
