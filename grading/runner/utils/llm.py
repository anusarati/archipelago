"""LLM utilities for grading runner."""

import asyncio
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import token_counter
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.files.main import ModelResponse
from loguru import logger
from pydantic import BaseModel

from runner.utils.decorators import with_concurrency_limit, with_retry
from runner.utils.settings import get_settings

settings = get_settings()

# Configure LiteLLM proxy routing if configured
if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
    litellm.use_litellm_proxy = True

# Default concurrency limit for LLM calls
LLM_CONCURRENCY_LIMIT = 10

# Context variable for grading run ID
grading_run_id_ctx: ContextVar[str | None] = ContextVar("grading_run_id", default=None)

TPM_WINDOW_SECONDS = 60.0
TOKEN_ESTIMATE_FALLBACK_CHARS_PER_TOKEN = 4
MIN_TPM_WAIT_SECONDS = 0.05


@dataclass
class _TokenReservation:
    timestamp: float
    tokens: int


class _TokenPerMinuteLimiter:
    """Sliding-window token limiter shared by calls in the same event loop."""

    def __init__(self, tokens_per_minute: int):
        self.tokens_per_minute = tokens_per_minute
        self._lock = asyncio.Lock()
        self._window: deque[_TokenReservation] = deque()
        self._tokens_in_window = 0

    def _evict_expired(self, now: float) -> None:
        cutoff = now - TPM_WINDOW_SECONDS
        while self._window and self._window[0].timestamp <= cutoff:
            reservation = self._window.popleft()
            self._tokens_in_window -= reservation.tokens

    async def acquire(self, requested_tokens: int, model: str) -> _TokenReservation | None:
        if requested_tokens <= 0:
            return None

        reserve_tokens = requested_tokens
        if reserve_tokens > self.tokens_per_minute:
            logger.warning(
                f"Estimated request tokens ({reserve_tokens}) exceed "
                f"LLM_MAX_TOKENS_PER_MINUTE ({self.tokens_per_minute}); "
                "capping limiter reservation to avoid deadlock."
            )
            reserve_tokens = self.tokens_per_minute

        while True:
            async with self._lock:
                now = time.monotonic()
                self._evict_expired(now)

                if self._tokens_in_window + reserve_tokens <= self.tokens_per_minute:
                    reservation = _TokenReservation(
                        timestamp=now,
                        tokens=reserve_tokens,
                    )
                    self._window.append(reservation)
                    self._tokens_in_window += reserve_tokens
                    return reservation

                if not self._window:
                    wait_for = MIN_TPM_WAIT_SECONDS
                else:
                    oldest_timestamp = self._window[0].timestamp
                    wait_for = max(
                        MIN_TPM_WAIT_SECONDS,
                        TPM_WINDOW_SECONDS - (now - oldest_timestamp),
                    )

                logger.debug(
                    f"[TPM_LIMIT] Waiting {wait_for:.2f}s "
                    f"(model={model}, reserved={reserve_tokens}, "
                    f"used={self._tokens_in_window}, limit={self.tokens_per_minute})"
                )
            await asyncio.sleep(wait_for)

    async def reconcile(
        self,
        reservation: _TokenReservation | None,
        actual_tokens: int | None,
        model: str,
    ) -> None:
        if reservation is None or actual_tokens is None or actual_tokens <= 0:
            return

        async with self._lock:
            now = time.monotonic()
            self._evict_expired(now)

            if reservation not in self._window:
                return

            delta = actual_tokens - reservation.tokens
            if delta == 0:
                return

            reservation.tokens = actual_tokens
            self._tokens_in_window += delta
            if self._tokens_in_window < 0:
                self._tokens_in_window = 0

            logger.debug(
                f"[TPM_LIMIT] Reconciled reservation by {delta} tokens "
                f"(model={model}, actual={actual_tokens}, used={self._tokens_in_window})"
            )


_loop_limiters: dict[int, _TokenPerMinuteLimiter] = {}


def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        if value <= 0:
            return None
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _coerce_non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if value < 0:
            return None
        return int(value)
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def _get_completion_token_reservation(extra_args: dict[str, Any]) -> int:
    for key in ("max_completion_tokens", "max_output_tokens", "max_tokens"):
        maybe_tokens = _coerce_positive_int(extra_args.get(key))
        if maybe_tokens is not None:
            return maybe_tokens
    return max(0, settings.LLM_TPM_DEFAULT_COMPLETION_TOKENS)


def _estimate_prompt_tokens(model: str, messages: list[dict[str, Any]]) -> int:
    try:
        estimated = token_counter(model=model, messages=messages)
        if estimated > 0:
            return estimated
    except Exception as e:
        logger.debug(f"Failed to estimate prompt tokens for model {model}: {e}")

    rough_chars = len(str(messages))
    return max(1, rough_chars // TOKEN_ESTIMATE_FALLBACK_CHARS_PER_TOKEN)


def _estimate_request_tokens(
    model: str,
    messages: list[dict[str, Any]],
    extra_args: dict[str, Any],
) -> int:
    prompt_tokens = _estimate_prompt_tokens(model, messages)
    completion_tokens = _get_completion_token_reservation(extra_args)
    estimated_total = prompt_tokens + completion_tokens
    multiplier = (
        settings.LLM_TPM_ESTIMATE_MULTIPLIER
        if settings.LLM_TPM_ESTIMATE_MULTIPLIER > 0
        else 1.0
    )
    return max(1, int(estimated_total * multiplier))


def _get_loop_limiter(tokens_per_minute: int) -> _TokenPerMinuteLimiter:
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    limiter = _loop_limiters.get(loop_id)
    if limiter is None or limiter.tokens_per_minute != tokens_per_minute:
        limiter = _TokenPerMinuteLimiter(tokens_per_minute)
        _loop_limiters[loop_id] = limiter
    return limiter


def _get_usage_field(usage: Any, field_name: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        return _coerce_non_negative_int(usage.get(field_name))
    return _coerce_non_negative_int(getattr(usage, field_name, None))


def _extract_usage_tokens_from_response(response: Any) -> int | None:
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None

    total_tokens = _get_usage_field(usage, "total_tokens")
    if total_tokens is not None and total_tokens > 0:
        return total_tokens

    prompt_tokens = _get_usage_field(usage, "prompt_tokens")
    if prompt_tokens is None:
        prompt_tokens = _get_usage_field(usage, "input_tokens")

    completion_tokens = _get_usage_field(usage, "completion_tokens")
    if completion_tokens is None:
        completion_tokens = _get_usage_field(usage, "output_tokens")

    if prompt_tokens is None and completion_tokens is None:
        return None

    return (prompt_tokens or 0) + (completion_tokens or 0)


async def _reserve_tpm_budget(
    model: str,
    messages: list[dict[str, Any]],
    extra_args: dict[str, Any],
) -> tuple[_TokenPerMinuteLimiter, _TokenReservation | None] | None:
    tokens_per_minute = _coerce_positive_int(settings.LLM_MAX_TOKENS_PER_MINUTE)
    if tokens_per_minute is None:
        return None

    estimated_tokens = _estimate_request_tokens(model, messages, extra_args)
    logger.debug(
        f"[TPM_LIMIT] Reserving {estimated_tokens} tokens "
        f"for chat_completions call (model={model}, limit={tokens_per_minute}/min)"
    )
    limiter = _get_loop_limiter(tokens_per_minute)
    reservation = await limiter.acquire(estimated_tokens, model)
    return limiter, reservation


async def _reconcile_tpm_budget(
    reservation_handle: tuple[_TokenPerMinuteLimiter, _TokenReservation | None] | None,
    response: Any,
    model: str,
) -> None:
    if reservation_handle is None:
        return

    actual_tokens = _extract_usage_tokens_from_response(response)
    if actual_tokens is None:
        logger.debug(f"[TPM_LIMIT] No usage in chat_completions response (model={model})")
        return

    limiter, reservation = reservation_handle
    await limiter.reconcile(reservation, actual_tokens, model)


def _is_non_retriable_error(e: Exception) -> bool:
    """
    Detect errors that are deterministic and should NOT be retried.

    These include:
    - Context window exceeded (content-based detection for providers that don't classify properly)
    - Configuration/validation errors that will always fail

    Note: Patterns must be specific enough to avoid matching transient errors
    like rate limits (e.g., "maximum of 100 requests" should NOT match).
    """
    error_str = str(e).lower()

    non_retriable_patterns = [
        # Context window patterns
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
        # Tool count errors - be specific to avoid matching rate limits
        "tools are supported",  # "Maximum of 128 tools are supported"
        "too many tools",
        # Model/auth errors
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "invalid base64",
    ]

    return any(pattern in error_str for pattern in non_retriable_patterns)


@contextmanager
def grading_context(grading_run_id: str) -> Generator[None]:
    """
    Context manager for setting grading_run_id, similar to logger.contextualize().

    Usage:
        with grading_context(grading_run_id):
            # All LLM calls in here automatically get the grading_run_id in metadata
            ...
    """
    token = grading_run_id_ctx.set(grading_run_id)
    try:
        yield
    finally:
        grading_run_id_ctx.reset(token)


def build_messages(
    system_prompt: str,
    user_prompt: str,
    images: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build messages list for LLM call.

    Args:
        system_prompt: System prompt content
        user_prompt: User prompt content
        images: Optional list of image dicts with 'url' key for vision models

    Returns:
        List of message dicts ready for LiteLLM
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]

    if images:
        # Build multimodal user message with text + images
        # Each image is preceded by a text label with its placeholder ID
        # so the LLM can correlate images with artifact content
        user_content: list[dict[str, Any]] = [
            {"type": "text", "text": user_prompt},
        ]
        for img in images:
            if img.get("url"):
                # Add text label before image to identify it
                placeholder = img.get("placeholder", "")
                if placeholder:
                    user_content.append(
                        {"type": "text", "text": f"IMAGE: {placeholder}"}
                    )
                user_content.append(
                    {"type": "image_url", "image_url": {"url": img["url"]}}
                )
        messages.append({"role": "user", "content": user_content})
    else:
        messages.append({"role": "user", "content": user_prompt})

    return messages


@with_retry(
    max_retries=10,
    base_backoff=5,
    jitter=5,
    retry_on=(
        RateLimitError,
        Timeout,
        BadRequestError,
        ServiceUnavailableError,
        APIConnectionError,
        InternalServerError,
        BadGatewayError,
    ),
    skip_on=(ContextWindowExceededError,),
    skip_if=_is_non_retriable_error,
)
@with_concurrency_limit(max_concurrency=LLM_CONCURRENCY_LIMIT)
async def call_llm(
    model: str,
    messages: list[dict[str, Any]],
    timeout: int,
    extra_args: dict[str, Any] | None = None,
    response_format: dict[str, Any] | type[BaseModel] | None = None,
) -> ModelResponse:
    """
    Call LLM with retry logic.

    Args:
        model: Full model string (e.g., "gemini/gemini-2.0-flash")
        messages: List of message dicts (caller builds system/user/images)
        timeout: Request timeout in seconds
        extra_args: Extra LLM arguments (temperature, max_tokens, etc.)
        response_format: For structured output - {"type": "json_object"} or Pydantic class

    Returns:
        ModelResponse from LiteLLM
    """
    resolved_extra_args = extra_args or {}

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
        **resolved_extra_args,
    }

    if response_format:
        kwargs["response_format"] = response_format

    # If LiteLLM proxy is configured, add tracking tags
    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        tags = ["service:grading"]
        grading_run_id = grading_run_id_ctx.get()
        if grading_run_id:
            tags.append(f"grading_run_id:{grading_run_id}")
        kwargs["extra_body"] = {"tags": tags}

    reservation_handle = await _reserve_tpm_budget(
        model=model,
        messages=messages,
        extra_args=resolved_extra_args,
    )

    response = await litellm.acompletion(**kwargs)
    validated = ModelResponse.model_validate(response)
    await _reconcile_tpm_budget(
        reservation_handle=reservation_handle,
        response=validated,
        model=model,
    )
    return validated
