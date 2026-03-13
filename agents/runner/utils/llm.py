"""LLM utilities for agents using LiteLLM."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import acompletion, aresponses, token_counter
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
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

from runner.agents.models import LitellmAnyMessage
from runner.utils.decorators import with_retry
from runner.utils.settings import get_settings

settings = get_settings()

# Configure LiteLLM proxy routing if configured
if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
    litellm.use_litellm_proxy = True

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

            # Reservation might already be evicted if reconciliation happens >60s later.
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


def _estimate_prompt_tokens(model: str, serialized_messages: list[Any]) -> int:
    try:
        estimated = token_counter(model=model, messages=serialized_messages)
        if estimated > 0:
            return estimated
    except Exception as e:
        logger.debug(f"Failed to estimate prompt tokens for model {model}: {e}")

    rough_chars = len(str(serialized_messages))
    return max(1, rough_chars // TOKEN_ESTIMATE_FALLBACK_CHARS_PER_TOKEN)


def _estimate_request_tokens(
    model: str,
    serialized_messages: list[Any],
    extra_args: dict[str, Any],
) -> int:
    prompt_tokens = _estimate_prompt_tokens(model, serialized_messages)
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
    serialized_messages: list[Any],
    extra_args: dict[str, Any],
    api_name: str,
) -> tuple[_TokenPerMinuteLimiter, _TokenReservation | None] | None:
    tokens_per_minute = _coerce_positive_int(settings.LLM_MAX_TOKENS_PER_MINUTE)
    if tokens_per_minute is None:
        return None

    estimated_tokens = _estimate_request_tokens(model, serialized_messages, extra_args)
    logger.debug(
        f"[TPM_LIMIT] Reserving {estimated_tokens} tokens "
        f"for {api_name} call (model={model}, limit={tokens_per_minute}/min)"
    )
    limiter = _get_loop_limiter(tokens_per_minute)
    reservation = await limiter.acquire(estimated_tokens, model)
    return limiter, reservation


async def _reconcile_tpm_budget(
    reservation_handle: tuple[_TokenPerMinuteLimiter, _TokenReservation | None] | None,
    response: Any,
    model: str,
    api_name: str,
) -> None:
    if reservation_handle is None:
        return

    actual_tokens = _extract_usage_tokens_from_response(response)
    if actual_tokens is None:
        logger.debug(f"[TPM_LIMIT] No usage in {api_name} response (model={model})")
        return

    limiter, reservation = reservation_handle
    await limiter.reconcile(reservation, actual_tokens, model)


def _safe_model_dump(value: Any) -> Any:
    """Serialize pydantic-style objects without emitting noisy serializer warnings."""
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return str(value)

    # pydantic v2 supports warnings=False; older implementations may not.
    try:
        return model_dump(mode="json", warnings=False)
    except TypeError:
        pass
    except Exception:
        return str(value)

    try:
        return model_dump(mode="json")
    except TypeError:
        pass
    except Exception:
        return str(value)

    try:
        return model_dump(warnings=False)
    except TypeError:
        pass
    except Exception:
        return str(value)

    try:
        return model_dump()
    except Exception:
        return str(value)


def _serialize_any_message(msg: LitellmAnyMessage) -> Any:
    """Convert a LiteLLM message object to JSON-friendly data for logging."""
    if isinstance(msg, dict):
        return msg

    return _safe_model_dump(msg)


def _serialize_model_response(response: ModelResponse) -> Any:
    """Convert ModelResponse to JSON-friendly data for logging."""
    return _safe_model_dump(response)


def _is_context_window_error(e: Exception) -> bool:
    """
    Detect context window exceeded errors that LiteLLM doesn't properly classify.

    Some providers (notably Gemini) return context window errors as BadRequestError
    instead of ContextWindowExceededError. This predicate catches those cases
    by checking the error message content.

    Known error patterns:
    - Gemini: "input token count exceeds the maximum number of tokens allowed"
    - OpenAI: "context_length_exceeded" (usually caught as ContextWindowExceededError)
    - Anthropic: "prompt is too long" (usually caught as ContextWindowExceededError)
    """
    error_str = str(e).lower()

    # Common patterns indicating context/token limit exceeded
    context_patterns = [
        "token count exceeds",
        "context_length_exceeded",
        "context length exceeded",
        "maximum context length",
        "maximum number of tokens",
        "prompt is too long",
        "input too long",
        "exceeds the model's maximum context",
    ]

    return any(pattern in error_str for pattern in context_patterns)


def _is_non_retriable_bad_request(e: Exception) -> bool:
    """
    Detect BadRequestErrors that are deterministic and should NOT be retried.

    These are configuration/validation errors that will always fail regardless
    of retry attempts. Retrying wastes time and resources.

    Note: Patterns must be specific enough to avoid matching transient errors
    like rate limits (e.g., "maximum of 100 requests" should NOT match).
    """
    error_str = str(e).lower()

    non_retriable_patterns = [
        # Tool count errors - be specific to avoid matching rate limits
        "tools are supported",  # "Maximum of 128 tools are supported"
        "too many tools",
        # Model/auth errors
        "model not found",
        "does not exist",
        "invalid api key",
        "authentication failed",
        "unauthorized",
        "unsupported parameter",
        "unsupported value",
    ]

    return any(pattern in error_str for pattern in non_retriable_patterns)


def _should_skip_retry(e: Exception) -> bool:
    """Combined check for all non-retriable errors."""
    return _is_context_window_error(e) or _is_non_retriable_bad_request(e)


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
    skip_if=_should_skip_retry,
)
async def generate_response(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[ChatCompletionToolParam],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> ModelResponse:
    """
    Generate a response from the LLM with retry logic.

    Args:
        model: The model identifier to use
        messages: The conversation messages (input AllMessageValues or output Message)
        tools: Available tools for the model to call
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments to pass to the completion call
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The model response
    """
    serialized_messages = [_serialize_any_message(m) for m in messages]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "timeout": llm_response_timeout,
        **extra_args,
    }

    logger.bind(
        message_type="llm_request",
        model=model,
        payload={
            "messages": serialized_messages,
            "tools": tools,
            "stream": stream,
            "timeout": llm_response_timeout,
            "extra_args": extra_args,
            "trajectory_id": trajectory_id,
        },
    ).info("LLM request")

    # If LiteLLM proxy is configured, add tracking tags
    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        tags = ["service:trajectory"]
        if trajectory_id:
            tags.append(f"trajectory_id:{trajectory_id}")
        kwargs["extra_body"] = {"tags": tags}

    reservation_handle = await _reserve_tpm_budget(
        model=model,
        serialized_messages=serialized_messages,
        extra_args=extra_args,
        api_name="chat_completions",
    )

    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stream_iter: Any = await acompletion(**kwargs)
        chunks: list[ModelResponse] = []
        async for chunk in stream_iter:
            chunks.append(chunk)
        rebuilt = litellm.stream_chunk_builder(chunks, messages=messages)
        if rebuilt is None:
            raise RuntimeError("stream_chunk_builder returned None — empty stream")
        response = ModelResponse.model_validate(rebuilt)
        await _reconcile_tpm_budget(
            reservation_handle=reservation_handle,
            response=response,
            model=model,
            api_name="chat_completions",
        )
        logger.bind(
            message_type="llm_response",
            model=model,
            payload=_serialize_model_response(response),
        ).info("LLM response")
        return response

    response = await acompletion(**kwargs)
    validated = ModelResponse.model_validate(response)
    await _reconcile_tpm_budget(
        reservation_handle=reservation_handle,
        response=validated,
        model=model,
        api_name="chat_completions",
    )
    logger.bind(
        message_type="llm_response",
        model=model,
        payload=_serialize_model_response(validated),
    ).info("LLM response")
    return validated


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
    skip_if=_should_skip_retry,
)
async def call_responses_api(
    model: str,
    messages: list[LitellmAnyMessage],
    tools: list[dict[str, Any]],
    llm_response_timeout: int,
    extra_args: dict[str, Any],
    trajectory_id: str | None = None,
    stream: bool = False,
) -> Any:
    """
    Generate a response using a provider's Responses API (e.g., web search) with retry logic.

    Uses litellm.aresponses() which is the native async version.

    Args:
        model: The model identifier to use (e.g., 'openai/gpt-4o')
        messages: The conversation messages
        tools: Tools for web search (e.g., [{"type": "web_search"}])
        llm_response_timeout: Timeout in seconds for the LLM response
        extra_args: Additional arguments (reasoning, etc.)
        trajectory_id: Optional trajectory ID for tracking/tagging

    Returns:
        The OpenAI responses API response object
    """
    serialized_messages = [_serialize_any_message(m) for m in messages]
    kwargs: dict[str, Any] = {
        "model": model,
        "input": messages,
        "tools": tools,
        "timeout": llm_response_timeout,
        **extra_args,
    }

    logger.bind(
        message_type="llm_request",
        model=model,
        payload={
            "messages": serialized_messages,
            "tools": tools,
            "stream": stream,
            "timeout": llm_response_timeout,
            "extra_args": extra_args,
            "trajectory_id": trajectory_id,
            "api": "responses",
        },
    ).info("LLM request")

    if settings.LITELLM_PROXY_API_BASE and settings.LITELLM_PROXY_API_KEY:
        kwargs["api_base"] = settings.LITELLM_PROXY_API_BASE
        kwargs["api_key"] = settings.LITELLM_PROXY_API_KEY
        tags = ["service:trajectory"]
        if trajectory_id:
            tags.append(f"trajectory_id:{trajectory_id}")
        kwargs["extra_body"] = {"tags": tags}

    reservation_handle = await _reserve_tpm_budget(
        model=model,
        serialized_messages=serialized_messages,
        extra_args=extra_args,
        api_name="responses",
    )

    if stream:
        kwargs["stream"] = True
        stream_iter: Any = await aresponses(**kwargs)
        completed_response = None
        async for event in stream_iter:
            if getattr(event, "type", None) == "response.completed":
                completed_response = getattr(event, "response", None)
        if completed_response is None:
            raise RuntimeError(
                "No response.completed event received from Responses API stream"
            )
        await _reconcile_tpm_budget(
            reservation_handle=reservation_handle,
            response=completed_response,
            model=model,
            api_name="responses",
        )
        logger.bind(
            message_type="llm_response",
            model=model,
            payload=_safe_model_dump(completed_response),
        ).info("LLM response")
        return completed_response

    response = await aresponses(**kwargs)
    await _reconcile_tpm_budget(
        reservation_handle=reservation_handle,
        response=response,
        model=model,
        api_name="responses",
    )
    logger.bind(
        message_type="llm_response",
        model=model,
        payload=_safe_model_dump(response),
    ).info("LLM response")
    return response
