from enum import Enum
from functools import cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(Enum):
    LOCAL = "local"
    DEV = "dev"
    DEMO = "demo"
    PROD = "prod"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ENV: Environment = Environment.LOCAL

    SAVE_WEBHOOK_URL: str | None = None
    SAVE_WEBHOOK_API_KEY: str | None = None
    SCORE_WEBHOOK_URL: str | None = None

    # Datadog
    DATADOG_LOGGING: bool = False
    DATADOG_API_KEY: str | None = None
    DATADOG_APP_KEY: str | None = None

    # LiteLLM Proxy
    # If set, all LLM requests will be routed through the proxy
    LITELLM_PROXY_API_BASE: str | None = None
    LITELLM_PROXY_API_KEY: str | None = None

    # Optional token-per-minute guardrails for LLM calls
    # Set LLM_MAX_TOKENS_PER_MINUTE to enable throttling (unset/<=0 disables it)
    LLM_MAX_TOKENS_PER_MINUTE: int | None = 750_000
    # Reserved output tokens used when max_tokens is not provided in request args
    LLM_TPM_DEFAULT_COMPLETION_TOKENS: int = 2048
    # Conservative multiplier applied to prompt + completion estimate
    LLM_TPM_ESTIMATE_MULTIPLIER: float = 1.1

    # Scraping / web content (used by ACE link verification)
    ACE_FIRECRAWL_API_KEY: str | None = None

    # Data Delivery API (document parsing with caching)
    MERCOR_DELIVERY_API_KEY: str | None = None


@cache
def get_settings() -> Settings:
    return Settings()
