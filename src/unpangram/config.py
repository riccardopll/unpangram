"""Configuration: defaults, overridden by a YAML file, overridden by CLI flags.

API keys are read from the environment only (PANGRAM_API_KEY, OPENROUTER_API_KEY).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG_FILE = "unpangram.yaml"
CONFIG_ENV = "UNPANGRAM_CONFIG"
PANGRAM_KEY_ENV = "PANGRAM_API_KEY"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


class ConfigError(Exception):
    pass


@dataclass
class Config:
    pangram_base_url: str = "https://text.external-api.pangram.com"
    pangram_model: str = "pangram-4"
    poll_interval: float = 1.0
    poll_timeout: float = 120.0
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    rotation: list[str] = field(
        default_factory=lambda: [
            "moonshotai/kimi-k2.5",
            "deepseek/deepseek-v4.1-flash",
            "qwen/qwen3.8-flash",
        ]
    )
    generate_model: str = "anthropic/claude-sonnet-4.6"
    embedding_model: str = "openai/text-embedding-3-small"
    max_rounds: int = 2
    similarity_threshold: float = 0.8
    max_usd: float = 1.0
    context_chars: int = 600
    output_dir: str = "unpangram-out"
    # Used to estimate cost before a call, and after it when OpenRouter reports no cost.
    input_price_per_mtok: float = 1.0
    output_price_per_mtok: float = 5.0


def load_config(path: str | None = None) -> Config:
    """Load config from `path`, else $UNPANGRAM_CONFIG, else ./unpangram.yaml if present."""
    explicit = path or os.environ.get(CONFIG_ENV)
    file = Path(explicit or DEFAULT_CONFIG_FILE)
    if not file.exists():
        if explicit:
            raise ConfigError(f"config file not found: {file}")
        return Config()
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{file}: expected a YAML mapping")
    known = {f.name for f in fields(Config)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{file}: unknown config keys: {', '.join(unknown)}")
    cfg = Config(**data)
    if not isinstance(cfg.rotation, list) or not cfg.rotation:
        raise ConfigError(f"{file}: rotation must be a non-empty list of model ids")
    return cfg


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"environment variable {name} is not set")
    return value
