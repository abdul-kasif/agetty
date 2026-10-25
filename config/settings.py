"""
Central configuration for the agetty Agentic OS.

Every module in this project imports `settings` from here instead of
hardcoding paths, URLs, or model names. Values can be overridden at
runtime via environment variables prefixed with AGETTY_.
"""

from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings

# Absolute path to the project root (the directory containing this package)
_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Global, immutable-ish application settings."""

    # ---------- Identity ----------
    app_name: str = "agetty"
    version: str = "0.1.0"

    # ---------- Filesystem layout ----------
    project_root: Path = _ROOT
    config_dir: Path = _ROOT / "config"
    data_dir: Path = _ROOT / "data"
    workspace_root: Path = _ROOT / "workspaces"
    sandbox_image: str = "agentic-sandbox"

    # ---------- Database ----------
    db_path: Path = _ROOT / "data" / "agetty.db"

    # ---------- Ollama / LLM backend ----------
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 300

    # ---------- Model tiers ----------
    router_model: str = "qwen2.5-coder:3b"
    coder_model: str = "qwen2.5-coder:7b"
    thinker_model: str = "qwen2.5-coder:14b"
    embedding_model: str = "nomic-embed-text"

    # ---------- Resource budgets ----------
    max_context_tokens: int = 4096
    max_agent_iterations: int = 10
    sandbox_memory_limit: str = "512m"
    sandbox_cpu_limit: float = 1.0
    sandbox_timeout_seconds: int = 30

    # ---------- API server ----------
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    model_config = {
        "env_prefix": "AGETTY_",
        "env_file": ".env",
        "extra": "ignore",
    }

    def model_post_init(self, /, __context: Any) -> None:
        """Ensure required runtime directories exist on startup."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)


# Singleton used across the whole application
settings = Settings()
