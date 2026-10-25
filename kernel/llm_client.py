"""
LLM Client for the Agentic OS.

This module provides a clean Python wrapper around the Ollama REST API.
No other module in the system should talk to Ollama directly.
"""

import json

import httpx

from config.settings import settings
from kernel.exceptions import (
    GenerationError,
    ModelNotFoundError,
    OllamaConnectionError,
    OllamaError,
)


class OllamaClient:
    """Wraps the Ollama REST API for model inference and management."""

    def __init__(self):
        self.base_url = settings.ollama_host
        self.timeout = settings.ollama_timeout_seconds
        self.client = httpx.Client(base_url=self.base_url, timeout=self.timeout)

    def _handle_request_error(self, e: httpx.HTTPError, context: str):
        """Maps httpx exceptions to our custom Agentic OS exceptions."""
        if isinstance(e, (httpx.ConnectError, httpx.RemoteProtocolError)):
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.base_url}"
            ) from e
        if isinstance(e, httpx.TimeoutException):
            raise OllamaConnectionError(
                f"Connection to Ollama timed out during {context}"
            ) from e
        if isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code == 404:
                raise ModelNotFoundError(f"Model not found during {context}") from e
            raise GenerationError(
                f"Ollama API error during {context}: {e.response.status_code} - {e.response.text}"
            ) from e
        raise OllamaError(f"Unexpected error during {context}: {e}") from e

    def is_available(self) -> bool:
        """Checks if the Ollama server is running and reachable."""
        try:
            response = self.client.get("/api/version")
            response.raise_for_status()
            return True
        except httpx.HTTPError:
            return False

    def list_models(self) -> list[str]:
        """Returns a list of available model names installed in Ollama."""
        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.HTTPError as e:
            self._handle_request_error(e, "list_models")
        return []  # Unreachable

    def generate(self, model: str, prompt: str, system: str = "") -> str:
        """Sends a prompt and returns the complete response as a string."""
        payload = {"model": model, "prompt": prompt, "system": system, "stream": False}
        try:
            response = self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except httpx.HTTPError as e:
            self._handle_request_error(e, "generate")
        return ""  # Unreachable

    def generate_stream(self, model: str, prompt: str, system: str = ""):
        """Sends a prompt and yields response chunks as they are generated."""
        payload = {"model": model, "prompt": prompt, "system": system, "stream": True}
        try:
            with self.client.stream("POST", "/api/generate", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
        except httpx.HTTPError as e:
            self._handle_request_error(e, "generate_stream")

    def unload(self, model: str) -> None:
        """Unloads a specific model from Ollama's memory by setting keep_alive to 0."""
        payload = {"model": model, "keep_alive": 0}
        try:
            self.client.post("/api/generate", json=payload)
        except httpx.HTTPError as e:
            self._handle_request_error(e, "unload")

    def get_loaded_models(self) -> list[str]:
        """Returns a list of model names currently loaded in memory (VRAM/RAM)."""
        try:
            response = self.client.get("/api/ps")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except httpx.HTTPError as e:
            self._handle_request_error(e, "get_loaded_models")
        return []  # Unreachable

    def close(self):
        """Cleanly close the HTTP client connection."""
        self.client.close()
