"""
Custom exceptions for the agetty LLM client.
These exceptions hide the underlying HTTP library from the rest of the system.
"""

class OllamaError(Exception):
    """Base exception for all Ollama-related errors."""

class OllamaConnectionError(OllamaError):
    """Raised when the client cannot connect to the Ollama server."""

class ModelNotFoundError(OllamaError):
    """Raised when the requested model is not installed or available."""

class GenerationError(OllamaError):
    """Raised when the model fails to generate a response."""
