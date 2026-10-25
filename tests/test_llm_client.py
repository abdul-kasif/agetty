"""
Unit tests for the OllamaClient using pytest and unittest.mock.
We mock httpx so these tests run instantly without consuming VRAM.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from kernel.exceptions import ModelNotFoundError, OllamaConnectionError
from kernel.llm_client import OllamaClient


@pytest.fixture
def client():
    """Provides an OllamaClient with a mocked httpx.Client."""
    with patch("kernel.llm_client.httpx.Client") as mock_client_class:
        mock_client_instance = MagicMock()
        mock_client_class.return_value = mock_client_instance
        mock_client_instance.close.return_value = None
        yield OllamaClient()


def test_is_available_true(client):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    client.client.get.return_value = mock_response

    assert client.is_available() is True
    client.client.get.assert_called_once_with("/api/version")


def test_is_available_false(client):
    client.client.get.side_effect = httpx.ConnectError("Connection refused")
    assert client.is_available() is False


def test_list_models(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "models": [{"name": "qwen2.5-coder:3b"}, {"name": "qwen2.5-coder:7b"}]
    }
    mock_response.raise_for_status.return_value = None
    client.client.get.return_value = mock_response

    models = client.list_models()
    assert models == ["qwen2.5-coder:3b", "qwen2.5-coder:7b"]


def test_generate(client):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Hello world"}
    mock_response.raise_for_status.return_value = None
    client.client.post.return_value = mock_response

    result = client.generate("qwen2.5-coder:3b", "hi")
    assert result == "Hello world"

    call_args = client.client.post.call_args
    assert call_args[0][0] == "/api/generate"
    payload = call_args[1]["json"]
    assert payload["model"] == "qwen2.5-coder:3b"
    assert payload["stream"] is False


def test_unload(client):
    mock_response = MagicMock()
    client.client.post.return_value = mock_response

    client.unload("qwen2.5-coder:3b")

    call_args = client.client.post.call_args
    payload = call_args[1]["json"]
    assert payload["keep_alive"] == 0


def test_connection_error_raises_custom_exception(client):
    client.client.get.side_effect = httpx.ConnectError("Connection refused")

    with pytest.raises(OllamaConnectionError):
        client.list_models()


def test_404_raises_model_not_found(client):
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "model not found"
    mock_request = MagicMock()
    client.client.post.side_effect = httpx.HTTPStatusError(
        "Not Found", request=mock_request, response=mock_response
    )

    with pytest.raises(ModelNotFoundError):
        client.generate("non-existent-model", "hi")
