import json

import httpx
import pytest

from unpangram.openrouter import OpenRouterClient, OpenRouterError


def make_client(handler):
    return OpenRouterClient("or-key", transport=httpx.MockTransport(handler))


def test_chat():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "cost": 0.0002},
            },
        )

    messages = [{"role": "user", "content": "hi"}]
    result = make_client(handler).chat("some/model", messages, max_tokens=10)

    assert seen[0].url == "https://openrouter.ai/api/v1/chat/completions"
    assert seen[0].headers["Authorization"] == "Bearer or-key"
    assert json.loads(seen[0].content) == {
        "model": "some/model",
        "messages": messages,
        "max_tokens": 10,
    }
    assert (result.text, result.finish_reason) == ("hello", "stop")
    assert result.usage["cost"] == 0.0002


def test_embed_returns_vectors_in_input_order():
    def handler(request):
        assert request.url.path == "/api/v1/embeddings"
        assert json.loads(request.content) == {"model": "emb/model", "input": ["a", "b"]}
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 2},
            },
        )

    vectors, usage = make_client(handler).embed("emb/model", ["a", "b"])
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert usage == {"prompt_tokens": 2}


def test_http_error_raises():
    client = make_client(lambda request: httpx.Response(401, text="bad key"))
    with pytest.raises(OpenRouterError, match="401"):
        client.chat("m", [])


def test_error_body_raises():
    client = make_client(lambda request: httpx.Response(200, json={"error": {"message": "nope"}}))
    with pytest.raises(OpenRouterError, match="nope"):
        client.chat("m", [])
