import json

import httpx
import pytest

from unpangram.pangram import PangramClient, PangramError, pangram_cost

from .conftest import pangram_response, words


def make_client(handler, sleeps=None, **kwargs):
    sleeps = sleeps if sleeps is not None else []
    return PangramClient(
        "test-key", transport=httpx.MockTransport(handler), sleep=sleeps.append, **kwargs
    )


def test_score_posts_task_polls_and_parses():
    text = words(60)
    seen = []
    polls = iter([{"task_id": "t1", "stage": "STAGE_PREPROCESSING"}, None])

    def handler(request):
        seen.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "t1"})
        body = next(polls) or pangram_response(text, flagged=[text])
        return httpx.Response(200, json=body)

    sleeps = []
    result = make_client(handler, sleeps, poll_interval=2.0).score(text)

    post = seen[0]
    assert post.url == "https://text.external-api.pangram.com/task"
    assert post.headers["x-api-key"] == "test-key"
    assert json.loads(post.content) == {"text": text, "model": "pangram-4"}
    assert [r.url.path for r in seen[1:]] == ["/task/t1", "/task/t1"]
    assert sleeps == [2.0]

    assert result.prediction_short == "AI"
    assert (result.fraction_ai, result.fraction_ai_assisted, result.fraction_human) == (1, 0, 0)
    assert result.version == "4.1"
    w = result.windows[0]
    assert (w.text, w.label, w.confidence) == (text, "AI-Generated", "High")
    assert (w.start_index, w.end_index) == (0, len(text))
    assert (w.ai_assistance_score, w.is_humanized, w.humanizer_score) == (0.9, False, 0.1)
    assert result.flagged_windows == [w]
    assert not result.fully_human


def test_fully_human():
    text = words(60)

    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "t1"})
        return httpx.Response(200, json=pangram_response(text))

    assert make_client(handler).score(text).fully_human


def test_retries_429_using_retry_after():
    text = words(60)
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "7"}),
            httpx.Response(200, json={"task_id": "t1"}),
            httpx.Response(429),
            httpx.Response(200, json=pangram_response(text)),
        ]
    )
    sleeps = []
    result = make_client(lambda request: next(responses), sleeps).score(text)
    assert result.prediction_short == "Human"
    assert sleeps == [7.0, 5.0]


def test_gives_up_after_repeated_429():
    client = make_client(lambda request: httpx.Response(429, headers={"Retry-After": "0"}))
    with pytest.raises(PangramError, match="429"):
        client.score(words(60))


def test_failed_stage_raises_with_reason():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "t1"})
        return httpx.Response(200, json={"stage": "STAGE_FAILED", "headline": "no valid text"})

    with pytest.raises(PangramError, match="no valid text"):
        make_client(handler).score(words(60))


def test_poll_timeout():
    def handler(request):
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "t1"})
        return httpx.Response(200, json={"stage": "STAGE_PREPROCESSING"})

    client = make_client(handler, poll_interval=1.0, poll_timeout=3.0)
    with pytest.raises(PangramError, match="did not finish"):
        client.score(words(60))


def test_refuses_short_text_without_any_request():
    def handler(request):
        raise AssertionError("no request expected")

    with pytest.raises(PangramError, match="at least 50"):
        make_client(handler).score(words(49))


def test_http_error_raises():
    client = make_client(lambda request: httpx.Response(403, text="forbidden"))
    with pytest.raises(PangramError, match="403"):
        client.score(words(60))


def test_cost_is_per_started_100_words():
    assert pangram_cost(words(100)) == pytest.approx(0.05)
    assert pangram_cost(words(101)) == pytest.approx(0.10)
    assert pangram_cost(words(250)) == pytest.approx(0.15)
