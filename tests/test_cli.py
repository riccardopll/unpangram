import json

import httpx
import pytest

from unpangram.cli import main

from .conftest import pangram_response, words

HUMAN = words(40, "plain")
AI = words(40, "robot")
DOC = f"{HUMAN}\n\n{AI}"


def fake_apis(request):
    """Pangram flags paragraphs of 'robot'; OpenRouter rewrites them to 'person'."""
    path = request.url.path
    if request.url.host == "text.external-api.pangram.com":
        if request.method == "POST":
            fake_apis.text = json.loads(request.content)["text"]
            return httpx.Response(200, json={"task_id": "t1"})
        text = fake_apis.text
        flagged = [p for p in text.split("\n\n") if p.startswith("robot")]
        return httpx.Response(200, json=pangram_response(text, flagged=flagged))
    assert request.url.host == "openrouter.ai"
    if path.endswith("/embeddings"):
        data = [{"index": 0, "embedding": [1.0, 0.0]}, {"index": 1, "embedding": [1.0, 0.1]}]
        return httpx.Response(200, json={"data": data, "usage": {"cost": 0.0001}})
    body = json.loads(request.content)
    content = DOC if body["model"] == "anthropic/claude-sonnet-4.6" else words(40, "person")
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"cost": 0.001},
        },
    )


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PANGRAM_API_KEY", "pk")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ok")
    (tmp_path / "in.txt").write_text(DOC)
    return tmp_path


def run(*argv):
    return main(list(argv), transport=httpx.MockTransport(fake_apis))


def test_score_prints_result_and_marks_flagged_windows(env, capsys):
    assert run("score", "in.txt") == 0
    out = capsys.readouterr().out
    assert "prediction:  AI" in out
    assert "flagged:     1 of 1 windows" in out
    assert f"{HUMAN}\n\n[[AI-Generated: {AI}]]" in out


def test_rewrite_writes_outputs(env, capsys):
    assert run("rewrite", "in.txt", "--output-dir", "out") == 0
    final = (env / "out" / "final.txt").read_text()
    assert final == f"{HUMAN}\n\n{words(40, 'person')}"
    before = json.loads((env / "out" / "before.json").read_text())
    after = json.loads((env / "out" / "after.json").read_text())
    assert before["prediction_short"] == "AI"
    assert (after["prediction_short"], after["text"]) == ("Human", final)
    assert not (env / "out" / "generated.txt").exists()
    assert "fully Human" in capsys.readouterr().out


def test_rewrite_generate(env):
    assert run("rewrite", "--generate", "write about robots") == 0
    out = env / "unpangram-out"
    assert (out / "generated.txt").read_text() == DOC + "\n"
    assert (out / "final.txt").read_text() == f"{HUMAN}\n\n{words(40, 'person')}"


def test_rewrite_needs_exactly_one_input(env, capsys):
    assert run("rewrite") == 1
    assert run("rewrite", "in.txt", "--generate", "x") == 1
    assert "exactly one input" in capsys.readouterr().err


def test_missing_api_key_is_a_clean_error(env, monkeypatch, capsys):
    monkeypatch.delenv("PANGRAM_API_KEY")
    assert run("score", "in.txt") == 1
    assert "PANGRAM_API_KEY" in capsys.readouterr().err


def test_cost_cap_flag(env, capsys):
    assert run("rewrite", "in.txt", "--max-usd", "0.01") == 1
    assert "cap" in capsys.readouterr().err


def test_score_highlights_the_text_pangram_returned(env, capsys):
    # Offsets index into Pangram's normalized text, not into the file on disk.
    (env / "in.txt").write_text(f"  {HUMAN}\n\n\n\n{AI}\n")

    def normalizing(request):
        if request.method == "POST":
            return httpx.Response(200, json={"task_id": "t1"})
        return httpx.Response(200, json=pangram_response(DOC, flagged=[AI]))

    assert main(["score", "in.txt"], transport=httpx.MockTransport(normalizing)) == 0
    assert f"{HUMAN}\n\n[[AI-Generated: {AI}]]" in capsys.readouterr().out


def test_rewrite_failure_still_writes_outputs_and_exits_nonzero(env, capsys):
    def failing_model(request):
        if request.url.host == "openrouter.ai":
            return httpx.Response(404, text="no such model")
        return fake_apis(request)

    code = main(
        ["rewrite", "in.txt", "--output-dir", "out"], transport=httpx.MockTransport(failing_model)
    )
    assert code == 1
    assert (env / "out" / "final.txt").read_text() == DOC
    before = json.loads((env / "out" / "before.json").read_text())
    assert before == json.loads((env / "out" / "after.json").read_text())
    assert "round 1 (moonshotai/kimi-k2.5) failed" in capsys.readouterr().err


def test_bad_config_and_unreadable_input_are_clean_errors(env, capsys):
    (env / "unpangram.yaml").write_text("max_rounds: many\n")
    assert run("score", "in.txt") == 1
    assert "max_rounds must be a whole number" in capsys.readouterr().err
    (env / "unpangram.yaml").unlink()
    (env / "in.txt").write_bytes(b"\xff\xfe" * 40)
    assert run("score", "in.txt") == 1
    assert run("score", "missing.txt") == 1
