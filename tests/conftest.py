import httpx
import pytest


@pytest.fixture(autouse=True)
def no_real_http(monkeypatch):
    """Fail loudly if anything tries to reach the network."""

    def blocked(self, request):
        raise AssertionError(f"real HTTP call attempted: {request.method} {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.delenv("UNPANGRAM_CONFIG", raising=False)


def words(n: int, word: str = "word") -> str:
    return " ".join([word] * n)


def window(text, start, label="AI-Generated"):
    return {
        "text": text,
        "label": label,
        "ai_assistance_score": 0.9 if label != "Human Written" else 0.02,
        "confidence": "High",
        "start_index": start,
        "end_index": start + len(text),
        "is_humanized": False,
        "humanizer_score": 0.1,
    }


def pangram_response(text, flagged=(), prediction=None):
    """A STAGE_SUCCESS body where each string in `flagged` is an AI-Generated window."""
    windows = [window(part, text.index(part)) for part in flagged]
    if not windows:
        windows = [window(text, 0, "Human Written")]
    ai = 1.0 if flagged else 0.0
    return {
        "stage": "STAGE_SUCCESS",
        "text": text,
        "version": "4.1",
        "prediction_short": prediction or ("AI" if flagged else "Human"),
        "fraction_ai": ai,
        "fraction_ai_assisted": 0.0,
        "fraction_human": 1.0 - ai,
        "windows": windows,
    }
