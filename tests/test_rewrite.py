import pytest

from unpangram.config import Config
from unpangram.openrouter import ChatResult
from unpangram.pangram import parse_result
from unpangram.rewrite import (
    SYSTEM_PROMPT,
    Budget,
    BudgetExceeded,
    cosine,
    run_rewrite,
    usage_cost,
)

from .conftest import pangram_response, words

HUMAN = words(40, "plain")
AI = words(40, "robot")
DOC = f"{HUMAN}\n\n{AI}"
SAME, DIFFERENT = [1.0, 0.0], [0.0, 1.0]


class FakePangram:
    """Flags every run of 'robot'/'android' words; everything else is human."""

    def __init__(self, flag_words=("robot",)):
        self.flag_words = flag_words
        self.scored = []

    def score(self, text):
        self.scored.append(text)
        flagged = [p for p in text.split("\n\n") if p.split()[0] in self.flag_words]
        return parse_result(pangram_response(text, flagged=flagged))


class FakeLLM:
    def __init__(self, rewrites, vectors=(SAME, SAME), finish_reason="stop"):
        self.rewrites = rewrites  # model id -> replacement word
        self.vectors = vectors
        self.finish_reason = finish_reason
        self.chats = []

    def chat(self, model, messages, max_tokens=None):
        self.chats.append((model, messages))
        text = words(40, self.rewrites[model])
        return ChatResult(
            text=f"  {text}\n", usage={"cost": 0.001}, finish_reason=self.finish_reason
        )

    def embed(self, model, inputs):
        return [list(v) for v in self.vectors], {"cost": 0.0001}


def cfg(**overrides):
    return Config(rotation=["fam-a/one", "fam-b/two", "fam-c/three"], **overrides)


def test_stops_early_when_fully_human():
    pangram, llm, budget = FakePangram(), FakeLLM({"fam-a/one": "person"}), Budget(1.0)
    outcome = run_rewrite(DOC, cfg(), pangram, llm, budget)

    assert outcome.final_text == f"{HUMAN}\n\n{words(40, 'person')}"
    assert outcome.before.prediction_short == "AI"
    assert outcome.after.fully_human
    assert "fully Human" in outcome.stop_reason
    assert len(outcome.rounds) == 1
    assert pangram.scored == [DOC, outcome.final_text]  # re-scored the full document
    # 2 scores of 80 words ($0.05 each) + one chat + one embedding call
    assert budget.spent == pytest.approx(0.05 + 0.05 + 0.001 + 0.0001)


def test_only_flagged_window_is_sent_with_context():
    llm = FakeLLM({"fam-a/one": "person"})
    run_rewrite(DOC, cfg(), FakePangram(), llm, Budget(1.0))

    assert len(llm.chats) == 1
    system, user = llm.chats[0][1]
    assert system["content"] == SYSTEM_PROMPT
    assert f"<passage>\n{AI}\n</passage>" in user["content"]
    assert f"<context_before>\n{HUMAN}\n\n\n</context_before>" in user["content"]


def test_rotates_models_and_respects_round_limit():
    pangram = FakePangram(flag_words=("robot", "android"))
    llm = FakeLLM({"fam-a/one": "android", "fam-b/two": "robot", "fam-c/three": "android"})
    outcome = run_rewrite(DOC, cfg(max_rounds=3), pangram, llm, Budget(5.0))

    assert [model for model, _ in llm.chats] == ["fam-a/one", "fam-b/two", "fam-c/three"]
    assert [r.model for r in outcome.rounds] == ["fam-a/one", "fam-b/two", "fam-c/three"]
    assert "round limit" in outcome.stop_reason
    assert len(pangram.scored) == 4
    assert not outcome.after.fully_human


def test_default_is_two_rounds():
    pangram = FakePangram(flag_words=("robot", "android"))
    llm = FakeLLM({"fam-a/one": "android", "fam-b/two": "robot"})
    outcome = run_rewrite(DOC, cfg(), pangram, llm, Budget(5.0))
    assert len(outcome.rounds) == 2


def test_similarity_gate_rejects_and_keeps_original():
    pangram = FakePangram()
    llm = FakeLLM({"fam-a/one": "person", "fam-b/two": "person"}, vectors=(SAME, DIFFERENT))
    outcome = run_rewrite(DOC, cfg(), pangram, llm, Budget(1.0))

    assert outcome.final_text == DOC
    assert [(r.accepted, r.rejected) for r in outcome.rounds] == [(0, 1), (0, 1)]
    assert pangram.scored == [DOC]  # unchanged text is not re-scored
    assert outcome.after is outcome.before


def test_truncated_output_is_rejected():
    llm = FakeLLM({"fam-a/one": "person"}, finish_reason="length")
    outcome = run_rewrite(DOC, cfg(max_rounds=1), FakePangram(), llm, Budget(1.0))
    assert outcome.final_text == DOC


def test_cap_too_small_for_first_score():
    pangram = FakePangram()
    with pytest.raises(BudgetExceeded):
        run_rewrite(DOC, cfg(), pangram, FakeLLM({}), Budget(0.01))
    assert pangram.scored == []


def test_cap_stops_before_rewriting():
    # Enough for the first score, not enough to rewrite and re-score.
    pangram, llm, budget = FakePangram(), FakeLLM({"fam-a/one": "person"}), Budget(0.08)
    outcome = run_rewrite(DOC, cfg(), pangram, llm, budget)

    assert llm.chats == []
    assert outcome.final_text == DOC
    assert "cost cap" in outcome.stop_reason
    assert budget.spent <= budget.cap


def test_usage_cost_prefers_reported_cost_then_falls_back_to_tokens():
    config = Config(input_price_per_mtok=1.0, output_price_per_mtok=5.0)
    assert usage_cost(config, {"cost": 0.25, "prompt_tokens": 10**6}) == 0.25
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 200_000}
    assert usage_cost(config, usage) == pytest.approx(2.0)


def test_cosine():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_prompt_states_the_preservation_rules():
    lowered = SYSTEM_PROMPT.lower()
    for needle in ("fact", "name", "number", "meaning", "no typos", "unicode", "garbled"):
        assert needle in lowered
