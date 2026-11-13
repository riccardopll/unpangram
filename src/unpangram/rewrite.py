"""The capped rewrite loop: rewrite flagged windows, gate on similarity, re-score."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .config import Config
from .openrouter import OpenRouterClient
from .pangram import USD_PER_100_WORDS, PangramClient, PangramResult, pangram_cost

SYSTEM_PROMPT = """You rewrite one passage of a longer document so it reads as natural, \
plainly written human prose.

Rules:
- Preserve every fact, name, number, date, quotation and the full meaning. Add nothing, drop \
nothing.
- Keep roughly the same length, the same language, and the same paragraph breaks.
- Use ordinary punctuation and plain characters only. No typos, no misspellings, no unusual \
Unicode, no invisible characters, no garbled or broken sentences.
- The passage must still fit between the surrounding context, which is shown for reference \
only and must not be rewritten or repeated.
- Reply with the rewritten passage only: no preamble, no quotes, no commentary."""


class BudgetExceeded(Exception):
    pass


class Budget:
    """Hard USD cap. Callers check `can_afford` with an estimate before every paid call."""

    def __init__(self, cap: float) -> None:
        self.cap = cap
        self.spent = 0.0

    def can_afford(self, estimate: float) -> bool:
        return self.spent + estimate <= self.cap + 1e-9

    def require(self, estimate: float, what: str) -> None:
        if not self.can_afford(estimate):
            raise BudgetExceeded(
                f"{what} is estimated at ${estimate:.4f}, but only "
                f"${max(self.cap - self.spent, 0):.4f} of the ${self.cap:.2f} cap is left"
            )

    def charge(self, amount: float) -> None:
        self.spent += amount


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def token_cost(cfg: Config, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * cfg.input_price_per_mtok + output_tokens * cfg.output_price_per_mtok
    ) / 1_000_000


def usage_cost(cfg: Config, usage: dict[str, Any]) -> float:
    """Actual cost of a call: OpenRouter's reported cost, else token counts at config prices."""
    if usage.get("cost") is not None:
        return float(usage["cost"])
    return token_cost(
        cfg, int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
    )


def cosine(a: list[float], b: list[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    if norm == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / norm


def build_messages(passage: str, before: str, after: str) -> list[dict[str, str]]:
    user = (
        f"<context_before>\n{before}\n</context_before>\n\n"
        f"<passage>\n{passage}\n</passage>\n\n"
        f"<context_after>\n{after}\n</context_after>\n\n"
        "Rewrite the passage."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def locate_spans(doc: str, result: PangramResult) -> list[tuple[int, int]]:
    """Character spans of the flagged windows in `doc`, in order and non-overlapping."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for w in sorted(result.flagged_windows, key=lambda w: w.start_index):
        if not w.text.strip():
            continue
        start = w.start_index if doc[w.start_index : w.end_index] == w.text else -1
        if start < cursor:
            start = doc.find(w.text, cursor)
        if start < 0:
            continue
        spans.append((start, start + len(w.text)))
        cursor = start + len(w.text)
    return spans


@dataclass
class RoundLog:
    round: int
    model: str
    flagged: int
    accepted: int
    rejected: int
    result: PangramResult | None  # None when nothing changed, so nothing was re-scored


@dataclass
class RewriteOutcome:
    before: PangramResult
    after: PangramResult
    final_text: str
    rounds: list[RoundLog]
    stop_reason: str


def run_rewrite(
    text: str,
    cfg: Config,
    pangram: PangramClient,
    llm: OpenRouterClient,
    budget: Budget,
    log: Callable[[str], None] = lambda _msg: None,
) -> RewriteOutcome:
    budget.require(pangram_cost(text), "the initial Pangram score")
    budget.charge(pangram_cost(text))
    before = pangram.score(text)
    log(f"before: {describe(before)}")

    doc, current = text, before
    rounds: list[RoundLog] = []
    stop_reason = f"reached the round limit ({cfg.max_rounds})"

    for number in range(1, cfg.max_rounds + 1):
        if current.fully_human:
            stop_reason = "document is fully Human with no flagged windows"
            break
        model = cfg.rotation[(number - 1) % len(cfg.rotation)]
        spans = locate_spans(doc, current)
        new_doc, accepted, rejected, out_of_budget = _rewrite_spans(
            doc, spans, model, cfg, llm, budget, log
        )
        entry = RoundLog(number, model, len(spans), accepted, rejected, None)
        rounds.append(entry)

        if accepted and budget.can_afford(pangram_cost(new_doc)):
            budget.charge(pangram_cost(new_doc))
            current = pangram.score(new_doc)
            doc = new_doc
            entry.result = current
            log(f"round {number} ({model}): {accepted}/{len(spans)} rewritten, {describe(current)}")
        elif accepted:
            # Cannot pay to re-score, so drop this round: the final text must match `after`.
            out_of_budget = True
            entry.accepted = 0
        else:
            log(f"round {number} ({model}): no rewrite accepted, document unchanged")

        if out_of_budget:
            stop_reason = f"hit the ${cfg.max_usd:.2f} cost cap"
            break
    else:
        if current.fully_human:
            stop_reason = "document is fully Human with no flagged windows"

    return RewriteOutcome(before, current, doc, rounds, stop_reason)


def _rewrite_spans(
    doc: str,
    spans: list[tuple[int, int]],
    model: str,
    cfg: Config,
    llm: OpenRouterClient,
    budget: Budget,
    log: Callable[[str], None],
) -> tuple[str, int, int, bool]:
    """Rewrite each span. Returns (new_doc, accepted, rejected, out_of_budget)."""
    # Keep enough back to re-score the rewritten document (one extra 100-word block of slack).
    reserve = pangram_cost(doc) + USD_PER_100_WORDS
    replacements: list[tuple[int, int, str]] = []
    rejected = 0
    out_of_budget = False

    for start, end in spans:
        passage = doc[start:end]
        messages = build_messages(
            passage,
            doc[max(0, start - cfg.context_chars) : start],
            doc[end : end + cfg.context_chars],
        )
        max_tokens = max(1024, 2 * estimate_tokens(passage))
        prompt_tokens = sum(estimate_tokens(m["content"]) for m in messages)
        # Chat prompt, plus the two passages sent for embedding, plus the worst-case output.
        estimate = token_cost(cfg, prompt_tokens + 2 * estimate_tokens(passage), max_tokens)
        if not budget.can_afford(estimate + reserve):
            out_of_budget = True
            break

        chat = llm.chat(model, messages, max_tokens=max_tokens)
        budget.charge(usage_cost(cfg, chat.usage))
        candidate = chat.text.strip()
        if not candidate or chat.finish_reason == "length":
            rejected += 1
            log(f"  window {start}-{end}: rejected (empty or truncated output)")
            continue

        vectors, usage = llm.embed(cfg.embedding_model, [passage, candidate])
        budget.charge(usage_cost(cfg, usage))
        similarity = cosine(vectors[0], vectors[1])
        if similarity < cfg.similarity_threshold:
            rejected += 1
            log(f"  window {start}-{end}: rejected (similarity {similarity:.2f})")
            continue
        log(f"  window {start}-{end}: accepted (similarity {similarity:.2f})")
        # Keep the window's own leading/trailing whitespace so the document still joins up.
        lead = passage[: len(passage) - len(passage.lstrip())]
        trail = passage[len(passage.rstrip()) :]
        replacements.append((start, end, lead + candidate + trail))

    new_doc = doc
    for start, end, replacement in reversed(replacements):
        new_doc = new_doc[:start] + replacement + new_doc[end:]
    return new_doc, len(replacements), rejected, out_of_budget


def describe(result: PangramResult) -> str:
    return (
        f"{result.prediction_short or '?'} "
        f"(ai {result.fraction_ai:.2f}, assisted {result.fraction_ai_assisted:.2f}, "
        f"human {result.fraction_human:.2f}, {len(result.flagged_windows)} flagged windows)"
    )
