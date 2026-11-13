"""Command line interface: `unpangram score` and `unpangram rewrite`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

from .config import (
    OPENROUTER_KEY_ENV,
    PANGRAM_KEY_ENV,
    Config,
    ConfigError,
    load_config,
    require_env,
)
from .openrouter import OpenRouterClient, OpenRouterError
from .pangram import PangramClient, PangramError, PangramResult, pangram_cost
from .rewrite import (
    Budget,
    BudgetExceeded,
    describe,
    estimate_tokens,
    run_rewrite,
    token_cost,
    usage_cost,
)

GENERATE_OUTPUT_TOKENS = 2000  # assumed output size when estimating a --generate call
COLORS = {"AI-Generated": "\033[41m", "AI-Assisted": "\033[43m"}
RESET = "\033[0m"


def highlight(text: str, result: PangramResult, color: bool) -> str:
    """The document with flagged windows marked (ANSI colour, or [[label: ...]] markers)."""
    out, cursor = [], 0
    for w in sorted(result.flagged_windows, key=lambda w: w.start_index):
        start, end = w.start_index, w.end_index
        if start < cursor or end > len(text) or start >= end:
            continue
        out.append(text[cursor:start])
        if color:
            out.append(f"{COLORS.get(w.label, COLORS['AI-Generated'])}{text[start:end]}{RESET}")
        else:
            out.append(f"[[{w.label}: {text[start:end]}]]")
        cursor = end
    out.append(text[cursor:])
    return "".join(out)


def format_result(result: PangramResult) -> str:
    lines = [
        f"prediction:  {result.prediction_short}",
        f"ai:          {result.fraction_ai:.3f}",
        f"ai-assisted: {result.fraction_ai_assisted:.3f}",
        f"human:       {result.fraction_human:.3f}",
        f"version:     {result.version}",
        f"flagged:     {len(result.flagged_windows)} of {len(result.windows)} windows",
    ]
    for w in result.flagged_windows:
        humanized = f", humanized {w.humanizer_score:.2f}" if w.is_humanized else ""
        lines.append(
            f"  [{w.start_index}:{w.end_index}] {w.label}, score {w.ai_assistance_score:.2f}, "
            f"confidence {w.confidence}{humanized}"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="unpangram", description=__doc__)
    parser.add_argument("--config", help="path to a YAML config file (default: ./unpangram.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)

    score = sub.add_parser("score", help="score a file with Pangram")
    score.add_argument("file")

    rewrite = sub.add_parser("rewrite", help="rewrite flagged windows and re-score")
    rewrite.add_argument("file", nargs="?", help="input text file (omit with --generate)")
    rewrite.add_argument("--generate", metavar="PROMPT", help="generate the input text first")
    rewrite.add_argument("--generate-model", help="model used by --generate")
    rewrite.add_argument("--rounds", type=int, help="maximum rewrite rounds")
    rewrite.add_argument("--models", help="comma-separated model rotation")
    rewrite.add_argument("--threshold", type=float, help="minimum similarity to accept a rewrite")
    rewrite.add_argument("--max-usd", type=float, help="hard cost cap in USD")
    rewrite.add_argument("--output-dir", help="where to write results")
    return parser


def apply_flags(cfg: Config, args: argparse.Namespace) -> None:
    if args.rounds is not None:
        cfg.max_rounds = args.rounds
    if args.models:
        cfg.rotation = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.threshold is not None:
        cfg.similarity_threshold = args.threshold
    if args.max_usd is not None:
        cfg.max_usd = args.max_usd
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.generate_model:
        cfg.generate_model = args.generate_model
    if not cfg.rotation:
        raise ConfigError("the model rotation is empty")


def make_pangram(cfg: Config, transport: httpx.BaseTransport | None) -> PangramClient:
    return PangramClient(
        require_env(PANGRAM_KEY_ENV),
        base_url=cfg.pangram_base_url,
        model=cfg.pangram_model,
        poll_interval=cfg.poll_interval,
        poll_timeout=cfg.poll_timeout,
        transport=transport,
    )


def cmd_score(args: argparse.Namespace, cfg: Config, transport: httpx.BaseTransport | None) -> int:
    text = Path(args.file).read_text(encoding="utf-8")
    result = make_pangram(cfg, transport).score(text)
    print(format_result(result))
    print()
    # Window offsets index into the text Pangram returns, which it may have normalized.
    print(highlight(result.text or text, result, color=sys.stdout.isatty()))
    print(f"\ncost: ${pangram_cost(text):.2f}", file=sys.stderr)
    return 0


def cmd_rewrite(
    args: argparse.Namespace, cfg: Config, transport: httpx.BaseTransport | None
) -> int:
    apply_flags(cfg, args)
    if bool(args.file) == bool(args.generate):
        raise ConfigError("give exactly one input: a file, or --generate")
    pangram = make_pangram(cfg, transport)
    llm = OpenRouterClient(
        require_env(OPENROUTER_KEY_ENV), base_url=cfg.openrouter_base_url, transport=transport
    )
    budget = Budget(cfg.max_usd)
    out_dir = Path(cfg.output_dir)

    if args.generate:
        estimate = token_cost(cfg, estimate_tokens(args.generate), GENERATE_OUTPUT_TOKENS)
        budget.require(estimate, "generation")
        chat = llm.chat(cfg.generate_model, [{"role": "user", "content": args.generate}])
        budget.charge(usage_cost(cfg, chat.usage))
        text = chat.text.strip()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "generated.txt").write_text(text + "\n", encoding="utf-8")
    else:
        text = Path(args.file).read_text(encoding="utf-8")

    outcome = run_rewrite(text, cfg, pangram, llm, budget, log=print)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, result in (("before.json", outcome.before), ("after.json", outcome.after)):
        (out_dir / name).write_text(json.dumps(result.raw, indent=2) + "\n", encoding="utf-8")
    (out_dir / "final.txt").write_text(outcome.final_text, encoding="utf-8")

    print(f"after:  {describe(outcome.after)}")
    print(f"stopped: {outcome.stop_reason}")
    print(f"estimated cost: ${budget.spent:.4f} of ${budget.cap:.2f} cap")
    print(f"wrote before.json, after.json and final.txt to {out_dir}/")
    if outcome.failed:
        print(f"unpangram: {outcome.stop_reason}", file=sys.stderr)
    return 1 if outcome.failed else 0


def main(argv: list[str] | None = None, transport: httpx.BaseTransport | None = None) -> int:
    """`transport` is for tests only: it replaces the network for both API clients."""
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
        command = cmd_score if args.command == "score" else cmd_rewrite
        return command(args, cfg, transport)
    except (
        ConfigError,
        PangramError,
        OpenRouterError,
        BudgetExceeded,
        OSError,
        UnicodeDecodeError,
    ) as exc:
        print(f"unpangram: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
