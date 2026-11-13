# unpangram

A small research CLI that measures whether a multi-model rewrite loop moves
[Pangram](https://www.pangram.com) AI-detection scores.

It scores a document with Pangram, rewrites only the windows Pangram flagged using a rotation of
models from different families (via [OpenRouter](https://openrouter.ai)), rejects rewrites that
drift in meaning, and re-scores. It runs a fixed, small number of rounds under a hard cost cap. It
is not an "until 100% human" loop.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
export PANGRAM_API_KEY=...      # needed by score and rewrite
export OPENROUTER_API_KEY=...   # needed by rewrite
```

## Usage

```sh
uv run unpangram score essay.txt
uv run unpangram rewrite essay.txt
uv run unpangram rewrite --generate "Write 300 words on tidal power."
```

`score` prints the document-level result (prediction, AI / AI-assisted / human fractions, model
version), lists each flagged window, and prints the text with flagged windows highlighted (colour
in a terminal, `[[label: ...]]` markers when piped). The text shown is the copy Pangram returns,
which it may have normalized, because window offsets refer to that copy. Pangram refuses text
under 50 words, and so does this tool.

`rewrite` does the following:

1. Scores the input (with `--generate`, it first writes the text with `generate_model`).
2. For each round, sends every flagged window, with surrounding context, to the next model in the
   rotation. The prompt requires facts, names, numbers and meaning to be preserved, with no typos,
   odd Unicode or garbling.
3. Keeps a rewrite only if its embedding cosine similarity to the original window is at or above
   the threshold. Otherwise the original window stays.
4. Re-scores the full document, and stops early if it is fully Human with no flagged windows.
5. Writes `before.json` and `after.json` (raw Pangram responses) and `final.txt` to the output
   directory, plus `generated.txt` when `--generate` was used.

If a model or Pangram call fails part-way through, the run stops, writes the results of the rounds
that completed, and exits with code 1. A flagged window that cannot be found in the document is
reported and left alone.

Flags for `rewrite`: `--rounds`, `--models a,b,c`, `--threshold`, `--max-usd`, `--output-dir`,
`--generate-model`. Flags override the config file.

## Configuration

Settings come from a YAML file: `--config PATH` (before the subcommand), else `$UNPANGRAM_CONFIG`,
else `./unpangram.yaml` if it exists. API keys come only from the environment. Every key is
optional; these are the defaults:

```yaml
pangram_model: pangram-4
rotation:
  - moonshotai/kimi-k2.5
  - deepseek/deepseek-v4.1-flash
  - qwen/qwen3.8-flash
generate_model: anthropic/claude-sonnet-4.6
embedding_model: openai/text-embedding-3-small
max_rounds: 2
similarity_threshold: 0.8
max_usd: 1.0
context_chars: 600          # context sent on each side of a window
output_dir: unpangram-out
input_price_per_mtok: 1.0   # USD per million tokens, used for cost estimates
output_price_per_mtok: 5.0
poll_interval: 1.0          # seconds between Pangram task polls
poll_timeout: 120.0
```

`pangram_base_url` and `openrouter_base_url` can also be set.

## Cost

Pangram charges $0.05 per started 100 words for every scoring call, so a 450-word document costs
$0.25 per score, and a default two-round `rewrite` scores it up to three times ($0.75). Model and
embedding calls are usually a small fraction of that.

`rewrite` enforces `max_usd` (default $1.00) as a hard cap on its own estimate of spend. Before
each call it checks the estimated cost against what is left, keeping enough back to re-score, and
stops when the next step does not fit. The estimate uses Pangram's list price, the cost OpenRouter
reports for each call, and `input_price_per_mtok` / `output_price_per_mtok` when no cost is
reported. It is an estimate, not a bill: set those two prices at or above your most expensive
model's rates. `score` is a single call and is not capped.

## Caveat

This is a research tool for measuring detector behaviour, not a way to guarantee any outcome.
Detectors change: Pangram retrains and ships new model versions, so a result measured today may
not hold tomorrow, and a "Human" score from one run says nothing about other detectors or later
versions. Treat the numbers as measurements taken at one point in time, record the Pangram
`version` alongside them, and do not use this tool to misrepresent who or what wrote a text.

## Development

```sh
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

Tests mock all HTTP and never call Pangram or OpenRouter.
