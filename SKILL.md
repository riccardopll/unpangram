---
name: unpangram
description: Score text with the Pangram AI detector and run a short, cost-capped multi-model rewrite of the flagged windows to measure how the score moves. Use when asked to check a text file's Pangram score, see which passages are flagged, or run the unpangram rewrite experiment.
---

# unpangram

A research CLI in this repo. It measures whether rewriting Pangram-flagged windows with a rotation
of models changes the Pangram score. It does not loop until the text passes.

## When to use

- The user wants a Pangram score for a text file, or wants to see which passages are flagged.
- The user wants to run the rewrite experiment on a file, or on freshly generated text.

Do not use it for text under 50 words (Pangram refuses it), and do not run it without the user's
go-ahead: every call costs real money.

## Requirements

- `uv sync` has been run in the repo.
- `PANGRAM_API_KEY` is set (both commands). `OPENROUTER_API_KEY` is set (`rewrite`).
- If a key is missing the CLI exits with code 1 and names the variable. Ask the user to set it;
  never write keys into files.

## Commands

```sh
uv run unpangram score <file>
uv run unpangram rewrite <file> [--rounds N] [--models a,b,c] [--threshold 0.8] \
    [--max-usd 1.0] [--output-dir DIR]
uv run unpangram rewrite --generate "<prompt>" [--generate-model MODEL]
uv run unpangram --config path.yaml rewrite <file>
```

- `score` prints prediction, AI / AI-assisted / human fractions, the Pangram version, each flagged
  window with its offsets, then the text with flagged windows marked as `[[label: ...]]`.
- `rewrite` prints one line per round and a final summary, and writes `before.json`, `after.json`
  and `final.txt` (plus `generated.txt` with `--generate`) to the output directory
  (default `unpangram-out/`).

Defaults: 2 rounds, similarity threshold 0.8, $1.00 cap. See the README for the YAML config keys.

## Cost

Pangram costs $0.05 per started 100 words per scoring call, and `rewrite` scores the document once
per round plus once up front. Tell the user the expected cost before running, and pass `--max-usd`
if they give a budget. The run stops on its own when the cap would be exceeded.

## Reading the result

Compare `before.json` and `after.json`: `prediction_short`, `fraction_ai`, `fraction_ai_assisted`,
`fraction_human`, and the per-window `label` values. Report the stop reason the CLI printed (fully
Human, round limit, or cost cap) and the estimated spend. Report the numbers as measured, including
when the score did not move. Detectors change, so quote the Pangram `version` with any result.
