# Frozen GSM8K data design

Status: **FROZEN BEFORE MATERIALIZATION, TOKENIZER INSPECTION, OR MODEL RUN**  
Date: 2026-08-10 (Asia/Singapore)

## Source

- Corpus: GSM8K `main`.
- Upstream: official OpenAI `grade-school-math` repository.
- Commit: `3101c7d5072418e28b9008a6636bde82a006892c`.
- Source files: `grade_school_math/data/train.jsonl` and `test.jsonl`.
- License: MIT, copied from the same pinned commit.
- Human-written provenance and evaluator rationale are documented by the
  read-only sibling audit at `../natural-dataset-literature-audit/AUDIT.md`.

## Split amendment and rationale

Protocol N1 proposed a corpus-wide grouped 70/15/15 split because it was written
before the corpus was selected. GSM8K already has an official train/test
boundary. The initiative therefore freezes a stricter source-aware design:

- train: 1,000 records sampled only from official train;
- validation: 256 records sampled from official test;
- final test: 512 different records sampled from official test.

The remaining source records are unused. Validation and final test are disjoint,
and no official test record enters training. This gives the cleanest bounded
no-leakage claim while keeping six 576-update arms and exact-answer evaluation
feasible within the eight-hour local window.

Within each source split, records are ranked by answer whitespace-token length
with source ID as tie breaker. The lower half is `short` and upper half is
`long`. Each materialized partition is exactly half short and half long. Within
each stratum, SHA-256 of `20260810:<partition>:<source-id>` fixes selection and
order. No model output, loss, or test result affects inclusion.

Exact duplicate prompt/completion records are removed before sampling with an
exclusion ledger. Selected partitions must also have no cross-partition pair
whose canonical prompt-plus-completion token 5-shingle Jaccard similarity is at
least 0.90; otherwise materialization fails and this design must be amended
before any model run.

## Formatting and evaluator

The user message is the source `question`. The assistant completion is the full
human-written source `answer`, including its final `#### <number>` marker.
Completion tokens only are supervised. The deterministic capability target is
the first number following `####`; commas are removed and values are compared
as decimal numbers, so an integer and the same value with trailing `.0` agree.
Missing markers are format failures and never fall back to a last-number guess.

Tokenizer preflight must use the read-only local Qwen tokenizer, fingerprint it,
verify every record's full formatted length, and enforce maximum length 256
with zero included-record truncation. It may not load model weights. A gate
failure requires a recorded pre-outcome amendment; records are not silently
dropped after tokenization.

Amendment A1 records that V1 failed this gate. For V2, apply the fixed renderer
and pinned tokenizer to the complete upstream pools before sampling, exclude
every source record over 256 tokens with a preserved ID/length ledger, then
repeat the same deterministic split algorithm. No V1 model run exists.
