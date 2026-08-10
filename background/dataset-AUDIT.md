# Natural dataset literature audit

Status: complete for the bounded selection question; no model weights were loaded and no training was run.

## Decision

Use GSM8K `main` as the primary natural instruction-tuning task.

Task mapping:

- input: `Solve this grade-school math word problem. Show a concise derivation and end with #### <number>.` followed by the GSM8K `question`;
- target: the original GSM8K `answer`, including its natural-language derivation and final `#### number` line;
- primary selection signal: final validation loss, choosing between token-mean and example-mean LoRA arms;
- final task metric: exact final-number match, using the dataset authors' `####` parser. Missing or malformed final answers count as wrong.

The recommended local pilot is a frozen 800/200/200 slice:

| split | source | rows | purpose |
| --- | --- | ---: | --- |
| train | official GSM8K train | 800 | LoRA training |
| validation | disjoint carve-out of official GSM8K train | 200 | loss-normalization selection only |
| test | untouched official GSM8K test | 200 | sealed endpoint comparison |

This is small enough to be a first Mac experiment while preserving a meaningful response-length range. If the effect is non-inconclusive, the leader should rerun the sealed endpoint on the complete 1,319-row official test split rather than treating the 200-row pilot as a final generalization claim.

## Frozen selection rule

No result-dependent selection was used. For each original split independently:

1. Normalize a question by Unicode-preserving whitespace collapse and case-folding.
2. Sort rows into ten contiguous deciles by raw answer-character length.
3. Within each decile, order by SHA-256 of the normalized question.
4. From each train decile, take the first 80 rows for train and the next 20 for validation.
5. From each test decile, take the first 20 rows for test.

The rule is deterministic, length-aware, and does not inspect model outcomes. The exact source paths, hashes, and reproducibility code are in `SOURCES.md`, `subset_spec.json`, and `audit_subset.py`.

## Evidence from the local cached source

The reusable local GSM8K cache contains 7,473 train and 1,319 test examples. The full-source audit found:

- zero normalized-question overlap between train and test;
- zero duplicate normalized questions within either split;
- zero shared first-80-character question prefixes between train and test;
- zero exact answer-string overlap between train and test;
- every row has a non-empty final answer after `####`;
- full train response length: Qwen tokenizer median 111 target tokens, range 29-453;
- full test response length: Qwen tokenizer median 115 target tokens, range 27-413;
- full train prompt-plus-target maximum under the fixed pilot prompt: 556 Qwen tokens;
- full test prompt-plus-target maximum: 539 Qwen tokens.

A concurrent model-free lock proposal in this same project folder additionally pins the upstream release to commit `3101c7d`, hashes the raw train/test files, and reports no exact duplicate groups or 5-shingle near-duplicate pairs across all 8,792 records. I treat that as supplementary local provenance, not independent verification; its status remains `PROPOSAL / MODEL-FREE / NOT CONFIRMATORY`.

The Qwen numbers were obtained with the local tokenizer files only. The Qwen3 model weights were not loaded.

For the proposed slice:

| split | rows | target tokens min / median / max | prompt+target min / median / max |
| --- | ---: | ---: | ---: |
| train | 800 | 33 / 111 / 348 | 91 / 193 / 489 |
| validation | 200 | 41 / 103 / 352 | 102 / 182 / 470 |
| test | 200 | 42 / 112.5 / 346 | 88 / 191 / 474 |

There are no prompt-plus-target examples above 768 or 1024 tokens in the slice. Exact chat-template and end-of-sequence accounting must still be checked by the eventual trainer before the protocol is frozen.

## Why GSM8K fits the question

The official repository describes 8.5K human-written, linguistically diverse grade-school problems, split into roughly 7.5K train and 1K test, with 2-8-step solutions and natural-language answers. The repository explicitly defines final-answer extraction after `####`, and its evaluator compares the extracted numeric answer. This gives a task that is natural enough to differ from synthetic associative recall, has variable supervised response length, and has an objective evaluator that does not reward verbosity.

The dataset card and repository both identify MIT licensing. Preserve the MIT notice and dataset citation in any derivative local artifact. This is a public-use determination from the dataset's primary release metadata, not legal advice about every downstream representation.

## Evaluator validity and protocol boundary

The evaluator is valid for the stated narrow capability: whether the model reaches the labeled final numeric answer. It does not validate the intermediate reasoning, and exact match can hide a flawed derivation. Therefore the eventual run should retain both:

- final-number exact match as the frozen primary endpoint;
- teacher-forced target-token NLL and exact match by response-length decile as diagnostics.

Selection must use validation loss only. Do not inspect test exact match, test NLL, or per-length test results before sealing the selected arm. A malformed completion without `####` is invalid/wrong; do not repair it with a second model call.

Important limitations:

- the 200-row pilot test is noisy and is a screening result, not a publication-grade estimate;
- GSM8K is a math reasoning task, not broad assistant instruction following;
- the official split is train/test, so the proposed validation split is a documented train-only carve-out;
- pretraining contamination of Qwen3 by GSM8K is not established by this audit;
- source-level exact deduplication does not rule out semantic paraphrases or benchmark contamination;
- the target includes author-provided calculation annotations, which may be removed only as a predeclared task amendment, not after seeing outcomes.

## Fallback

Use SQuAD v2 only if the leader decides that a reading-comprehension task is a more useful natural transfer test. It has explicit train/validation splits, crowdsourced natural questions, answerable and unanswerable cases, CC BY-SA 4.0 metadata, and an official exact/F1 evaluator. Derive a small 800/200/200 slice from its 130,319/11,873 train/validation rows, with no answer text or question overlap across splits checked before training.

SQuAD v2 is a weaker fit for this particular loss-normalization question: its answer targets are mostly short spans or empty strings, so it supplies less response-length variation and less natural derivation structure. It is a validity/licensing fallback, not the primary recommendation.

SVAMP was also screened because it is MIT and has 1,000 natural math word problems, but its release is a challenge/evaluation set rather than a clean train/validation package; its own README warns that the source benchmarks it diagnoses admit shallow heuristics. It is better reserved as a secondary stress test than as the primary LoRA training dataset.

## Mac feasibility

The local machine has already hosted the relevant MLX stack and a local 3-bit Qwen3-0.6B asset. The existing loss-normalization record reports an M4 run with `mlx-lm==0.31.3`, 576-update LoRA arms, about 0.73 GB peak MLX memory, and roughly 16 seconds per arm on the much smaller prior fixture. That is feasibility evidence, not a runtime prediction for GSM8K.

The proposed slice has a maximum of 489 prompt-plus-target Qwen tokens before chat-template overhead and is only 1,000 training/validation rows. It is a reasonable first local run, but the lab's one-model-slot rule still applies: serialize model work, inspect memory/process pressure immediately before launch, and never run two Qwen/MLX trainers concurrently. No cloud, spend, upload, publication, or model run is authorized by this audit.

## Handoff to the initiative leader

Primary recommendation: freeze GSM8K `main` with the deterministic 800/200/200 slice and compare token-mean versus example-mean LoRA training under the existing validation-selection rule. Use exact final-number match as the endpoint, retain target-token NLL and length-bin diagnostics, and escalate to the full official test only if the pilot is not Inconclusive.

Before any training, the leader should inspect `subset_spec.json`, rerun `audit_subset.py` against the existing cache and local tokenizer, freeze the exact chat template/mask/optimizer/seed budget, and record the generated slice hashes. This project does not authorize model loading or training; it hands off a ready, source-audited data decision.
