# Canonical Protocol N1-GSM8K

Status: **FROZEN BEFORE SUBSET MATERIALIZATION, TOKENIZER RESULTS, OR MODEL RUN**  
Date: 2026-08-10 (Asia/Singapore)

This is the sole canonical protocol for this initiative. The sibling evaluator
folder contains two preserved recommendations written by separate Luna tasks;
their incompatible choices must not be mixed into this protocol.

## Question and claim boundary

On one fixed small subset of human-written GSM8K, can validation token-weighted
completion NLL select between token-mean and example-mean Qwen3-0.6B 3-bit LoRA
training better than either fixed endpoint across three fresh paired seeds?

Any conclusion is bounded to the pinned corpus subset, local model conversion,
renderer, LoRA configuration, optimizer, and seed budget. GSM8K is a public 2021
benchmark, so final accuracy is a within-dataset transfer signal, not an
uncontaminated arithmetic-generalization claim.

## Data

Use only the source-aware 1,000 train / 256 validation / 512 final-test design
in `DATA-DESIGN.md`, pinned to OpenAI `grade-school-math` commit
`3101c7d5072418e28b9008a6636bde82a006892c`. Train comes only from upstream
train. Validation and final are disjoint subsets of upstream test. Selection is
deterministic under seed 20260810, balanced short/long, exact-duplicate audited,
and cross-partition near-duplicate safe. The immutable dataset lock must contain
source, script, split, manifest, tokenizer, renderer, and protocol hashes.

Every included rendered prompt plus complete target must fit 256 tokens. No
target truncation is allowed. Completion tokens only are supervised; prompt and
padding targets have zero mask weight.

## Arms, seeds, and order

The scientific difference is only batch loss aggregation:

- token: sum completion-token losses divided by supervised completion tokens;
- example: mean each example's completion-token loss, then mean examples.

Both arms share model, initialization seed, minibatches/order, optimizer,
schedule, update count, checkpoint, mask, and renderer. Constants are rank 8,
scale 20, dropout 0, final 8 linear layers, AdamW 1e-4, weight decay 0, batch 4,
no gradient accumulation, maximum sequence length 256, and final endpoint after
exactly 576 optimizer updates. No early stopping or intermediate checkpoint
selection.

Fresh confirmation seeds and arm order are exactly:

| Seed | First | Second |
| ---: | --- | --- |
| 20260820 | token | example |
| 20260821 | example | token |
| 20260822 | token | example |

No seed replacement, supplementation, or import from the competing 20260823
draft is allowed.

## Prospective selection and test order

After both arms for one seed finish, evaluate both on the complete validation
set using one common statistic independent of the training objective:
token-weighted completion NLL. Lower wins; an exact numerical tie selects token.
Write an overwrite-safe selection record with both values and all hashes before
either arm can access the final-test file.

Only after sealing selection, evaluate both arms on all 512 final records.
Teacher forcing preserves per-record completion NLL. Greedy generation uses
temperature 0, EOS stopping, and at most 256 new tokens. Preserve raw generated
text. The unadapted base model is a diagnostic only and is never selectable.

## Endpoints

Primary: final-test macro-example completion NLL, the mean of each record's own
completion-token mean. Report selected-minus-unselected per seed plus selected,
always-token, and always-example mean and worst-seed values.

Complementary capability endpoint: strict GSM8K final-number exact match. Extract
only the first numeric value following `####`; remove commas and compare Decimal
values, including trailing `.0` equivalence. A missing marker is a format
failure and never falls back to another number. Report marker rate separately.
This endpoint is a distinct generated-answer measurement of the same held-out
question/answer pair, not an independent construct and not a rationale check.

Diagnostics: token-weighted test NLL, short/long macro NLL and exact match,
target-token accuracy, generated length, wall time, and peak MLX memory. ROUGE-L
and lexical overlap from the competing draft are not gates and need not be run.

## Validity, informativeness, and classification

All six arms must exit 0, contain finite losses, complete exactly 576 updates,
load matching adapters, match every frozen hash, and evaluate every required
record. Per-record evidence is mandatory. Selection timestamps must precede
final-test timestamps. Only one MLX/Qwen process may run at a time with roughly
20% system-memory headroom.

NLL is uninformative if all per-record arm values are identical within 1e-8 or
both arms tie within 1e-6 on every seed. Exact answer is uninformative if it is
all zero or all one for both arms across every seed. A would-be positive result
that fails either informativeness gate is Inconclusive.

- Supported bounded transfer: all validity/informativeness gates pass; selected
  primary NLL is lower than unselected on at least 2/3 seeds; selected mean and
  worst-seed primary NLL are strictly lower than both fixed policies; selected
  complementary exact answer is no worse than unselected on at least 2/3 seeds and strictly
  better on at least one.
- Negative bounded result: validity/informativeness pass; selected primary NLL
  loses on at least 2/3 seeds and selected mean and worst-seed primary NLL do
  not beat either fixed policy; exact-answer evidence does not reverse that
  direction.
- Mixed: validity/informativeness pass but neither conjunction holds, including
  conflicting NLL and exact-answer directions.
- Inconclusive: any missing pair, leakage/hash/mask/order defect, target
  truncation, nonfinite result, resource interruption, post-freeze deviation,
  or uninformative endpoint needed for a positive claim.

Stop after three valid paired seeds are classified, or earlier for a preserved
blocker. Do not change data, selector, seeds, endpoints, or gates after final
test evidence exists.

The strongest positive wording permitted is **supported bounded transfer on the
locked <=256-token GSM8K subset with a complementary final-answer endpoint**.
It is not broad natural-instruction transfer.

## Amendment A1 — tokenizer eligibility before training

Status: **FROZEN AFTER TOKENIZER-ONLY V1 FAILURE, BEFORE ANY MODEL-WEIGHT LOAD**

The first deterministic materialization passed source, count, exact-duplicate,
and cross-partition near-duplicate checks, but tokenizer-only preflight found
that 182/1,000 train, 58/256 validation, and 96/512 final records exceeded 256
tokens. `DATASET-LOCK-v1.json` and `preflight/tokenizer-v1/` preserve this failed
attempt. Training V1 would violate the zero-truncation validity gate.

Keep maximum length 256 and all scientific seeds, arms, metrics, and gates.
Before deterministic sampling, filter the complete pinned upstream train and
test pools to records whose fixed Qwen chat rendering (with thinking disabled)
fits at most 256 tokens. Record every excluded source ID and length, tokenizer
fingerprint, and script hash. Then reapply the same source-aware counts,
short/long rank balance, SHA-256 seed 20260810, duplicate audit, and no-leakage
checks to create V2. This is a resource/validity correction based only on input
lengths, not losses or generated/test outcomes. The claim is correspondingly
bounded to the <=256-token GSM8K subset.

## Amendment A2 — skeleton audit and natural ordering

Status: **FROZEN BEFORE MODEL-WEIGHT LOAD**

Hash a question-only skeleton for every chosen record after NFKC normalization,
lowercasing, whitespace collapse, and replacement of signed integers, decimals,
percentages, and currency amounts with `<NUM>`. Preserve every cross-partition
skeleton overlap in `SKELETON-AUDIT.json`; this is a leakage diagnostic and
claim limitation, never an outcome metric.

Training shuffles all 1,000 locked natural records once per epoch with a single
deterministic seed stream. It does not form length-balanced minibatches,
oversample a stratum, or sample with replacement. The two arms for a seed use
identical record indices and batch boundaries. The already-frozen 500/500
short/long subset composition remains a dataset-selection fact only; length is
diagnostic during training and evaluation.
