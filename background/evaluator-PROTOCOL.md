# Prospective natural-data evaluator protocol

Protocol N1 — candidate frozen design, 2026-08-10 (Asia/Singapore)

This project answers one bounded question: **on one small, provenance-clear
natural instruction corpus, can a final-validation rule choose between
token-mean and example-mean completion-loss LoRA training better than either
fixed endpoint?** The conclusion is limited to the locked corpus, Qwen3-0.6B
3-bit MLX LoRA implementation, and this run budget. It is not a claim about
general instruction following or about all data/model combinations.

The prior synthetic result was inspected read-only. Its three-seed selection
and exact-match gates are not copied: natural responses are open-ended, so the
protocol uses canonical held-out NLL plus a predeclared structured-response
endpoint when the selected corpus supports one.

## Freeze point and ownership

No model run is confirmatory until a dataset lock exists beside this protocol.
The lock must contain the corpus name and version/commit, provenance and
license evidence, the formatted-record manifest hash, split hash, tokenizer
fingerprint, counts, response-length summary, and the exact preprocessing
script hash. The evaluator may fill these fields once from the accepted audit;
after the lock, they are immutable. A missing or changed field invalidates the
run as confirmatory rather than silently creating a new split.

The test partition is never read by training, selection, hyperparameter choice,
debugging against outcomes, or an evaluator that writes a choice. Test
evaluation starts only after an immutable per-seed selection record has been
sealed and hashed.

## Data and split

1. Use one versioned corpus selected by the dataset audit. Keep only records
   with a non-empty instruction/context and non-empty assistant completion.
   Preserve source IDs and any task-family or format labels; do not synthesize
   labels after seeing model outputs.
2. Canonicalize Unicode (NFKC), whitespace, and line endings for deduplication
   only. Exact duplicates are removed by SHA-256 of canonical prompt plus
   completion. Near-duplicates are connected components of token 5-shingle
   Jaccard at least 0.90. A source/task-group ID, when supplied, is also an
   indivisible group. No group may cross partitions.
3. Assign groups deterministically to train/validation/test in a 70/15/15
   stratified split using the frozen hash seed `20260810`, balancing declared
   task family and pre-tokenization completion-length quartile. Store the
   resulting ordered IDs and all hashes. There is no reshuffle after seeing a
   loss or generation.
4. The corpus is suitable only if the locked split has at least 300 records,
   at least 40 validation and 40 test records, at least 8 records per retained
   task family in validation and test, two or more non-empty completion-length
   strata, and a test 90th/10th completion-token-length ratio of at least 1.5.
   If these conditions cannot be met without hand selection, stop as a design
   blocker.
5. Format each record once with the frozen chat template. Supervise completion
   tokens only; prompt/context and padding are masked. With maximum sequence
   length 256, reject any corpus whose fixed formatting would truncate more
   than 1% of train, validation, or test records. The exact truncation count is
   reported even when zero.

## Training arms and endpoint

Both arms start from the same read-only base model and independently seeded
LoRA initialization. They share the exact ordered minibatches, optimizer,
learning rate, schedule, update count, sequence limit, and checkpoint timing.
The only scientific difference is completion-loss aggregation:

* **token mean:** sum supervised-token losses in the minibatch divided by the
  minibatch's supervised-token count;
* **example mean:** compute each example's supervised-token mean, then average
  those example means equally.

Candidate constants to freeze in the dataset lock/preflight receipt are the
  prior implementation's rank 8, scale 20, dropout 0, final 8 linear layers,
  AdamW learning rate `1e-4`, zero weight decay, batch size 4, no gradient
  accumulation, maximum length 256, and exactly 576 optimizer updates. If the
  implementation cannot honor these constants, do not alter them after an
  outcome; record a versioned protocol amendment and treat old runs as
  non-confirmatory.

Fresh confirmation seeds are fixed before training: **20260820, 20260821, and
20260822**. They are not seeds from the synthetic initiative. To counterbalance
thermal/order effects, run token then example for 20260820, example then token
for 20260821, and token then example for 20260822. Each run has its own
directory and refuses overwrite. There is no early stopping, checkpoint choice,
blend, or hyperparameter tuning on validation.

## Selection rule (validation only)

At update 576, evaluate both adapters on the complete validation split with a
single canonical statistic independent of the arm's training-log
normalization: **token-weighted completion NLL**, i.e. total supervised-token
negative log likelihood divided by total supervised-token count. Choose the arm
with the lower value; an exact numerical tie chooses token mean. Record both
values, the rule, seed, protocol/data/tokenizer hashes, adapter hashes, and run
receipts in an immutable selection JSON before opening either test file.

This is deliberately not the test statistic. The confirmatory primary is
macro-example NLL on test (below), and a positive transfer also requires an
independent structured-response endpoint where the corpus supplies one. Thus
validation ranking cannot by itself make the claim tautological.

## Test metrics and controls

Evaluate both arms on every test record after selection, with greedy decoding
(temperature 0), a fixed maximum of 128 new tokens, and no sampling or prompt
changes. Preserve raw per-example teacher-forced losses and generated text.

* **Primary endpoint:** macro-example completion NLL (mean of each record's
  completion-token mean). Report paired selected-minus-unselected differences
  per seed, mean and worst-seed values, and a test-record bootstrap 95% interval
  that was specified before looking at results.
* **Independent capability endpoint (required for a positive transfer):** if
  the locked corpus contains a deterministic/structured subset (for example,
  exact label, JSON-schema, or code-format records), use its predeclared exact
  or format-valid rate, with the subset IDs fixed in the dataset lock. If no
  such subset exists, mark this endpoint unavailable and do **not** upgrade an
  NLL-only win to broad natural-instruction success.
* **Diagnostics:** token-weighted test NLL; length-stratum and task-family
  macro NLL; non-empty-generation, prompt-echo, and declared-format validity;
  target-token accuracy only where target spans are native corpus metadata;
  generation length and wall time/peak memory. Diagnostics cannot replace a
  failed primary or unavailable independent endpoint.

The two fixed policies (always token and always example) are evaluated from the
same paired runs. The validation-selected policy is compared with both. A
no-training base-model evaluation on validation and test is a diagnostic
control, never a selectable arm. No random or post-hoc selector is introduced.

## Validity and no-floor gates (frozen before outcomes)

Every one of the six arm runs must have an exit-0 receipt, finite losses,
loadable adapter, matching frozen hashes, exactly 576 updates, and complete
validation/test records. A failed infrastructure-only run may be retried once
with identical code/settings; an implementation or protocol change invalidates
that pair. At least three valid paired seeds are required for any directional
classification.

The split must pass the group/dedup, provenance, mask, and truncation checks
above. Per-example test evidence must be present; aggregate-only logs are
invalid. Validation selection records must precede test timestamps. The model
slot must remain one MLX/Qwen process at a time with approximately 20% memory
headroom; interruption, swap pressure, or another initiative's slot claim
means preserve partial evidence and classify inconclusive.

The independent endpoint is informative only if it is not all zero or all one
for both arms across every seed. For NLL, reject a floor-only comparison when
all per-example values are identical within `1e-8` or when both arms tie within
`1e-6` on every seed. These checks prevent a saturated metric from masquerading
as transfer.

## Predeclared classifications and stopping

* **Supported bounded transfer:** all validity/no-floor gates pass; the
  validation-selected arm has lower primary test macro-example NLL than the
  unselected arm on at least 2 of 3 seeds; its mean and worst-seed primary NLL
  are strictly lower than both fixed policies; and it is no worse than the
  unselected arm on the independent structured endpoint on at least 2 of 3
  seeds (with a strict win in at least one). This is a bounded natural-data
  transfer result only.
* **Mixed:** all validity gates pass but the selected policy has conflicting
  primary/independent directions, wins on only 1 seed, or improves one endpoint
  while materially worsening the other.
* **Negative:** all validity gates pass, the selected policy loses primary NLL
  on at least 2 of 3 seeds and is not better than either fixed policy in mean
  or worst-seed primary NLL.
* **Inconclusive:** any validity failure, fewer than three valid pairs, a
  resource interruption, a floor-only/unavailable independent endpoint for a
  would-be broad claim, or any unregistered protocol/data change. Report exact
  observed differences but do not relabel them by tuning thresholds.

Stop after the three valid paired seeds are evaluated and classified, or
earlier only for a recorded blocker/resource limit. Do not add seeds, replace
the corpus, alter the tie rule, or tune the endpoint after seeing test output.

## Compute and evidence receipt

The planned envelope is six serialized training runs (2 arms × 3 seeds × 576
updates) plus twelve arm evaluations and three base-model diagnostics, on the
single local M4 MacBook. No cloud or rented compute is authorized. Before each
run, inspect memory pressure and competing model processes; after exit, release
the slot. Each run records command, timestamps, PID, environment/library
fingerprints, protocol/data hashes, seed/condition, stdout/stderr, adapter
hash, per-example metrics, and peak-resource receipt. Preserve failed and
partial runs; never overwrite them.

