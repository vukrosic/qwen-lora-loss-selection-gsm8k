# Verification checklist

This checklist is the handoff gate for the initiative leader and the
implementation/preflight lane. Every unchecked item is a stop condition
before a model process or test inspection.

## A. Corpus and split freeze

- [ ] Source, version/commit, license/provenance, raw hash, filtered hash, and
      field mapping are recorded.
- [ ] The materialized corpus has exactly 768 train, 128 validation, and 256
      test records, or a pre-model amendment is recorded.
- [ ] Stable IDs are unique; exact normalized duplicates are removed and the
      removal ledger is preserved.
- [ ] Near-duplicate/conversation/task-family groups are explicit, and no
      group ID appears in more than one split.
- [ ] The deterministic split algorithm and hash manifest reproduce the same
      record IDs from a clean read-only copy.
- [ ] Fixed target-length buckets are computed after the final renderer and
      tokenizer; at least two have the required validation/test counts.
- [ ] Every included prompt plus complete target fits 256 tokens; no target is
      truncated.
- [ ] Category and length distributions are reported without using test
      outcomes to rebalance them.

## B. Protocol and implementation freeze

- [ ] Protocol, trainer, evaluator, renderer, tokenizer, model, and data
      hashes are captured before any test output is inspected.
- [ ] Token-mean and example-mean aggregation pass a deterministic toy oracle:
      unequal completion lengths produce the expected different batch losses,
      while equal lengths agree.
- [ ] Completion masks cover response tokens only; prompt tokens contribute
      zero loss; EOS handling is identical across arms.
- [ ] Both arms use the same seed-specific initialization, data order,
      minibatches, optimizer, update count, and LoRA placement.
- [ ] The final checkpoint is fixed at update 576; no intermediate checkpoint
      selection or early stopping is enabled.
- [ ] The validation selector reads no test IDs, labels, generations, or test
      metrics and applies the exact balanced-example-NLL macro rule.
- [ ] A numerical tie within `1e-9` selects token mean.
- [ ] The test evaluator refuses to start without an immutable selection record
      for that seed and matching protocol/data hashes.
- [ ] Greedy decoding and all metric normalizers are deterministic and have
      standalone toy tests.
- [ ] Test metrics, bootstrap intervals, and classifications are computed only
      after all three selection records are sealed; no gate is edited then.

## C. Fresh-seed and run-order controls

- [ ] Seeds `20260821`, `20260822`, and `20260823` are confirmed unused for
      corpus choice, metric design, implementation tuning, or prior evidence.
- [ ] The predeclared alternating arm order is recorded and observed.
- [ ] Each arm/seed has a unique run directory; an existing directory cannot
      be overwritten.
- [ ] A launcher-only failure before any loss/metric is preserved separately;
      a corrected rerun uses the same seed and a new directory, never a silent
      replacement.
- [ ] No seed is added, removed, or substituted after freeze.

## D. Resource and evidence receipts

- [ ] Current memory pressure is checked before each process; only one
      MLX/Qwen process runs at a time; roughly 20% memory headroom is retained.
- [ ] Each arm stays within 576 updates, 30-minute training cap, and the
      approximately 4-hour total budget.
- [ ] Command, interpreter/import receipt, timestamps, PID/status, memory/time
      receipt, loss log, adapter hash, and protocol/data/trainer hashes are
      preserved.
- [ ] All six arms finish with finite loss and matching update/example counts.
- [ ] Every arm evaluates all 256 test records; the base diagnostic uses the
      same records and is clearly labelled non-selection evidence.

## E. Metrics and gates

- [ ] Raw per-example validation and test NLL are preserved before aggregation.
- [ ] Raw generations and references are preserved before ROUGE-L/token-F1
      aggregation.
- [ ] The frozen lexical tokenizer, ROUGE-L LCS calculation, prompt-echo
      prefix rule, repeated-4-gram rule, and length-ratio denominator match
      the protocol exactly.
- [ ] Balanced length-bucket macro means are used consistently; empty buckets
      are handled only by the frozen active-bucket rule.
- [ ] Non-empty, prompt-echo, repetition, and nonzero-overlap diagnostics are
      computed for selected, always-token, and always-example policies.
- [ ] No-floor/informativeness gates pass; otherwise classification is
      `INCONCLUSIVE / UNINFORMATIVE`.
- [ ] Per-seed non-inferiority, strict-win counts, large-reversal checks, mean
      aggregate margins, and worst-seed checks are calculated exactly as
      frozen, with no favorable-seed filtering.
- [ ] Fixed-seed bootstrap intervals use 5,000 resamples and seed 41001 as
      descriptive evidence only.
- [ ] Classification is exactly one of `SUPPORTED`, `NEGATIVE`, `MIXED`, or
      `INCONCLUSIVE`, with the failed gate and evidence pointer recorded.
- [ ] Final scope statement names the one corpus, model/configuration,
      renderer, tokenizer, and three-seed budget; no broad transfer claim is
      made.

## Stop conditions

Stop and report a concrete blocker if any of the following occurs: a split
leakage or hash mismatch; unavailable provenance/license; fewer than two
usable length buckets; target truncation; a mask/aggregation oracle failure;
protocol or evaluator change after test inspection; missing or invalid arm;
OOM/timeout/NaN that prevents the full paired comparison; or degenerate
outputs that fail the no-floor gate. Do not turn these into a negative
scientific result and do not silently repair them by changing the frozen
question.
