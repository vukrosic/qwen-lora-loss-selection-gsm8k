# Prospective natural-data evaluator protocol recommendation

Status: **candidate ready for leader freeze; no model run authorized by this
project**.

This protocol answers a narrow question: on one frozen, small natural
instruction corpus and one fixed Qwen3-0.6B 3-bit MLX LoRA configuration, does
the lower-validation-loss choice between token-mean and example-mean training
predict a held-out arm that is at least as good on predictive loss and
generated-answer quality?

The prior synthetic protocol is evidence for the selection workflow, not a
source for natural-data task gates. In particular, its held-out prompt
templates, strict exact match, and exact-match tolerance are not reused as the
natural claim's decisive criteria.

## 1. Freeze before any test output

The leader should freeze one protocol hash, one corpus manifest, one renderer,
one trainer/evaluator version, and three fresh seeds before either arm's test
outputs or test metrics are inspected.

The freeze record must contain:

- the corpus source, version/commit, license/provenance note, raw-file hash,
  filtered-file hash, and exact field mapping;
- the split manifest and group assignments, including record IDs, split,
  source/task category, and target-token length;
- the prompt renderer, completion mask, tokenizer version, and maximum length;
- base-model identifier and hash, LoRA modules/rank/scale/dropout, optimizer,
  learning rate, weight decay, batch size, update budget, checkpoint rule,
  decoding rule, and metric implementations;
- the selection statistic, tie rule, seed list, arm order, gates, and failure
  classes; and
- the run directory and evidence naming convention.

Any change after a test output is inspected invalidates that confirmation
attempt. It may be recorded as a new exploratory protocol, but not blended
with this one.

## 2. Natural-data eligibility and split semantics

The dataset audit supplies a materialized corpus before model work. The
protocol accepts it only if all of the following are true:

1. It is genuinely natural, human-written instruction/response data with a
   provenance-clear source and a recorded license decision.
2. Each record has a stable source ID, instruction, optional context/input,
   response, and an auditable group ID. A group is a conversation lineage,
   source task family, paraphrase/variant family, or a deterministic
   near-duplicate component. If no defensible group ID can be produced, reject
   the corpus for this test rather than pretending that instance-level random
   splitting is independent.
3. Exact duplicates are removed before splitting using Unicode NFKC,
   whitespace collapse, and case-folded concatenated instruction/context/
   response keys. Near-duplicate groups are retained only within one split.
   The audit records the detector and all removals; original text is preserved
   read-only for provenance.
4. After fixed rendering and tokenization, every included prompt plus complete
   target fits `max_seq_len = 256`. No target truncation is allowed. Records
   that fail this check are filtered before the final split and the resulting
   corpus hash is frozen; post-freeze filtering is invalid.
5. The final materialization has exactly 768 train, 128 validation, and 256
   test records, unless the leader freezes a documented amendment before any
   model run. At least two fixed target-length buckets have at least 8
   validation and 16 test records each.

The recommended fixed target-length buckets are:

| Bucket | Completion tokens |
| --- | ---: |
| short | 1–64 |
| medium | 65–128 |
| long | 129–256 |

Only buckets meeting the minimum counts above are active. Fewer than two
active buckets is a data-design failure, not a reason to merge bins after
seeing results.

Splitting is group-disjoint and deterministic. Within each declared category
and active length stratum, sort groups by
`SHA256("natural-evaluator-split-v1|" + group_id)` and allocate groups to
train, validation, and test until the frozen record counts are reached. If a
group-level allocation cannot satisfy the counts without breaking a group,
stop and amend the split before training. The test split is never used for
sampling, balancing, hyperparameter choice, checkpoint choice, or rule choice.

The trainer sees only train records. The selector sees only validation records
and the final validation statistic. The test evaluator is a separate phase
that refuses to run unless an immutable per-seed selection record exists.

## 3. Intervention, matched controls, and fresh seeds

The only intervention is the loss reduction:

- **token mean:** one mean over all supervised completion tokens in a batch;
- **example mean:** compute each example's supervised-token mean first, then
  average examples in the batch.

Everything else is matched: same base model, initialization seed, shuffled
record order, minibatches, masks, optimizer, updates, LoRA modules, and
decoder. The two arms run in separate unique directories. Run order alternates
to expose order/environment defects:

| Seed | First arm | Second arm |
| ---: | --- | --- |
| 20260821 | token mean | example mean |
| 20260822 | example mean | token mean |
| 20260823 | token mean | example mean |

These seeds are confirmation-only and must not have been used for dataset
selection, metric design, implementation tuning, or prior LoRA evidence. They
cannot be replaced or supplemented after freeze. A same-seed rerun is a
reproducibility repair, not a new confirmation seed, and must preserve the
failed original record.

For every seed, both arms are trained and both are evaluated. This yields the
prospective selected policy plus the counterfactual fixed policies:

- selected policy: the arm chosen from validation for that seed;
- always-token policy: token arm on every seed;
- always-example policy: example arm on every seed.

After selection records are sealed, also evaluate the unadapted base model as
a descriptive diagnostic. It is not a selection input and is not a success
gate.

## 4. Selection statistic and evaluation metrics

### Selection statistic (validation only)

At the fixed final update, compute per-example teacher-forced negative
log-likelihood over completion tokens, then take a macro mean over active
length buckets. This is called **balanced validation example NLL**. A lower
value wins. It is deliberately not raw batch token loss and is not identical
to either training reduction. A difference within `1e-9` is a numerical tie;
the predeclared tie rule selects token mean.

No validation generation, intermediate checkpoint, training loss, test metric,
or post-hoc blend may influence selection.

### Held-out metrics (test only after selection)

The two co-primary test metrics are:

1. **Balanced test example NLL**: the same length-bucket macro construction,
   computed on held-out gold responses. Lower is better. This measures
   predictive fit without making exact string agreement the sole criterion.
2. **Balanced generated ROUGE-L F1**: greedy generated answers compared with
   the reference responses after only frozen Unicode normalization and
   whitespace/punctuation tokenization. Higher is better. No answer-specific
   cleanup or per-task metric tuning is allowed.

For reproducibility, the lexical tokenizer is `re.findall(r"\\w+|[^\\w\\s]",
NFKC(text).casefold(), flags=UNICODE)`. ROUGE-L F1 is the usual longest-common-
subsequence F1 on those tokens; lexical token F1 uses multiset overlap. The
diagnostic definitions are also fixed: an output is non-empty when it has at
least one lexical token; prompt-echo is true only when the first eight output
tokens exactly equal the final eight rendered-prompt tokens; and repeated-
4-gram rate is the fraction of output tokens belonging to a 4-gram whose
normalized count is at least two. Length ratio is generated-token count divided
by reference-token count.

Report these diagnostics for each arm and policy, but do not use them to alter
the gates:

- lexical token F1, as a second overlap view;
- non-empty response rate, prompt-echo rate, repeated-4-gram rate, and output
  length/reference-length ratio;
- exact match, explicitly labelled brittle and not decisive; and
- each active length bucket and declared task category separately.

The paired per-record differences and a fixed-seed bootstrap 95% interval
(5,000 resamples, bootstrap seed `41001`) are reported for both co-primary
metrics. These intervals are descriptive and cannot replace the frozen gates.

## 5. Training, decoding, and local compute budget

The recommended natural-data budget is:

- read-only Qwen3-0.6B 3-bit base model already available locally;
- LoRA rank 8, scale 20, dropout 0, and the same fixed target modules as the
  prior endpoint comparison;
- AdamW, learning rate `1e-4`, weight decay 0, batch size 4, no gradient
  accumulation;
- 3 complete passes over 768 train records = exactly 576 optimizer updates
  per arm and 3,456 total training updates across six arm/seed runs;
- maximum sequence length 256, completion-only supervision, no target
  truncation; and
- greedy decoding, temperature 0, maximum 256 new tokens, stop at EOS.

Hard resource limits are one MLX/Qwen process at a time, six training runs,
three per-seed base-model diagnostics, and no automatic seed replacement.
Reserve at most 30 minutes wall time per training arm and 10 minutes per
validation/test evaluation phase; the full model budget is approximately 6
hours including setup and receipts. This is a ceiling, not authorization to
extend the run or add seeds.
Before every model process, inspect memory pressure. Stop launching work if
free memory approaches the lab's 20% headroom boundary, compression/swap is
rapidly rising, or the Mac becomes unresponsive. A timeout, OOM, NaN, or
launcher defect is preserved and classified; it does not justify silently
changing sequence length, batch size, update count, or seed.

## 6. Frozen gates

### Validity gates (all required)

- The corpus is provenance/licence accepted, group-disjoint, duplicate-audited,
  count-correct, and hash-matched.
- All six arm runs complete the same 576 updates with finite losses and no
  target truncation; both arms use the same seed-specific data/init receipt.
- All six arm evaluations cover exactly the 256 test records, and the base
  diagnostic covers the same records.
- Each test evaluation starts only after its seed's selection record is
  sealed with validation metrics, adapter hash, trainer/data/protocol hashes,
  and timestamps.
- No protocol, evaluator, split, seed, hyperparameter, checkpoint, or metric
  change occurred after test inspection.

### Informativeness / no-floor gates (all required)

For each of the selected, always-token, and always-example policies, across
the full 256-record test set:

- non-empty rate is at least 0.80;
- prompt-echo rate is at most 0.20; and
- at least 0.20 of records have nonzero ROUGE-L F1 or lexical token F1.

These are guardrails against declaring success when every policy is an empty,
copied, or zero-overlap output system. If they fail, classify the run
**INCONCLUSIVE / UNINFORMATIVE**, even if relative numbers appear favorable.

### Per-seed transfer gates

For every seed, the validation-selected arm must be non-inferior to the other
arm on both co-primary test metrics:

- selected NLL ≤ other-arm NLL + `0.02` nats/example; and
- selected ROUGE-L F1 ≥ other-arm ROUGE-L F1 − `0.02`.

The tolerance is a frozen practical margin, not a post-hoc confidence
threshold. At least two of the three seeds must be strict wins for NLL and at
least two must be strict wins for ROUGE-L F1. The selected arm may not exceed
`0.05` NLL loss or fall below `0.05` ROUGE-L F1 on any seed; such a large
reversal is a per-seed failure.

### Aggregate transfer gates

Average metrics equally over the three seeds. The selected policy must:

- have mean NLL at least `0.01` nats/example lower than both fixed policies;
- have mean ROUGE-L F1 at least `0.01` higher than both fixed policies; and
- be no worse than each fixed policy by more than the per-seed tolerance on
  worst-seed NLL and worst-seed ROUGE-L F1.

Exact match cannot rescue a failure of either co-primary metric.

## 7. Precommitted classifications

| Classification | Rule |
| --- | --- |
| **SUPPORTED, bounded transfer** | All validity and no-floor gates pass; all per-seed non-inferiority gates pass; strict-win counts pass; and all aggregate gates pass. Claim only one frozen corpus/configuration. |
| **NEGATIVE, bounded failure** | All validity and no-floor gates pass, but the selected policy is worse than both fixed policies on at least one co-primary aggregate metric by the frozen margin and fails non-inferiority on at least two seeds for that metric. |
| **MIXED** | Valid and informative, but neither the full supported conjunction nor the negative rule holds; for example, NLL transfers but generation quality does not. |
| **INCONCLUSIVE** | Any leakage, split/hash/implementation invalidity, missing arm, timeout/OOM/NaN, post-freeze deviation, insufficient length strata, or no-floor/informativeness failure. Infrastructure failure is not a negative scientific result. |

Do not replace a failed seed, remove an arm, relax a margin, add a seed,
select a checkpoint, or report only favorable categories. The conclusion is
bounded to this corpus, base model, tokenizer, renderer, LoRA configuration,
optimizer, and 3-seed budget. It does not establish universal transfer,
general data-selection theory, or broad model generalization.

## 8. Required evidence package and stopping rule

For each run preserve the command, environment/interpreter receipt, protocol/
data/trainer hashes, seed, arm, update count, loss log, adapter hash, memory/
time receipt, and status. Preserve raw per-record validation/test metrics and
generations; aggregate tables are derived, not the only evidence.

The project stops after the protocol is frozen and the verification checklist
passes, or after a specific design blocker is recorded. It does not load Qwen
or launch training as part of this protocol-design project. A later model run
must stop after the predeclared six-arm evaluation and classification; a
follow-up protocol is needed for any changed corpus, model, metric, or budget.

### Read-only provenance inspected

- Initiative brief and constraints for `lora-natural-data-selection-20260810`.
- Prior synthetic protocol/result:
  `initiatives/lora-loss-normalization-20260809/projects/validation-selection-v1/`.
- Prior public protocol/result:
  `/Users/vukrosic/my-life/research-repos/qwen-lora-loss-normalization/`.

These locations were not modified.
