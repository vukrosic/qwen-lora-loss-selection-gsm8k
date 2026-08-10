# Protocol N1 critic: GSM8K candidate

Review date: 2026-08-10 (Asia/Singapore)  
Scope: read-only critique of `PROTOCOL.md`; no Qwen/tokenizer/model was
loaded, and no sibling project was edited.

## Verdict

GSM8K is a plausible bounded natural-language math corpus, but Protocol N1 is
not ready to run against it unchanged. The generic split, synthetic
short/long assumptions in the existing evaluator, one-percent truncation
allowance, 128-token generation cap, and the phrase **independent structured
endpoint** are not defensible for GSM8K. The amendments below are conditional:
they apply only if the dataset audit locks GSM8K. N1 remains unchanged for a
different corpus unless its own audit exposes the same issue.

The official source describes human-written problems with a `question` and
multi-step `answer`; the final numeric solution follows `####` and calculation
annotations may appear in the answer. The official source also provides a
parser, while the current Hugging Face card records 7,473 train and 1,319 test
examples under MIT. See the [archived OpenAI repository](https://github.com/openai/grade-school-math),
its [official extraction code](https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py),
and the [versioned dataset card](https://huggingface.co/datasets/openai/gsm8k).

## Findings and justified amendments

### 1. Preserve the official holdout; do not make a new 70/15/15 split

**Finding.** N1's generic 70/15/15 split would either merge the public GSM8K
train and test partitions or create a new test by sampling the public train
partition. Both choices discard a documented benchmark holdout and make it
easier to accidentally expose test answers during split construction.

**Amendment G1 (required).** When the source is GSM8K, lock one exact upstream
commit/revision and use:

* official `train` only for development, deterministically splitting it into
  train (85%) and validation (15%); and
* the complete official `test` partition (1,319 records in the cited card) as
  the sealed final test.

Do not merge, reshuffle, or subsample the official test after seeing any
metric. The train-to-validation assignment must use the frozen hash seed from
N1, but assign **question groups**, not individual rows. If the audit instead
locks a preselected GSM8K subset, its IDs and selection rule must be frozen
before any answer-level metric is read; no outcome-driven length or difficulty
subsampling is allowed.

This is still not a clean contamination-free benchmark: GSM8K is public and
the Qwen pretraining mixture is not fully auditable here. The result may show
within-corpus post-training behavior, not novel knowledge acquisition.

### 2. Detect GSM8K template leakage beyond 5-shingle similarity

**Finding.** N1's token 5-shingle Jaccard threshold can miss the same word
problem template with different numbers, names, or currency. GSM8K's numerical
answers and calculator annotations make that failure mode especially plausible.

**Amendment G2 (required).** In addition to N1's exact and near-duplicate
checks, derive a question-only skeleton before splitting:

1. NFKC-normalize, lowercase, and collapse whitespace;
2. replace signed integers, decimals, percentages, and currency amounts with
   `<NUM>` (do not use the answer field); and
3. hash the resulting skeleton and keep each identical-skeleton component in
   one partition.

Audit skeleton overlap across official train, derived validation, and official
test. A nonzero cross-partition overlap is preserved in the manifest and makes
the clean-transfer claim unavailable; it is not repaired by moving a test
record after training starts. This deterministic check is a leakage guard, not
an outcome metric.

### 3. Remove synthetic `short`/`long` balancing assumptions

**Finding.** The existing implementation's `MixedDataset`, `BalancedMixedIterator`,
and aggregate code expect equal `short` and `long` kinds. GSM8K has `question`
and `answer`, not those labels. Reusing the iterator would either fail or
silently oversample length strata, changing the natural data distribution and
making the normalization comparison a length-balancing experiment.

**Amendment G3 (required).** The GSM8K preflight must use a dataset adapter
with stable `id`, `question`, and `answer` fields and a single deterministic
shuffle stream per seed. Both loss arms consume the identical ordered batches;
there is no short/long oversampling, equal-stratum quota, or answer-length
reweighting. Report completion-length and step-count strata only for locked
diagnostics. A run that still depends on the synthetic `kind` contract is a
preflight failure, not a natural-data result.

### 4. Pin answer extraction and prevent parser gaming

**Finding.** Full-string generation exact match in N1 is not meaningful for
GSM8K: equivalent solutions can use different wording, and the documented
task endpoint is the final numeric answer. Conversely, a loose parser can give
credit to a model that emits a bare `####` answer or multiple conflicting
markers without a valid solution.

**Amendment G4 (required).** Freeze a parser module and hash it in the dataset
lock. Its behavior is:

* Ground truth must contain exactly one valid `####` numeric marker; reject the
  record during audit if not. Strip commas exactly as the official parser does,
  preserve sign and decimal spelling, and do not apply arithmetic equivalence,
  rounding, units, or a second answer normalization.
* A generated completion receives final-answer exact credit only when it has
  exactly one valid marker and one numeric value under the same grammar.
  Missing, malformed, or multiple markers are incorrect and separately counted
  as `invalid_or_ambiguous`.
* The parser is applied after the sealed selection record. Test answer strings,
  parsed values, and parser diagnostics are unavailable to training,
  validation selection, or debugging. The raw generation is retained.

Replace N1's full-completion exact match for GSM8K with `final_answer_exact`;
retain full text only as raw evidence. Report marker-validity rate and answer
exact rate separately. Do not use a validation answer score to tune a regex or
generation cap.

### 5. Treat length as a possible mechanism confound

**Finding.** On GSM8K, completion length is plausibly correlated with number of
calculation steps and difficulty. Token-mean training gives long solutions more
aggregate gradient weight; example-mean training gives each problem equal
weight. A token-weighted validation selector can therefore select an arm for a
length-composition advantage rather than a generally better solver.

**Amendment G5 (required).** Keep N1's token-weighted validation selector (it is
the predeclared mechanism under test), but add a frozen confound check:

* derive four completion-token-length cut points from train plus validation
  only, apply them unchanged to test, and require at least 20 test records in
  every nonempty bin;
* report macro-example NLL and final-answer exact rate in each bin, plus a
  length-standardized NLL equal to the unweighted mean of the four bin means;
* count a selected-policy win as robust only when its length-standardized NLL
  is lower than the unselected arm on at least 2 of 3 seeds and the aggregate
  win is not solely a Q4/longest-bin win; and
* report the number of `<<...>>` calculation annotations (a fixed step-count
  proxy) by bin, but never use test step counts to select an arm.

If the ordinary macro-example NLL improves while the standardized result loses
on at least three of four bins, classify the outcome as mixed rather than
positive. This amendment exposes the weighting tradeoff without changing the
selector after outcomes.

### 6. The GSM8K answer endpoint is complementary, not genuinely independent

**Finding.** Final numeric exact match is generated from the same held-out
question/answer pair as teacher-forced NLL. It is a distinct measurement
(greedy decoding, marker parsing, and a nonlinear correctness indicator), but
it is not an independent construct: both endpoints measure the same math
answer, and a model can print a correct number without a valid derivation.
Calling it an “independent structured-response endpoint” would overstate the
evidence.

**Amendment G6 (required).** For GSM8K rename the endpoint to
`complementary_final_answer_exact`; remove the word “independent” from the
GSM8K classification rule. A positive label, if earned, must be described as
**supported bounded GSM8K transfer with a complementary answer endpoint**, not
as broad natural-instruction transfer. Require both:

1. the N1 primary and length-standardized NLL gates; and
2. no selected-policy loss in final-answer exact rate on at least 2 of 3 seeds,
   with a strict win on at least one.

If the initiative requires a genuinely independent construct, GSM8K alone is
insufficient; a separately locked, non-overlapping task corpus or human-rated
endpoint would be needed. Adding one after seeing GSM8K outcomes is not
allowed.

### 7. Correct the truncation and generation-cap contract

**Finding.** N1 permits up to 1% truncation, but the current trainer's
`validate_dataset` rejects **any** record longer than `max_seq_length`. GSM8K's
multi-step answers also make the fixed 128 generated-token cap unsafe for the
`####` marker. Changing the cap after observing generations would be test
tuning.

**Amendment G7 (required).** Before model launch, run tokenizer-only length
audit on all locked partitions and freeze the smallest `L` in
`{256, 384, 512}` that has zero formatted-record truncation. If no candidate
has zero truncation, reject GSM8K for this bounded protocol. Do not retain N1's
one-percent allowance unless the trainer is separately versioned and proven to
implement the same declared truncation policy.

Freeze `max_generation_tokens = 256` for GSM8K. If more than 1% of train or
validation reference markers occur beyond token 240, or if the sealed test
audit shows the cap prevents evaluating the marker on more than 1% of records,
the answer endpoint is unavailable and cannot support a positive label. NLL
may still be reported as a bounded predictive result, subject to G6's scope.

### 8. Add a real M4 feasibility gate and remove redundant base repeats

**Finding.** Prior local receipts are encouraging but do not establish GSM8K
feasibility. The six earlier 576-update runs used max length 128 and reported
roughly 151.6–168.1 seconds each with peak MLX memory about 0.831 GB and cache
about 0.634–0.636 GB. Their 96-record, 24-token evaluations took about
19.1–21.4 seconds each. A linear planning extrapolation to all 1,319 official
test records is about 25 minutes per arm at 128 generated tokens, or about
2.5 hours for six arms; at 256 tokens it is roughly 5 hours. This is only a
planning estimate, not GSM8K evidence, and serial model reloads, longer
sequences, and thermal throttling may increase it.

**Amendment G8 (required).** Before confirmation, perform one non-confirming
pilot using train/validation records only at the frozen `L` and 256-token
generation setting. Record process RSS, MLX active/cache/peak memory, wall
time, and swap/compression. Proceed only if projected total work (six train
runs, six validation selections, six paired test evaluations, and one base
diagnostic pass) fits the remaining local window and process RSS stays below
80% of 16 GB (approximately 12.8 GB) with roughly 20% headroom and no rapidly
rising swap. A failed pilot is a design/resource blocker; do not shrink the
test or lower the cap after outcomes.

The base model has no seed-dependent training state. Replace N1's three base
diagnostics with one frozen base validation/test pass (or one pass per split
if implementation requires it); never count repeated base evaluations as
independent seeds.

## GSM8K-specific preflight stop condition

Before any Qwen launch, all of G1–G8 must be evidenced in the lock/preflight:
source revision and license, official split preservation, skeleton-overlap
audit, adapter schema, zero-truncation `L`, parser hash and extraction counts,
length-standardization bins, answer-marker cap audit, and M4 pilot receipt.
Failure of any item stops the natural-data claim. Partial work remains
diagnostic evidence and is not silently folded into N1.

