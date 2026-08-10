# Frozen result: natural-data validation selection

Status: **FINAL LOCAL RESULT — INSPECTED — NOT PUBLISHED**  
Protocol: `N1-GSM8K-A1`  
Classification: **MIXED; SUPPORTED BOUNDED TRANSFER NOT SHOWN**

## Claim

On the locked <=256-token GSM8K subset, validation-guided selection between
token-mean and example-mean LoRA training did not reproduce the prior synthetic
selection benefit. The common validation token-NLL selector chose example,
example, and token across the three fresh seeds, but the selected arm had worse
primary final-test macro-example NLL on all three seeds.

The complementary generated final-number endpoint moved in the opposite
direction on the first two seeds: the selected arm improved exact match on two
of three seeds, then lost on the third. Under the frozen conjunction this is
Mixed, not Supported and not Negative.

## Per-seed result

Lower validation NLL and lower final macro-example NLL are better. Positive
primary deltas mean the selected arm was worse. Positive exact deltas mean the
selected arm was better.

| Seed | Selected | Validation NLL token / example | Selected / unselected primary macro NLL | Primary delta | Selected / unselected exact | Exact delta |
| ---: | --- | --- | --- | ---: | --- | ---: |
| 20260820 | example | 0.942615341 / 0.941988786 | 0.911970814 / 0.910245831 | +0.001724983 | 0.06640625 / 0.03515625 | +0.03125 |
| 20260821 | example | 0.952068546 / 0.948560753 | 0.916401164 / 0.912764300 | +0.003636865 | 0.046875 / 0.03515625 | +0.01171875 |
| 20260822 | token | 0.951179021 / 0.953015239 | 0.926269400 / 0.924231491 | +0.002037909 | 0.03515625 / 0.041015625 | -0.005859375 |

The seed-20260822 diagnostic token-weighted final-test NLL delta is
`+0.0010099213948313`; it is not the frozen primary endpoint.

## Policy aggregates

| Policy | Primary macro NLL mean | Primary macro NLL worst seed | Exact mean | Exact worst seed |
| --- | ---: | ---: | ---: | ---: |
| Validation selected | 0.918213793 | 0.926269400 | 0.049479167 | 0.03515625 |
| Always token | 0.916426510 | 0.926269400 | 0.035156250 | 0.03515625 |
| Always example | 0.917534490 | 0.924231491 | 0.051432292 | 0.041015625 |

Selection lost the primary endpoint on 3/3 seeds. Its primary mean and worst
seed did not beat either fixed policy. Complementary exact match was no worse
on 2/3 seeds and strictly better on two, so exact evidence reverses the primary
direction under the protocol's gate. That conflict forces `MIXED`.

## Validity and informativeness

- All six training arms exited 0 at exactly 576 updates with finite losses.
- All six common validations evaluated 256 records; all six final tests
  evaluated 512 teacher-forced records and generated 512 answers.
- Every adapter, protocol, data lock, split, validation metric, selection, and
  test fingerprint matched.
- Each immutable selection timestamp preceded both final-test timestamps and
  recorded `NOT_STARTED` at seal time.
- The V2 lock passed zero-truncation, duplicate, near-duplicate, and frozen
  question-skeleton checks; the implementation preflight verified identical
  no-replacement natural record order without length-balanced batches.
- Primary NLL and complementary exact match both passed the frozen
  informativeness gates.
- Peak MLX memory was 2.313 GB in training and 1.756 GB in common validation;
  the process ledger records serialized claims/releases and preserved headroom.
- The model slot was explicitly released after the last test.

`RESULT.json` is the machine-verifiable result. Running `analyze_results.py`
rechecks the evidence and regenerates it.

## Limitations

- This is one fixed 1,000/256/512 subset of public GSM8K `main`, restricted to
  complete rendered examples of at most 256 tokens.
- It tests Qwen3-0.6B 3-bit, one LoRA recipe, one update budget, and three paired
  seeds on one M4/16GB machine.
- Public-benchmark exposure or pretraining contamination cannot be excluded;
  the result is only within-dataset transfer.
- Final-number exact match is complementary, not independent, and does not
  assess rationale correctness. Exact rates are low (3.52%–6.64%), and marker
  rates are below one (87.30%–93.16%).
- The experiment does not establish broad natural-instruction transfer, a
  universal preference between loss normalizations, or behavior on larger
  models, other datasets, or other budgets.

