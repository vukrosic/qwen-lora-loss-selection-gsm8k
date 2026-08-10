# Bounded post-hoc metric-conflict diagnosis

Status: **evidence-only descriptive follow-up; primary classification remains MIXED**

This note localizes the observed disagreement using only the packaged
per-record final-test evidence. It does not alter the frozen selector,
endpoints, or primary result, and it does not identify a causal mechanism.
Positive NLL deltas mean the selected arm is worse; positive exact and marker
deltas mean it is better.

| Seed | Selected | Overall macro-NLL delta | Short delta | Long delta | Exact delta | Marker delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20260820 | example | +0.001724983 | +0.010534841 | -0.007084875 | +0.031250000 | +0.000000000 |
| 20260821 | example | +0.003636865 | +0.006799788 | +0.000473942 | +0.011718750 | +0.013671875 |
| 20260822 | token | +0.002037909 | +0.005872222 | -0.001796404 | -0.005859375 | +0.009765625 |

The selected arm's macro-example NLL is worse in the locked short stratum on
all three seeds. The long stratum offsets that loss on seeds 20260820 and
20260822 and is nearly tied in the opposite direction on seed 20260821. Thus,
within this finite split, the 3/3 aggregate primary loss is concentrated in the
short stratum rather than shared uniformly across both length strata. This is
localization, not evidence that length caused the result.

Exact-answer differences come from sparse paired flips. Relative to the
unselected arm, the selected arm has net exact gains of 16/512 and 6/512
records on the first two seeds, then a net loss of 3/512 on the third. Marker
presence does not resolve the conflict: it is tied when exact improves most
(seed 20260820), while it improves by 5/512 records when exact worsens (seed
20260822). Teacher-forced NLL and generated final-number exact match therefore
capture different observed aspects of these six fitted arms.

Paired within-seed bootstrap intervals for the overall primary deltas all
include zero. They describe sensitivity to resampling these 512 locked records,
not population uncertainty. The evidence remains limited to one <=256-token
GSM8K subset, one Qwen3-0.6B 3-bit LoRA recipe, and three paired seeds. It does
not support a new selector, a causal explanation, or generalization to other
tasks, models, or budgets.

