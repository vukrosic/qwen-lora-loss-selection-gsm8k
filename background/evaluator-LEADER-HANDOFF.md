# Leader handoff: natural evaluator protocol

Status: **protocol recommendation ready; no model or tokenizer loaded**

## Recommendation

Use Protocol N1 in `PROTOCOL.md` after the dataset-audit lane writes an
immutable dataset lock. The strongest anti-leakage choice is to select on
final validation token-weighted completion NLL while making the confirmatory
test endpoint macro-example NLL and requiring a separate structured-response
endpoint for any positive natural-instruction claim. This avoids treating the
same validation statistic as a tautological test of itself.

## Decisive frozen choices

- Group/near-duplicate-safe 70/15/15 split, hash seed `20260810`; test is
  untouched until per-seed selection JSON is sealed.
- Two arms differ only in token-mean versus example-mean supervised completion
  loss; fixed constants, 576 updates, batch 4, max length 256, no early stop.
- Fresh paired seeds `20260820`, `20260821`, `20260822`, with counterbalanced
  arm order.
- Selection: lower canonical validation token-weighted NLL, ties to token.
- Primary test endpoint: macro-example NLL; independent structured/format
  endpoint is mandatory for a positive transfer label.
- Controls: both fixed policies from the same paired runs plus a diagnostic
  base-model evaluation; raw per-example outputs and hashes required.
- Three valid paired seeds and no-floor/no-leakage/resource gates are required;
  failures are inconclusive, not silently repaired by changing the protocol.

## Why this is not a copied synthetic gate

Natural instruction responses are open-ended and generally lack a reliable
universal exact-match target. The protocol therefore makes exact/format
evaluation conditional on a locked deterministic subset, treats NLL as the
primary predictive endpoint, and refuses a broad claim when only NLL is
available. Length and task-family stratification remain diagnostics rather
than invented synthetic skills.

## Blocking prerequisite

Do not launch Qwen/MLX until the sibling dataset audit supplies a corpus
version, provenance/license, split and tokenizer hashes, length summary,
structured-subset decision, and truncation check. Then the implementation lane
must pass model-free mask/aggregation and receipt preflight against this
protocol.

