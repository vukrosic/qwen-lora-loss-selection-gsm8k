# Validation loss was not a reliable LoRA guide on GSM8K

Status: **inspected research artifact**

We tested whether validation loss could choose between token-mean and
example-mean LoRA training for Qwen3-0.6B.

It did not give a dependable answer. The chosen model had worse primary test
NLL on all three seeds, although it answered more final numbers correctly on
two seeds.

**Practical takeaway:** do not treat one validation-loss number as proof that
a LoRA training objective is better. Check the final metric you actually care
about and preserve a held-out test.

The result is called `MIXED` because the two test measures point in different
directions—not because the experiment was unfinished.

See `RESULT.md` for the inspected result and limitations, `PROTOCOL.md` for the
frozen decision rules, and `RESULT.json` for machine-readable evidence.

## Recheck the evidence

With Python 3.12:

```bash
python analyze_results.py
```

The verifier checks all training, validation, selection, and test receipts;
adapter and data hashes; record counts and ordering; seal-before-test times;
frozen gates; and policy aggregates. It rewrites `RESULT.json` only after all
checks pass.

## Contents

- `PROTOCOL.md`: sole canonical pre-outcome protocol.
- `DATA-DESIGN.md`, `DATASET-LOCK-v*.json`, `SKELETON-AUDIT.json`: data and
  leakage controls, including the preserved V1 length failure and accepted V2.
- `data/v2/` and `source/`: locked subset and pinned upstream GSM8K source.
- `runs/`: all six adapters, receipts, selections, metrics, per-record
  teacher-forced losses, and generated answers. Redundant numbered adapter
  checkpoint copies are intentionally omitted.
- `preflight/`: tokenizer, mask, ordering, selector, and implementation checks.
- Python files in the root: preparation, training, evaluation, sealing, and
  analysis code.
- `background/`: preserved explorer recommendations and reviews; these are
  context, not competing canonical protocols.
- `MANIFEST.sha256`: integrity manifest for the packaged files.

The base model is not redistributed. Full reruns require a local
Qwen3-0.6B-3bit model path supplied to the recorded commands. Exact process
arguments and the original environment paths are preserved in each
`runs/*/receipt.json` for provenance.

No software license grant is supplied. See `LICENSE-NOTICE.md`.
