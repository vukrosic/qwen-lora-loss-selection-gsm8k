# Validation-guided LoRA loss selection on bounded GSM8K

Status: **inspected research artifact**

This repository contains the complete local evidence for a three-seed test of
whether validation-guided selection between token-mean and example-mean LoRA
training transfers from a prior synthetic task to a small natural instruction
setting.

The answer is bounded and mixed: the validation selector lost the primary
final-test macro-example NLL comparison on all three seeds, while complementary
GSM8K final-number exact match favored the selected arm on two seeds. The
frozen protocol therefore does not support bounded transfer and classifies the
result as `MIXED` rather than `NEGATIVE` because the endpoint directions
conflict.

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
